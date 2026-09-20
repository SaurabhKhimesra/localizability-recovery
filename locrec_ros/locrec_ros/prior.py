"""Pair each scan with the odometry pose at its own stamp, with no ROS import.

A scan's prior is the odometry at that scan's stamp, and "the latest odometry
message" is not that. The simulator publishes a scan before the odometry that
shares its stamp, so the latest is either one scan stale or racing the scan
depending on which callback runs first, and a real driver promises no order at
all. So a scan waits here until the odometry history reaches its stamp. An exact
stamp match is used as it is; otherwise the pose is interpolated between the two
samples either side of the stamp, linearly in translation and along the shortest
rotation. Nothing is extrapolated, and a scan that cannot be paired is dropped and
counted rather than passed on without a prior.

A scan can also be made to wait for a side message that carries exactly its stamp,
which is how the marker report reaches the estimator. That join is exact rather
than interpolated, because a report is about one scan and no other.
"""
from __future__ import annotations

import bisect
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["Paired", "PriorPairer", "interpolate_pose"]


class Paired(NamedTuple):
    """A scan that is ready to be processed."""

    stamp: int
    payload: object
    pose: np.ndarray
    side: object = None
    """The side message stamped exactly like the scan, when one is required."""


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """The pose a fraction ``alpha`` of the way from ``T0`` to ``T1``."""
    R0 = T0[:3, :3]
    rotvec = Rotation.from_matrix(R0.T @ T1[:3, :3]).as_rotvec()
    T = np.eye(4)
    T[:3, :3] = R0 @ Rotation.from_rotvec(alpha * rotvec).as_matrix()
    T[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    return T


class PriorPairer:
    """Holds scans until the odometry reaches their stamps, and releases them in order.

    Stamps are integer nanoseconds. Every ``add_`` method returns the scans that
    became ready, as ``Paired`` in stamp order, so the caller handles whatever comes
    back whichever message arrived first.

    With ``require_side`` a scan also waits for ``add_side`` to deliver a message
    with exactly its stamp. A scan whose side message can no longer come, because a
    later one already has, is dropped and counted like any other unpairable scan.
    """

    def __init__(
        self, max_pending: int = 10, max_history: int = 4096, require_side: bool = False
    ):
        self.require_side = bool(require_side)
        self._side: dict[int, object] = {}
        self._newest_side: int | None = None
        self.dropped_no_side = 0
        """Scans whose side message never came although a later one did."""
        self.max_pending = int(max_pending)
        """Scans held waiting for odometry. Beyond this the oldest is dropped."""
        self.max_history = int(max_history)
        """Odometry samples kept when no scan needs them, as a bound on memory."""
        self._odom_stamps: list[int] = []
        self._odom_poses: list[np.ndarray] = []
        self._pending: list[tuple[int, object]] = []
        self._last_released: int | None = None
        self.n_odometry = 0
        self.dropped_stale = 0
        """Scans older than every odometry sample still held."""
        self.dropped_overflow = 0
        """Scans pushed out by newer ones while waiting for odometry."""
        self.dropped_out_of_order = 0
        """Scans stamped at or before a scan already released."""

    @property
    def waiting(self) -> int:
        return len(self._pending)

    @property
    def dropped(self) -> int:
        return (
            self.dropped_stale
            + self.dropped_overflow
            + self.dropped_out_of_order
            + self.dropped_no_side
        )

    def add_side(self, stamp: int, payload: object) -> list[Paired]:
        stamp = int(stamp)
        self._side[stamp] = payload
        if self._newest_side is None or stamp > self._newest_side:
            self._newest_side = stamp
        return self._release()

    def add_odometry(self, stamp: int, pose: np.ndarray) -> list[Paired]:
        stamp = int(stamp)
        self.n_odometry += 1
        i = bisect.bisect_left(self._odom_stamps, stamp)
        if i < len(self._odom_stamps) and self._odom_stamps[i] == stamp:
            self._odom_poses[i] = np.array(pose, dtype=float)
        else:
            self._odom_stamps.insert(i, stamp)
            self._odom_poses.insert(i, np.array(pose, dtype=float))
        return self._release()

    def add_scan(self, stamp: int, payload: object) -> list[Paired]:
        stamp = int(stamp)
        if self._last_released is not None and stamp <= self._last_released:
            self.dropped_out_of_order += 1
            return []
        keys = [s for s, _ in self._pending]
        self._pending.insert(bisect.bisect_right(keys, stamp), (stamp, payload))
        while len(self._pending) > self.max_pending:
            self._pending.pop(0)
            self.dropped_overflow += 1
        return self._release()

    def _release(self) -> list[Paired]:
        out = []
        while self._pending:
            stamp, payload = self._pending[0]
            if not self._odom_stamps or self._odom_stamps[-1] < stamp:
                break  # the odometry has not reached this scan yet
            side = None
            if self.require_side:
                if stamp not in self._side:
                    if self._newest_side is None or self._newest_side < stamp:
                        break  # its side message may still be on the way
                    # a later one has arrived, so this scan's never will
                    self._pending.pop(0)
                    self.dropped_no_side += 1
                    continue
                side = self._side[stamp]
            self._pending.pop(0)
            if self._odom_stamps[0] > stamp:
                self.dropped_stale += 1
                continue
            out.append(Paired(stamp, payload, self._pose_at(stamp), side))
            self._last_released = stamp
        self._trim()
        return out

    def _pose_at(self, stamp: int) -> np.ndarray:
        i = bisect.bisect_left(self._odom_stamps, stamp)
        if self._odom_stamps[i] == stamp:
            return self._odom_poses[i].copy()
        t0, t1 = self._odom_stamps[i - 1], self._odom_stamps[i]
        return interpolate_pose(
            self._odom_poses[i - 1], self._odom_poses[i], (stamp - t0) / (t1 - t0)
        )

    def _trim(self) -> None:
        # every scan still to come is stamped after this, so only the last sample at
        # or before it is ever needed again
        floor = self._pending[0][0] if self._pending else self._last_released
        if floor is not None:
            j = bisect.bisect_right(self._odom_stamps, floor) - 1
            if j > 0:
                del self._odom_stamps[:j]
                del self._odom_poses[:j]
        excess = len(self._odom_stamps) - self.max_history
        if excess > 0:
            del self._odom_stamps[:excess]
            del self._odom_poses[:excess]
        # side messages at or before the floor belong to scans that are already gone
        if floor is not None:
            oldest_needed = self._pending[0][0] if self._pending else floor + 1
            for stamp in [s for s in self._side if s < oldest_needed]:
                del self._side[stamp]
        while len(self._side) > self.max_history:
            del self._side[min(self._side)]
