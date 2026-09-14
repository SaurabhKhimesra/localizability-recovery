"""Thin wrapper tying a generated world to a MuJoCo model, plus the two platforms.

Motion is kinematic: the sensor pose is prescribed, not integrated from forces.
The physics engine is used only for its ray caster and its scene graph.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from .lidar import Lidar, LidarSpec, Scan
from .se3 import euler_zyx, make_T
from .landmarks import beam_columns, fit_strip_centre
from .worlds import TunnelWorld, WorldSpec, build_world

__all__ = ["TunnelSim", "MarkerDetection", "PlatformSpec", "UGV", "Drone"]

MARKER_PARK_Z = -50.0


@dataclasses.dataclass
class MarkerDetection:
    """One marker seen in one scan."""

    slot: int
    point_sensor: np.ndarray
    """Centroid of this marker's returns, in the sensor frame."""
    n_beams: int
    range_m: float
    seen_width_m: float | None = None
    """How much of the strip's width the returns actually covered. ``None`` means
    the detector did not fit the strip, in which case the measurement model cannot
    charge for foreshortening."""
    n_columns: int | None = None
    """Distinct beam azimuths among the returns. One column means the strip was crossed by a
    single column of beams, and then where it sits between two beams is not observed: the fit has
    to guess, and it guesses the same way every time. Reported because it is what the estimator
    needs to know how far to trust the fitted point (``landmarks.strip_fit_bias``)."""


class TunnelSim:
    """A compiled tunnel world with marker slots and a LiDAR."""

    def __init__(
        self,
        seed: int,
        world_spec: WorldSpec | None = None,
        lidar_spec: LidarSpec | None = None,
        sensor_seed: int | None = None,
    ):
        self.world: TunnelWorld = build_world(seed, world_spec)
        self.model = mujoco.MjModel.from_xml_string(self.world.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.lidar = Lidar(lidar_spec or LidarSpec(), seed=seed if sensor_seed is None else sensor_seed)

        self._marker_geom_ids: list[int] = []
        self._marker_mocap_ids: list[int] = []
        for i in range(self.world.spec.n_marker_slots):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"marker_{i}")
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"marker_body_{i}")
            self._marker_geom_ids.append(int(gid))
            self._marker_mocap_ids.append(int(self.model.body_mocapid[bid]))
        self._marker_geom_id_set = set(self._marker_geom_ids)
        self.n_markers_placed = 0

    # ---- markers --------------------------------------------------------

    @property
    def marker_capacity(self) -> int:
        return len(self._marker_geom_ids)

    def place_marker(self, position: np.ndarray, yaw: float = 0.0) -> int:
        """Move the next free marker slot to ``position`` with its face normal at ``yaw``.

        The strip's local x axis is its thickness, so ``yaw`` should point away from
        the wall it is mounted on.
        """
        i = self.n_markers_placed
        if i >= self.marker_capacity:
            raise RuntimeError(
                f"out of marker slots ({self.marker_capacity}); raise WorldSpec.n_marker_slots"
            )
        mid = self._marker_mocap_ids[i]
        self.data.mocap_pos[mid] = np.asarray(position, dtype=float)
        self.data.mocap_quat[mid] = np.array(
            [np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=float
        )
        self.n_markers_placed = i + 1
        mujoco.mj_forward(self.model, self.data)
        return i

    def drop_marker_on_wall(
        self, T_world_sensor: np.ndarray, side: str = "auto", height: float = 1.2
    ) -> tuple[int, np.ndarray]:
        """Mount a strip on the tunnel wall beside the sensor.

        The robot measures the wall with a single sideways ray, so the offset it
        believes it used is the offset it actually gets. Returns the slot index and
        the mounting point expressed in the sensor frame, which is what the
        estimator registers the landmark from.
        """
        R = T_world_sensor[:3, :3]
        origin = np.ascontiguousarray(T_world_sensor[:3, 3], dtype=float)
        best = None
        sides = ("left", "right") if side == "auto" else (side,)
        for s in sides:
            sign = 1.0 if s == "left" else -1.0
            direction = np.ascontiguousarray(R @ np.array([0.0, sign, 0.0]), dtype=float)
            gid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(self.model, self.data, origin, direction, None, 1, -1, gid)
            if dist < 0:
                continue
            if best is None or dist < best[0]:
                best = (float(dist), direction, sign)
        if best is None:
            raise RuntimeError("no wall found to mount a marker on")

        dist, direction, _ = best
        spec = self.world.spec
        inset = spec.marker_thickness * 0.5 + 1e-3
        point_world = origin + direction * (dist - inset)
        point_world[2] = height
        yaw = float(np.arctan2(-direction[1], -direction[0]))
        slot = self.place_marker(point_world, yaw)
        offset_sensor = R.T @ (point_world - origin)
        return slot, offset_sensor

    def marker_positions(self) -> np.ndarray:
        idx = self._marker_mocap_ids[: self.n_markers_placed]
        if not idx:
            return np.zeros((0, 3))
        return np.array(self.data.mocap_pos[idx], dtype=float)

    def is_marker_geom(self, geom_ids: np.ndarray) -> np.ndarray:
        """Boolean mask over a scan's geom ids: True where the return is a marker.

        Detection by geom id stands in for a retroreflective intensity return.
        """
        if geom_ids.size == 0:
            return np.zeros(0, dtype=bool)
        return np.isin(geom_ids, self._marker_geom_ids[: self.n_markers_placed])

    def marker_detections(self, scan: Scan, fit_strip: bool = True) -> list["MarkerDetection"]:
        """Group a scan's marker returns by slot.

        Detection is by geom id, which assumes perfect classification of a
        retroreflective return against the tunnel rock. That is optimistic: in dust,
        water spray or with a dirty target, intensity thresholding both misses
        strips and fires on wet rock, and no data association layer is modelled
        here. A single return counts as a detection, which is the point of the
        strip: one beam gives a range and a bearing.
        """
        if scan.geom_ids.size == 0:
            return []
        out: list[MarkerDetection] = []
        active = self._marker_geom_ids[: self.n_markers_placed]
        spec = self.world.spec
        for slot, gid in enumerate(active):
            mask = scan.geom_ids == gid
            n = int(mask.sum())
            if n == 0:
                continue
            # The centroid of the returns is not the strip's mounting point: at
            # grazing incidence a strip of known width hides its own far half, so
            # the returns pile up on the near edge. That bias reached 3 cm at 4.3 m
            # against a modelled 1.7 cm standard deviation, and it is what leaked
            # the marker benefit out of the pipeline. ``fit_strip_centre`` uses the
            # known width, thickness and beam spacing instead.
            if fit_strip:
                point, seen = fit_strip_centre(
                    scan.points[mask], self.lidar.spec, spec.marker_width,
                    spec.marker_thickness,
                )
            else:
                point, seen = scan.points[mask].mean(axis=0), None
            out.append(
                MarkerDetection(
                    slot=slot,
                    point_sensor=point,
                    n_beams=n,
                    range_m=float(np.linalg.norm(point)),
                    seen_width_m=seen,
                    n_columns=beam_columns(scan.points[mask], self.lidar.spec),
                )
            )
        return out

    # ---- sensing --------------------------------------------------------

    def scan(self, T_world_sensor: np.ndarray) -> Scan:
        return self.lidar.scan(self.model, self.data, T_world_sensor)


# ---------------------------------------------------------------------------
# platforms
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PlatformSpec:
    sensor_height: float = 0.7
    step_length: float = 0.5
    """Arclength travelled between scans, metres."""
    speed_mps: float = 1.0
    """Forward speed. With ``step_length`` it fixes the scan interval used by the
    motion prior, so the prior's physical noise units mean what they say."""
    yaw_rate_limit_dps: float = 45.0
    """Sensor slew limit, degrees per second, for platforms that can steer the
    sensor independently of the direction of travel."""

    @property
    def max_yaw_step(self) -> float:
        """Slew limit per scan, radians."""
        dt = self.step_length / max(self.speed_mps, 1e-9)
        return float(np.deg2rad(self.yaw_rate_limit_dps) * dt)


class UGV:
    """Sensor yaw is locked to the direction of travel. A 360 deg scanner does not
    care, which is exactly why the UGV story needs markers rather than yaw."""

    def __init__(self, world: TunnelWorld, spec: PlatformSpec = PlatformSpec()):
        self.world = world
        self.spec = spec
        self.s = 0.0

    def pose(self) -> np.ndarray:
        p, yaw = self.world.pose_at(self.s, self.spec.sensor_height)
        return make_T(euler_zyx(yaw), p)

    def advance(self) -> float:
        self.s = min(self.s + self.spec.step_length, self.world.total_length)
        return self.s

    @property
    def finished(self) -> bool:
        return self.s >= self.world.total_length - 1e-9

    @property
    def path_length(self) -> float:
        return self.s


class Drone:
    """Sensor yaw is decoupled from velocity; the flown path is the centreline but
    the scanner may be commanded to look elsewhere, at a cost in yaw rate."""

    def __init__(self, world: TunnelWorld, spec: PlatformSpec = PlatformSpec()):
        self.world = world
        self.spec = spec
        self.s = 0.0
        _, yaw0 = world.pose_at(0.0)
        self.yaw = float(yaw0)

    def pose(self) -> np.ndarray:
        p, _ = self.world.pose_at(self.s, self.spec.sensor_height)
        return make_T(euler_zyx(self.yaw), p)

    def command_yaw(self, yaw_target: float) -> float:
        """Slew the sensor yaw toward the target, respecting the rate limit."""
        err = np.arctan2(np.sin(yaw_target - self.yaw), np.cos(yaw_target - self.yaw))
        step = float(np.clip(err, -self.spec.max_yaw_step, self.spec.max_yaw_step))
        self.yaw += step
        return abs(step)

    def track_yaw(self) -> float:
        """The heading the velocity is following, which forward gaze locks to."""
        _, yaw = self.world.pose_at(self.s)
        return float(yaw)

    def track_heading(self) -> float:
        return self.command_yaw(self.track_yaw())

    def advance(self) -> float:
        self.s = min(self.s + self.spec.step_length, self.world.total_length)
        return self.s

    @property
    def finished(self) -> bool:
        return self.s >= self.world.total_length - 1e-9

    @property
    def path_length(self) -> float:
        return self.s
