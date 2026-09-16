"""Where a limited-FOV scanner should look, and the policies that decide it.

The drone's sensor yaw is free of its velocity. That is the whole difference from
the UGV: it cannot leave localizability behind, but it can go and find some. The
question is where to point, and the answer has to be causal, so it comes from the
local map the estimator has already built rather than from the world.

A surface patch with unit normal ``n`` constrains translation along ``n`` and not
at all across it, so a set of visible patches contributes information
``sum n n^T``. Point the sensor where the smallest eigenvalue of that sum is
largest and you are pointing at the structure that fixes the direction you are
currently worst at. In a straight tunnel that is whatever breaks the symmetry:
a niche, a junction mouth, the far wall of a bend, the rib you passed ten metres
back. Pointing straight ahead sees only two parallel walls, which is the one thing
that tells you nothing about how far along them you are.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import small_gicp

from .lidar import LidarSpec
from .runner import StepContext

__all__ = [
    "map_normals",
    "predicted_information",
    "ForwardGaze",
    "SweepGaze",
    "GreedyGaze",
    "GlanceGaze",
    "OracleGaze",
]


def map_normals(points: np.ndarray, num_neighbors: int = 20, num_threads: int = 1):
    """Unit normals for the confirmed local map, by local PCA.

    ``small_gicp`` computes covariances on the downsampled source cloud during
    registration, not on the map, and does not expose them per map point through
    the Python bindings, so they are estimated here instead. At this map size it
    costs a few milliseconds, well under one percent of a step.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 8:
        return np.zeros((pts.shape[0], 3))
    cloud = small_gicp.PointCloud(pts)
    small_gicp.estimate_normals(cloud, num_neighbors=num_neighbors, num_threads=num_threads)
    n = np.asarray(cloud.normals())[:, :3]
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, norms, out=np.zeros_like(n), where=norms > 1e-9)


def predicted_information(
    points: np.ndarray,
    normals: np.ndarray,
    sensor_position: np.ndarray,
    yaw: float,
    spec: LidarSpec,
) -> np.ndarray:
    """3x3 translational information the map would give from this yaw.

    Only the geometry is predicted, not the returns: a map point counts if it is
    inside the range and inside the horizontal FOV. Grazing incidence is charged
    for by weighting each patch with the cosine between its normal and the line of
    sight, because a wall seen edge on returns few beams and constrains little.
    """
    if points.shape[0] == 0:
        return np.zeros((3, 3))
    rel = points - np.asarray(sensor_position, dtype=float)
    dist = np.linalg.norm(rel, axis=1)
    in_range = (dist > spec.min_range) & (dist <= spec.max_range)
    if not in_range.any():
        return np.zeros((3, 3))

    rel = rel[in_range]
    dist = dist[in_range]
    nrm = normals[in_range]

    bearing = np.arctan2(rel[:, 1], rel[:, 0])
    delta = np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw))
    half_fov = np.deg2rad(spec.fov_azimuth_deg) * 0.5
    visible = np.abs(delta) <= half_fov
    if not visible.any():
        return np.zeros((3, 3))

    rel = rel[visible]
    dist = dist[visible]
    nrm = nrm[visible]
    los = rel / dist[:, None]
    weight = np.abs(np.einsum("ij,ij->i", nrm, los))
    return np.einsum("i,ij,ik->jk", weight, nrm, nrm)


class _Gaze:
    name = "gaze"
    needs_map = False

    def __call__(self, ctx: StepContext) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"policy": self.name}


@dataclasses.dataclass
class ForwardGaze(_Gaze):
    """Yaw locked to the direction of travel. The do-nothing baseline."""

    name: str = "forward"

    def __call__(self, ctx: StepContext) -> dict:
        return {"yaw_target": ctx.platform.track_yaw()}


@dataclasses.dataclass
class SweepGaze(_Gaze):
    """Yaw oscillates continuously, regardless of what the Hessian says.

    Dumb and strong: it costs yaw rate everywhere, but it does look sideways, so if
    most of the benefit comes from simply not staring down the tunnel then this
    captures it and the greedy policy has nothing to add. That is a result worth
    having either way.
    """

    amplitude_deg: float = 60.0
    period_m: float = 20.0
    name: str = "sweep"

    def __call__(self, ctx: StepContext) -> dict:
        phase = 2.0 * np.pi * ctx.estimated_distance / max(self.period_m, 1e-9)
        offset = np.deg2rad(self.amplitude_deg) * np.sin(phase)
        return {"yaw_target": ctx.platform.track_yaw() + offset}


@dataclasses.dataclass
class GreedyGaze(_Gaze):
    """Information-greedy gaze over the confirmed local map.

    Engages only when the live ratio says localizability has collapsed. Above the
    threshold it relaxes toward forward, because looking sideways when the geometry
    is already doing the work costs yaw rate for nothing.
    """

    ratio_threshold: float
    step_deg: float = 15.0
    cone_deg: float | None = None
    """Restrict candidates to this half-angle about the direction of travel.

    ``None`` is the original policy, which was measured to be four to ten times
    worse than looking forward. The reason is in ``docs/failures.md``: information
    predicted from the local map is maximised by looking at the local map, which is
    behind, so the sensor stops seeing the tunnel it is flying into. The cone is
    the repair, and it was chosen from that measurement rather than by search."""
    name: str = "greedy"
    needs_map: bool = True

    def __call__(self, ctx: StepContext) -> dict:
        loc = ctx.localizability
        forward = ctx.platform.track_yaw()
        if loc is None or loc.ratio >= self.ratio_threshold:
            return {"yaw_target": forward}

        points = ctx.local_map_points
        normals = ctx.local_map_normals
        if points is None or points.shape[0] < 8:
            return {"yaw_target": forward}

        return {"yaw_target": _best_yaw(
            points, normals, ctx.T_est[:3, 3], ctx.platform.yaw, ctx.sim.lidar.spec,
            self.step_deg, have=_have(loc), cone_deg=self.cone_deg, reference=forward,
        )}

    def describe(self) -> dict:
        return {
            "policy": self.name,
            "ratio_threshold": self.ratio_threshold,
            "cone_deg": self.cone_deg,
        }


@dataclasses.dataclass
class GlanceGaze(_Gaze):
    """Look back at the best structure behind, briefly, then return to forward.

    The forward-restricted greedy policy showed that inside a forward cone there is
    usually nothing better to look at than forward, and the unrestricted one showed
    that a sensor free to stare backwards will do exactly that and stop mapping the
    tunnel it is flying into. This is the bounded version of looking back: when
    localizability collapses and there is structure behind worth looking at, turn to
    it at the rate limit, hold for at most ``hold_s``, then return to forward and do
    not look again for ``cooldown_s``. Scans keep going into the map throughout, so
    the cost is the forward map going stale for a few seconds, not a gap in it.

    Pre-registered: the three durations and the cone are fixed before the grid runs
    and are not tuned afterwards.
    """

    ratio_threshold: float
    cone_deg: float = 60.0
    step_deg: float = 15.0
    hold_s: float = 3.0
    cooldown_s: float = 5.0
    name: str = "glance"
    needs_map: bool = True

    def __post_init__(self) -> None:
        self._holding = 0.0
        self._cooling = self.cooldown_s  # ready to glance from the first step
        self._target_point: np.ndarray | None = None
        self.steps = 0
        self.glancing_steps = 0

    @property
    def glance_fraction(self) -> float:
        return self.glancing_steps / max(self.steps, 1)

    def __call__(self, ctx: StepContext) -> dict:
        self.steps += 1
        spec = getattr(ctx.platform, "spec", None)
        dt = (
            spec.step_length / max(spec.speed_mps, 1e-9) if spec is not None else 0.5
        )
        forward = ctx.platform.track_yaw()
        origin = ctx.T_est[:3, 3]

        if self._target_point is not None:
            self._holding += dt
            if self._holding >= self.hold_s:
                # the glance is over: back to forward, and the cooldown starts now
                self._target_point = None
                self._holding = 0.0
                self._cooling = 0.0
                return {"yaw_target": forward}
            else:
                self.glancing_steps += 1
                rel = self._target_point - origin
                return {"yaw_target": float(np.arctan2(rel[1], rel[0]))}

        if self._cooling < self.cooldown_s:
            self._cooling += dt
            return {"yaw_target": forward}

        loc = ctx.localizability
        points = ctx.local_map_points
        normals = ctx.local_map_normals
        if (
            loc is None
            or loc.ratio >= self.ratio_threshold
            or points is None
            or points.shape[0] < 8
        ):
            return {"yaw_target": forward}

        target = self._structure_behind(points, normals, origin, forward, ctx, loc)
        if target is None:
            return {"yaw_target": forward}
        self._target_point = target
        self._holding = 0.0
        self.glancing_steps += 1
        rel = target - origin
        return {"yaw_target": float(np.arctan2(rel[1], rel[0]))}

    def _structure_behind(self, points, normals, origin, forward, ctx, loc):
        """The best thing to look at outside the forward cone, or None.

        Outside the cone because inside it the sensor is already looking; ``None``
        when nothing out there beats what forward already offers, so the policy
        spends no slew on a glance that would tell it nothing.
        """
        spec = ctx.sim.lidar.spec
        have = _have(loc)
        step = np.deg2rad(self.step_deg)
        cone = np.deg2rad(self.cone_deg)
        offsets = np.arange(-np.pi, np.pi, step)
        outside = offsets[np.abs(offsets) > cone]
        if outside.size == 0:
            return None
        inside = offsets[np.abs(offsets) <= cone]
        best_inside = float(
            _score_yaws(points, normals, origin, forward + inside, spec, have).max()
        )
        scores = _score_yaws(points, normals, origin, forward + outside, spec, have)
        k = int(np.argmax(scores))
        if scores[k] <= best_inside:
            return None

        # the cluster itself, so that the sensor tracks it as the robot moves on
        yaw = float(forward + outside[k])
        rel = points - origin
        dist = np.linalg.norm(rel, axis=1)
        bearing = np.arctan2(rel[:, 1], rel[:, 0])
        delta = np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw))
        visible = (
            (dist > spec.min_range)
            & (dist <= spec.max_range)
            & (np.abs(delta) <= np.deg2rad(spec.fov_azimuth_deg) * 0.5)
        )
        if not visible.any():
            return None
        return points[visible].mean(axis=0)

    def describe(self) -> dict:
        return {
            "policy": self.name,
            "ratio_threshold": self.ratio_threshold,
            "cone_deg": self.cone_deg,
            "hold_s": self.hold_s,
            "cooldown_s": self.cooldown_s,
        }


@dataclasses.dataclass
class OracleGaze(_Gaze):
    """Upper bound on gaze, and only on gaze.

    It reads the true world rather than the part of it the robot has mapped and
    confirmed, so the gap between this and the greedy policy is the cost of having
    to discover structure before you can look at it. It plans from the *estimated*
    pose, not the true one, because a policy that knows where it really is would be
    bounding the estimator as well as the gaze, and the estimator is not what this
    comparison is about.

    The same forward cone as the greedy policy, for the same reason: without it the
    bound would be the bound on a policy nobody would fly.
    """

    ratio_threshold: float
    step_deg: float = 15.0
    cone_deg: float | None = 60.0
    name: str = "oracle"

    def __post_init__(self) -> None:
        self._points = None
        self._normals = None

    def _world_cloud(self, ctx: StepContext):
        if self._points is None:
            self._points, self._normals = _sample_world_surface(ctx.sim)
        return self._points, self._normals

    def __call__(self, ctx: StepContext) -> dict:
        loc = ctx.localizability
        forward = ctx.platform.track_yaw()
        if loc is None or loc.ratio >= self.ratio_threshold:
            return {"yaw_target": forward}

        points, normals = self._world_cloud(ctx)
        # true world, estimated pose: a bound on where to look, not on the estimator
        return {"yaw_target": _best_yaw(
            points, normals, ctx.T_est[:3, 3], ctx.platform.yaw,
            ctx.sim.lidar.spec, self.step_deg, have=_have(loc),
            cone_deg=self.cone_deg, reference=forward,
        )}

    def describe(self) -> dict:
        return {
            "policy": self.name,
            "ratio_threshold": self.ratio_threshold,
            "cone_deg": self.cone_deg,
        }


def _candidate_yaws(current, reference, step_deg, cone_deg=None):
    """Absolute yaws to score.

    Unrestricted, the candidates cover the full circle around the current yaw, so
    the sensor can use anything it has already driven past. Restricted, they cover
    a cone about the direction of travel instead, which is what stops the policy
    from solving its own objective by staring at the map behind it.
    """
    if cone_deg is None:
        return current + np.deg2rad(np.arange(-180.0, 180.0, step_deg))
    n = int(np.floor(float(cone_deg) / step_deg))
    return reference + np.deg2rad(np.arange(-n, n + 1) * step_deg)


def _score_yaws(points, normals, origin, candidates, spec, have=None):
    """E-optimal score of each candidate yaw: the weakest eigenvalue of what the
    robot would know after looking there."""
    base = np.zeros((3, 3))
    if have is not None:
        tr = float(np.trace(have))
        if tr > 0:
            base = np.asarray(have, dtype=float) / tr
    preds = [
        predicted_information(points, normals, origin, yaw, spec) for yaw in candidates
    ]
    scale = max((float(np.trace(H)) for H in preds), default=0.0)
    scale = scale if scale > 0 else 1.0
    return np.array([float(np.linalg.eigvalsh(base + H / scale)[0]) for H in preds])


def _best_yaw(points, normals, origin, current, spec, step_deg, have=None,
              cone_deg=None, reference=None):
    """Candidate yaw maximising the weakest eigenvalue of what would then be known.

    The objective is ``lambda_min(H_have + H_pred)``, not ``lambda_min(H_pred)``.
    The difference matters and is not a refinement: a single flat surface gives
    rank-one information, so its smallest eigenvalue is exactly zero, and in a
    tunnel almost every candidate view is one or two parallel surfaces. Scoring
    the increment alone therefore returns zero for nearly every yaw and the policy
    picks whichever it looked at first. Scoring what the robot would know after
    looking is the standard E-optimal next-best-view criterion and is what the
    increment was standing in for.

    ``H_have`` is the current registration Hessian's translational block. It is
    normalised to unit trace, and every candidate's prediction is divided by the
    single largest trace across candidates, because one is an information matrix
    in real units and the other is a count of predicted surface patches, and their
    absolute scales are not comparable. Dividing each candidate by its own trace
    instead would throw away how much surface it sees and leave only which way the
    normals point, and then a yaw that catches a sliver of the far wall ties with
    one that sees all of it.

    Ties go to the smallest slew from the current yaw, so the sensor settles
    rather than chattering. A tie is common and not a degenerate case: a tunnel is
    symmetric, so looking left and looking right usually score the same.
    """
    reference = current if reference is None else reference
    candidates = _candidate_yaws(current, reference, step_deg, cone_deg)
    scores = _score_yaws(points, normals, origin, candidates, spec, have)
    best = scores.max()
    if not np.isfinite(best) or best <= 0.0:
        near = np.flatnonzero(scores >= best)
    else:
        near = np.flatnonzero(scores >= best * 0.99)
    slew = np.abs((candidates[near] - current + np.pi) % (2.0 * np.pi) - np.pi)
    return float(candidates[near][np.argmin(slew)])


def _have(loc) -> np.ndarray | None:
    """Reconstruct the current translational information from the detector output."""
    if loc is None or not hasattr(loc, "eigenvectors"):
        return None
    V = np.asarray(loc.eigenvectors, dtype=float)
    lam = np.asarray(loc.eigenvalues, dtype=float)
    if V.shape != (3, 3) or lam.shape != (3,):
        return None
    return V @ np.diag(lam) @ V.T


def _sample_world_surface(sim, spacing: float = 0.4):
    """Points and normals on the true tunnel shell, for the oracle only."""
    world = sim.world
    spec = world.spec
    pts, nrm = [], []
    for i in range(0, len(world.s)):
        p = world.centreline[i]
        yaw = float(world.heading[i])
        nx, ny = -np.sin(yaw), np.cos(yaw)
        half = world.width[i] * 0.5
        for sign in (1.0, -1.0):
            base = np.array([p[0] + sign * nx * half, p[1] + sign * ny * half])
            for z in np.arange(0.2, spec.height, spacing):
                pts.append([base[0], base[1], z])
                nrm.append([-sign * nx, -sign * ny, 0.0])
        for off in np.arange(-half, half, spacing):
            pts.append([p[0] + nx * off, p[1] + ny * off, 0.0])
            nrm.append([0.0, 0.0, 1.0])
    return np.array(pts), np.array(nrm)
