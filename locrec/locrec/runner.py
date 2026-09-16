"""One open-loop pass down a tunnel, with a hook for a recovery policy.

This is the harness every later milestone plugs into. A policy sees the
localizability signal and the platform, and may act on the platform (drop a
marker, command sensor yaw). It never sees ground truth.
"""
from __future__ import annotations

import dataclasses
from typing import Literal, Protocol

import numpy as np

from .landmarks import LandmarkSpec
from .lidar import LIMITED_FOV, SPINNING_360, LidarSpec
from .localizability import Localizability
from .odometry import MotionPrior, MotionPriorSpec, Odometry, OdometryConfig, OdomStep
from .se3 import inv_T, pose_error
from .sim import Drone, MarkerDetection, PlatformSpec, TunnelSim, UGV
from .worlds import WorldSpec

__all__ = [
    "StepContext",
    "Policy",
    "RunConfig",
    "RunResult",
    "default_registration_range",
    "run_pass",
]

PlatformName = Literal["ugv", "drone"]


@dataclasses.dataclass
class StepContext:
    """Everything a policy is allowed to look at. No ground truth here."""

    step: int
    sim: TunnelSim
    platform: UGV | Drone
    T_est: np.ndarray
    """The estimated sensor pose. The policy may use its own estimate, never truth."""
    localizability: Localizability | None
    detections: list[MarkerDetection]
    """Markers visible in the most recent scan."""
    registration: OdomStep
    """Last odometry step, for policies that watch registration health instead."""
    markers_placed: int
    markers_remaining: int
    estimated_distance: float
    """Distance travelled according to the estimate, not the truth."""
    local_map_points: np.ndarray | None = None
    """Confirmed local map, world frame. What a gaze policy is allowed to plan on."""
    local_map_normals: np.ndarray | None = None
    path_length: float = 0.0
    """True arclength. Present for logging only; a policy that reads it is cheating."""

    @property
    def marker_in_view(self) -> bool:
        return len(self.detections) > 0


class Policy(Protocol):
    def __call__(self, ctx: StepContext) -> dict:
        """Return an action dict. Recognised keys: ``drop_marker``, ``yaw_effort``."""
        ...


@dataclasses.dataclass(frozen=True)
class RunConfig:
    seed: int
    platform: PlatformName = "ugv"
    world: WorldSpec = WorldSpec()
    lidar: LidarSpec | None = None
    """Defaults to SPINNING_360 for the UGV and LIMITED_FOV for the drone."""
    platform_spec: PlatformSpec = PlatformSpec()
    odometry: OdometryConfig = OdometryConfig()
    prior: MotionPriorSpec = MotionPriorSpec()
    landmark: LandmarkSpec = LandmarkSpec()
    max_steps: int = 10_000

    def lidar_spec(self) -> LidarSpec:
        if self.lidar is not None:
            return self.lidar
        return SPINNING_360 if self.platform == "ugv" else LIMITED_FOV


@dataclasses.dataclass
class RunResult:
    config: RunConfig
    rows: list[dict]
    final_translation_error: float
    final_along_track_error: float
    final_lateral_error: float
    final_rotation_error: float
    path_length: float
    yaw_effort: float
    """Total commanded sensor yaw beyond what tracking the path required, radians."""
    markers_placed: int
    odometry: object | None = None
    """The estimator that produced this run, kept for diagnostics.

    Its ``fix_log`` is the only place a marker fix's magnitude and weighting can be
    seen from outside, and a fix that quietly does nothing looks like a fix that
    worked from every other signal the run records."""

    def array(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.rows], dtype=float)

    def summary(self) -> dict:
        return {
            "seed": self.config.seed,
            "platform": self.config.platform,
            "steps": len(self.rows),
            "path_length_m": self.path_length,
            "final_trans_err_m": self.final_translation_error,
            "final_along_err_m": self.final_along_track_error,
            "final_lateral_err_m": self.final_lateral_error,
            "final_rot_err_rad": self.final_rotation_error,
            "yaw_effort_rad": self.yaw_effort,
            "markers_placed": self.markers_placed,
        }


def _track_delta(platform) -> float:
    """How far the sensor yaw would have moved just to keep following the path."""
    return float(
        np.arctan2(
            np.sin(platform.track_yaw() - platform.yaw),
            np.cos(platform.track_yaw() - platform.yaw),
        )
    )


def compute_normals(points):
    from .gaze import map_normals

    return map_normals(points)


def default_registration_range(
    lidar: LidarSpec, odometry: OdometryConfig, step_length: float
) -> float:
    """The registration range every calibrated threshold was produced with.

    The smaller of two limits. The geometric one keeps the leading shell of the
    scan, which the map does not reach yet, out of the registration (see
    ``OdometryConfig.registration_range``). The sampling one stops where the scan's
    azimuth spacing exceeds half a map voxel (see ``LidarSpec.nyquist_range`` and
    ``docs/failures.md`` number 2).

    ``run_pass`` and the ROS 2 detector both call this. The detector used to derive
    its own range as a fixed fraction of the sensor range, which dropped the
    sampling limit and registered the UGV out to 8.5 m instead of 5.73 m, so the
    ratio it compared against the threshold was not the ratio the threshold was
    calibrated on.
    """
    return min(
        lidar.max_range - odometry.max_correspondence_distance - step_length,
        lidar.nyquist_range(odometry.map_resolution),
    )


def run_pass(
    cfg: RunConfig,
    policy: Policy | None = None,
    detection_hook=None,
    sim: TunnelSim | None = None,
    step_hook=None,
) -> RunResult:
    """Drive one platform from one end of a generated tunnel to the other.

    ``detection_hook(sim, T_true, detections)`` may replace the marker detections
    before the estimator sees them. It exists for the ablations in
    ``experiments/marker_leak.py``, which need to substitute perfect detections to
    separate a detection fault from an estimator fault, and it is not used by any
    experiment that produces a headline number.

    ``sim`` runs the pass in a simulator other than the MuJoCo one ``cfg`` builds, such
    as ``locrec.gazebo.GazeboTunnelSim``. It has to be built from the same seed, world
    and LiDAR as ``cfg``, and fresh: a pass mounts markers in it.

    ``step_hook(k, sim, odom)`` runs at the top of step ``k``, before the platform moves and so
    before the scan that step is registered against. It exists for the team passes in
    ``experiments/team_pass.py``, where one vehicle's markers have to appear on the wall of
    another vehicle's world, and be handed to its estimator, at the step the first vehicle
    mounted them. Like ``detection_hook`` it is a way to run a different experiment through the
    same code path, not a way to change what a pass does.
    """
    if sim is None:
        sim = TunnelSim(seed=cfg.seed, world_spec=cfg.world, lidar_spec=cfg.lidar_spec())
    elif (sim.world.seed, sim.world.spec, sim.lidar.spec) != (cfg.seed, cfg.world, cfg.lidar_spec()):
        raise ValueError("sim was not built from this config's seed, world and LiDAR")
    elif sim.n_markers_placed:
        raise ValueError("sim already has markers mounted; a pass needs a fresh one")
    platform: UGV | Drone = (
        UGV(sim.world, cfg.platform_spec)
        if cfg.platform == "ugv"
        else Drone(sim.world, cfg.platform_spec)
    )
    dt = cfg.platform_spec.step_length / max(cfg.platform_spec.speed_mps, 1e-9)
    prior = MotionPrior(cfg.prior, seed=cfg.seed, dt=dt)

    T_true = platform.pose()
    lspec = cfg.lidar_spec()
    odom_cfg = dataclasses.replace(
        cfg.odometry,
        map_radius=max(cfg.odometry.map_radius, 2.0 * lspec.max_range),
        registration_range=(
            cfg.odometry.registration_range
            if cfg.odometry.registration_range is not None
            else default_registration_range(lspec, cfg.odometry, cfg.platform_spec.step_length)
        ),
    )
    odom = Odometry(
        T_true.copy(),
        odom_cfg,
        prior_spec=cfg.prior,
        dt=dt,
        lidar_spec=lspec,
        landmark_spec=cfg.landmark,
        marker_height=cfg.world.marker_height,
        marker_width=cfg.world.marker_width,
        marker_thickness=cfg.world.marker_thickness,
    )
    rows: list[dict] = []
    yaw_effort = 0.0
    estimated_distance = 0.0
    needs_map = getattr(policy, "needs_map", False)

    scan = sim.scan(T_true)
    detections = sim.marker_detections(scan)
    if detection_hook is not None:
        detections = detection_hook(sim, T_true, detections)
    step_out = odom.step(scan.points, np.eye(4), detections)
    rows.append(_row(0, platform, T_true, step_out, sim, 0.0, 0))

    for k in range(1, cfg.max_steps):
        if platform.finished:
            break
        if step_hook is not None:
            step_hook(k, sim, odom)
        T_prev_true = T_true

        # the policy acts before the move, on the information the last scan gave it
        extra_yaw = 0.0
        dropped = 0
        if policy is not None:
            map_points = map_normals = None
            if needs_map:
                map_points = odom.map.points
                map_normals = compute_normals(map_points)
            ctx = StepContext(
                step=k,
                sim=sim,
                platform=platform,
                T_est=odom.T.copy(),
                localizability=step_out.localizability,
                detections=detections,
                registration=step_out,
                markers_placed=sim.n_markers_placed,
                markers_remaining=sim.marker_capacity - sim.n_markers_placed,
                estimated_distance=estimated_distance,
                local_map_points=map_points,
                local_map_normals=map_normals,
                path_length=platform.path_length,
            )
            action = policy(ctx) or {}
            extra_yaw = float(action.get("yaw_effort", 0.0))
            if "yaw_target" in action and isinstance(platform, Drone):
                slew = platform.command_yaw(float(action["yaw_target"]))
                # the cost is the slew beyond what simply following the path needed
                extra_yaw += max(slew - abs(_track_delta(platform)), 0.0)
            if action.get("drop_marker") and sim.n_markers_placed < sim.marker_capacity:
                slot, offset_sensor = sim.drop_marker_on_wall(T_true)
                odom.register_landmark(slot, offset_sensor)
                dropped = 1
        elif isinstance(platform, Drone):
            platform.track_heading()

        platform.advance()
        T_true = platform.pose()
        yaw_effort += extra_yaw

        delta_true = inv_T(T_prev_true) @ T_true
        delta_prior = prior.predict(delta_true)

        T_before = odom.T.copy()
        scan = sim.scan(T_true)
        detections = sim.marker_detections(scan)
        if detection_hook is not None:
            detections = detection_hook(sim, T_true, detections)
        step_out = odom.step(scan.points, delta_prior, detections)
        estimated_distance += float(np.linalg.norm(step_out.T[:3, 3] - T_before[:3, 3]))
        rows.append(_row(k, platform, T_true, step_out, sim, extra_yaw, dropped))

    te, re = pose_error(odom.T, T_true)
    return RunResult(
        config=cfg,
        rows=rows,
        final_translation_error=te,
        final_along_track_error=abs(rows[-1]["along_err_m"]),
        final_lateral_error=rows[-1]["lateral_err_m"],
        final_rotation_error=re,
        path_length=platform.path_length,
        yaw_effort=yaw_effort,
        markers_placed=sim.n_markers_placed,
        odometry=odom,
    )


def _row(k, platform, T_true, step_out: OdomStep, sim: TunnelSim, extra_yaw, dropped) -> dict:
    te, re = pose_error(step_out.T, T_true)
    # split the error along and across the tunnel: markers only address the first,
    # so reporting one scalar would hide whether a policy did its job
    axis = T_true[:3, :3] @ np.array([1.0, 0.0, 0.0])
    err = step_out.T[:3, 3] - T_true[:3, 3]
    along = float(err @ axis)
    row = {
        "step": k,
        "s": platform.s,
        "trans_err_m": te,
        "along_err_m": along,
        "lateral_err_m": float(np.linalg.norm(err - along * axis)),
        "rot_err_rad": re,
        "x_true": T_true[0, 3],
        "y_true": T_true[1, 3],
        "x_est": step_out.T[0, 3],
        "y_est": step_out.T[1, 3],
        "n_scan_points": step_out.n_scan_points,
        "num_inliers": step_out.num_inliers,
        "registered": int(step_out.registered),
        "converged": int(step_out.converged),
        "remapped_axes": step_out.remapped_axes,
        "landmarks_used": step_out.landmarks_used,
        "yaw_fixed": step_out.yaw_fixed,
        "markers_placed": sim.n_markers_placed,
        "marker_dropped": dropped,
        "extra_yaw_rad": extra_yaw,
        "yaw": getattr(platform, "yaw", float("nan")),
    }
    if step_out.localizability is not None:
        row.update(step_out.localizability.as_row())
    else:
        row.update(
            {
                "loc_lambda_min": np.nan,
                "loc_lambda_mid": np.nan,
                "loc_lambda_max": np.nan,
                "loc_ratio": np.nan,
                "loc_lambda_min_per_point": np.nan,
                "loc_weak_x": np.nan,
                "loc_weak_y": np.nan,
                "loc_weak_z": np.nan,
                "loc_rot_ratio": np.nan,
                "loc_n_points": 0,
            }
        )
    return row


def write_csv(rows: list[dict], path: str) -> None:
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
