"""The detector and the action rule, with no ROS import.

Everything the node decides lives here and in ``prior.py`` so it can be tested
without a graph. The node is then a thin shell: convert, pair each scan with its
prior, call this, publish.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from locrec.lidar import LidarSpec
from locrec.localizability import LocalizabilityConfig
from locrec.odometry import LocalMap, OdometryConfig, Odometry
from locrec.policies import LocalizabilityScheduler
from locrec.runner import default_registration_range
from locrec.se3 import inv_T, make_T
from locrec.sim import MarkerDetection, PlatformSpec
from locrec.worlds import WorldSpec

__all__ = ["DetectorConfig", "DetectorOutput", "LocalizabilityDetector"]


@dataclasses.dataclass(frozen=True)
class DetectorConfig:
    ratio_threshold: float
    """Below this the scan is called degenerate. Required, with no default.

    There is deliberately no value here to copy. The ratio depends on the sensor's
    field of view, its range and the size of the space, so a number calibrated on one
    is meaningless on another, and a default would be a number that looks calibrated
    and is not. It has one source, `locrec/results/thresholds.json`, produced by
    `locrec/experiments/calibrate_thresholds.py`: `ratio_threshold` for a 360 degree scanner
    and `drone_ratio_threshold` for the drone. A constant transcribed into a second
    place and left to go stale is `docs/failures.md` number 20."""
    min_lambda_per_point: float = 0.0
    max_range: float = 10.0
    fov_azimuth_deg: float = 360.0
    azimuth_beams: int = 360
    """Horizontal beams across ``fov_azimuth_deg``. Together they set the azimuth
    spacing, which caps the registration range."""
    platform: str = "ugv"
    """``ugv`` recommends dropping a marker, ``drone`` recommends a yaw."""
    gaze: str = "across"
    """How the drone picks its yaw. ``across`` looks across the weak direction whenever the
    scan is degenerate, the rule this node started with. ``forward`` and ``glance`` are the
    study's ``ForwardGaze`` and ``GlanceGaze`` from ``locrec.gaze``, called with what
    ``run_pass`` gives them; both need the track heading with every scan. Ignored for the
    UGV, whose sensor yaw is its heading."""
    registration_threads: int = 4
    """Threads small_gicp registers with. Not only a speed setting: over a 300 m run the result
    depends on it, deterministically. On Gazebo, mixed world, seed 1, with markers, mean
    along-track error was 0.36 m on one thread and 0.55 m on four, each repeated exactly
    (``docs/failures.md`` number 30). The study's grids register on one thread."""
    start_at_odometry: bool = False
    """Start the estimate at the odometry pose of the first scan instead of the identity.

    ``run_pass`` starts its estimate at the platform's true first pose, and a driver whose
    odometry frame is the world frame at the start (``gz_driver``) makes that the first
    odometry pose; with this set the estimator then runs in the frame ``run_pass`` runs it
    in, voxel grid included. Off, the estimate's origin is the first scan's pose, which is
    all an odometry can know about where it started. The ``forward`` and ``glance`` gazes
    need it on: they steer relative to a track heading given in the odometry frame."""
    elevation_beams: int = 16
    fov_elevation_deg: float = 30.0
    """Vertical beams and opening. They reach the marker measurement model and the gaze
    policies' view of what a yaw would see, so the drone's 112 over 60 degrees matter."""
    map_resolution: float = 0.2
    map_radius: float = 30.0
    """Crop radius of the local map, metres, before ``run_pass``'s floor of twice the
    sensor range. The map has to reach further than the sensor does, or the scan
    sticks out past both ends of it (``docs/failures.md`` number 1)."""
    step_length: float = PlatformSpec().step_length
    """Nominal travel between scans, metres, as ``run_pass`` used it. It only enters
    the registration range, and for both study sensors the sampling limit binds
    before it does."""
    share_landmarks: bool = False
    """Record every strip this vehicle mounts, for the node to tell the team about.

    What gets recorded is the message and only the message: slot, the position this estimator
    believes, the 2x2 of the observation that placed it, and the face normal. See
    ``docs/DECISIONS.md`` under "The team" for why it is those four and nothing else."""
    use_shared_landmarks: bool = False
    """Localize against strips a teammate mounted, in the teammate's frame.

    A vehicle with this on mounts nothing itself. It is the drone half of the team: it inherits
    the frame of whoever placed the strips, and with it that vehicle's chain error, which is the
    floor on how right it can be about the world."""
    marker_width: float = WorldSpec().marker_width
    marker_height: float = WorldSpec().marker_height
    marker_thickness: float = WorldSpec().marker_thickness
    """The strip the robot mounts, as ``WorldSpec`` describes it. The estimator needs all three:
    the width and height size the measurement covariance, and the thickness drives the fit's
    end-face correction, which is off when it is zero. Left to ``Odometry``'s own defaults these
    were silently the core's, with thickness zero, so a ROS run corrected nothing."""
    marker_reliable_range_m: float | None = None
    """Range at which a marker is still reliably detected, metres. Required for the
    UGV, with no default, for the same reason the threshold has none: it is
    calibrated, it lives in `locrec/results/thresholds.json`, and the 5.5 m spacing this
    class used to carry was a copy of an earlier rule that the scheduler had since
    replaced (`docs/failures.md` number 23)."""
    scheduler_margin_m: float | None = None
    """Range headroom the scheduler keeps, metres. `scheduler_margin_m` in the same file."""


@dataclasses.dataclass
class DetectorOutput:
    eigenvalues: np.ndarray
    weak_direction: np.ndarray
    ratio: float
    lambda_min_per_point: float
    is_degenerate: bool
    n_points: int
    action: str
    """One of ``none``, ``drop_marker``, ``yaw_to:<deg>``."""
    landmarks_used: int = 0
    """Known markers that corrected the estimate on this scan."""


class _SchedulerInput:
    """The two things ``LocalizabilityScheduler`` reads from a ``StepContext``.

    Nothing else is offered, on purpose: if the scheduler ever starts reading a
    third thing, this fails loudly instead of the wrapper quietly deciding on less
    than the policy the study measured.
    """

    __slots__ = ("localizability", "estimated_distance")

    def __init__(self, localizability, estimated_distance: float):
        self.localizability = localizability
        self.estimated_distance = float(estimated_distance)


class _GazePlatform:
    __slots__ = ("spec", "_track_yaw")

    def __init__(self, spec: PlatformSpec, track_yaw: float):
        self.spec = spec
        self._track_yaw = float(track_yaw)

    def track_yaw(self) -> float:
        return self._track_yaw


class _GazeLidar:
    __slots__ = ("spec",)

    def __init__(self, spec: LidarSpec):
        self.spec = spec


class _GazeSim:
    __slots__ = ("lidar",)

    def __init__(self, spec: LidarSpec):
        self.lidar = _GazeLidar(spec)


class _GazeInput:
    """What ``GlanceGaze`` and ``ForwardGaze`` read from a ``StepContext``, and nothing else,
    for the reason ``_SchedulerInput`` gives."""

    __slots__ = ("platform", "sim", "T_est", "localizability", "local_map_points", "local_map_normals")

    def __init__(self, platform, sim, T_est, localizability, points, normals):
        self.platform = platform
        self.sim = sim
        self.T_est = T_est
        self.localizability = localizability
        self.local_map_points = points
        self.local_map_normals = normals


class LocalizabilityDetector:
    """Runs the locrec odometry over incoming scans and emits the signal.

    The threshold is only meaningful for a ratio produced the way ``run_pass``
    produced it during calibration, so the registration is configured exactly as it
    configured it: the motion prior, the registration range, and the local map
    radius, which ``run_pass`` floors at twice the sensor range.

    The motion prior. Every scan comes with the odometry pose at its own stamp, and
    consecutive poses are differenced into the increment ``Odometry.step`` expects;
    the first scan defines the origin. There is no default. This detector used to
    assume no motion when no prior was given, on the argument that the Hessian
    describes the geometry rather than the pose. That argument does not survive the
    map being built from the estimate: with the robot moving and the estimate
    standing still, scans are written into the map in the wrong place, the estimate
    stops following the robot, and the blind-stretch ratio rises above the
    calibrated threshold, so nothing is ever called degenerate.

    The registration range, from ``default_registration_range``, including the
    sampling cap measured in ``docs/failures.md`` number 2. Either fault on its own
    was enough to keep every blind-stretch ratio above the threshold.

    The marker rule is not reimplemented here either. It is
    ``locrec.policies.LocalizabilityScheduler`` itself, fed the same two things
    ``run_pass`` feeds it. This class used to carry its own rule, a copy of the
    scheduler as it stood before the chain learned to survive a recovery, and it
    paid for every threshold crossing with a marker.

    Markers close the loop. ``process`` takes the markers mounted since the last
    scan, which are registered from the estimate as it stands before this scan is
    stepped, and the markers seen in this scan, which go to the estimator's absolute
    stage. That is the order ``run_pass`` does it in.

    The estimate starts at the identity, or with ``start_at_odometry`` at the odometry pose
    of the first scan, as ``run_pass`` starts it at the platform's first pose.
    """

    def __init__(self, config: DetectorConfig):
        if config.fov_azimuth_deg <= 0.0 or config.azimuth_beams < 1:
            raise ValueError("fov_azimuth_deg must be positive and azimuth_beams at least 1")
        if config.platform not in ("ugv", "drone"):
            raise ValueError(f"platform must be ugv or drone, got {config.platform!r}")
        if config.gaze not in ("across", "forward", "glance"):
            raise ValueError(f"gaze must be across, forward or glance, got {config.gaze!r}")
        if config.platform == "drone" and config.gaze != "across" and not config.start_at_odometry:
            raise ValueError(f"gaze {config.gaze!r} steers relative to a track heading in the odometry "
                             "frame, so the estimate has to start in that frame: set start_at_odometry")
        self.config = config
        lidar = LidarSpec(
            n_azimuth=int(config.azimuth_beams),
            fov_azimuth_deg=float(config.fov_azimuth_deg),
            max_range=float(config.max_range),
            n_elevation=int(config.elevation_beams),
            fov_elevation_deg=float(config.fov_elevation_deg),
        )
        self.lidar_spec = lidar
        # ``correct_strip_bias`` is deliberately left at its default, which is off. It is
        # sensor-specific: measured in situ, the strip fit's single-column bias is +3.28 cm on
        # MuJoCo's ray casts and +0.43 cm on Gazebo's rendered depth, and this node drives Gazebo
        # (docs/failures.md number 34). ``measure_strip_spread`` stays on, because the spread does
        # transfer between the two sensors.
        base = OdometryConfig(
            map_resolution=config.map_resolution,
            map_radius=max(config.map_radius, 2.0 * config.max_range),
            gicp_ratio_floor=config.ratio_threshold,
            num_threads=int(config.registration_threads),
        )
        odom_cfg = dataclasses.replace(
            base,
            registration_range=default_registration_range(lidar, base, config.step_length),
        )
        # without the sensor model the estimator silently ignores every marker
        self.odometry = Odometry(np.eye(4), odom_cfg, lidar_spec=lidar,
                                 marker_width=float(config.marker_width),
                                 marker_height=float(config.marker_height),
                                 marker_thickness=float(config.marker_thickness))
        self.loc_config = LocalizabilityConfig(
            ratio_threshold=config.ratio_threshold,
            min_lambda_per_point=config.min_lambda_per_point,
        )
        self.shared: list[dict] = []
        self._shared_at: dict[int, np.ndarray] = {}
        """Strips mounted since the node last drained this, each ready to send."""
        self._foreign: set[int] = set()
        self._scheduler: LocalizabilityScheduler | None = None
        if config.platform == "ugv":
            if config.marker_reliable_range_m is None or config.scheduler_margin_m is None:
                raise ValueError(
                    "the ugv recommends markers, so marker_reliable_range_m and "
                    "scheduler_margin_m are required: both are calibrated and live in "
                    "locrec/results/thresholds.json"
                )
            self._scheduler = LocalizabilityScheduler(
                ratio_threshold=config.ratio_threshold,
                reliable_range_m=float(config.marker_reliable_range_m),
                margin_m=float(config.scheduler_margin_m),
            )
        self._gaze = None
        if config.platform == "drone" and config.gaze != "across":
            from locrec.gaze import ForwardGaze, GlanceGaze

            # the policies as locrec/experiments/drone_gaze.py builds them, pre-registered there
            self._gaze = (ForwardGaze() if config.gaze == "forward"
                          else GlanceGaze(ratio_threshold=config.ratio_threshold, cone_deg=60.0))
        self._estimated_distance = 0.0
        self._last_odom_pose: np.ndarray | None = None

    @property
    def needs_track_yaw(self) -> bool:
        """Whether ``process`` needs the track heading with every scan."""
        return self._gaze is not None

    @property
    def pose(self) -> np.ndarray:
        """The estimate: in the odometry's frame with ``start_at_odometry``, else in a frame
        whose origin is the pose of the first scan."""
        return self.odometry.T.copy()

    @property
    def estimated_distance(self) -> float:
        return self._estimated_distance

    def reset(self) -> None:
        self.odometry.map = LocalMap(
            radius=self.odometry.cfg.map_radius, resolution=self.config.map_resolution
        )

    def _share(self, slot: int) -> dict:
        """What this vehicle tells the team about a strip it has just mounted."""
        lm = self.odometry.landmarks.get(slot)
        normal = -(lm.T_drop[:3, :3] @ lm.offset_drop)[:2]
        n = float(np.linalg.norm(normal))
        R = np.eye(2) * self.odometry.landmark_spec.min_sigma_m**2 if lm.R_drop is None else lm.R_drop
        return {"slot": slot, "position": lm.position.copy(), "drop_covariance": np.asarray(R, dtype=float),
                "normal": (normal / n if n > 1e-9 else np.array([1.0, 0.0]))}

    def _reshare_moved_strips(self) -> None:
        """Send a strip again when this vehicle's own estimate of where it is has moved.

        A strip mounted since the last fix drifted with the estimate, so the next fix moves it
        (``Odometry._absolute_fix``). The vehicle knows where it is better afterwards than it did
        when it bolted it on, and a teammate that was told the first answer and never the second
        would be holding a position its owner has already disowned. It happens at most once per
        strip in practice, at the first fix after the drop.
        """
        for slot, sent in list(self._shared_at.items()):
            lm = self.odometry.landmarks.get(slot)
            if lm is None or np.allclose(lm.position, sent, atol=1e-9):
                continue
            self.shared.append(self._share(slot))
            self._shared_at[slot] = lm.position.copy()

    def add_shared_landmark(self, slot: int, position: np.ndarray, drop_covariance: np.ndarray,
                            normal: np.ndarray) -> bool:
        """Take a strip a teammate mounted. Returns whether it was new.

        A teammate republishes a strip when its own estimate of it moves, so the same slot arrives
        more than once. The later answer replaces the position and the covariance, and touches
        nothing else: re-registering would reset the anchor's relative record and hand this
        vehicle a fix it has not earned, which is a different and worse thing than a stale
        position.
        """
        if not self.config.use_shared_landmarks:
            return False
        if int(slot) in self._foreign:
            lm = self.odometry.landmarks.get(int(slot))
            if lm is not None:
                lm.position = np.asarray(position, dtype=float)
                lm.R_drop = np.asarray(drop_covariance, dtype=float).reshape(2, 2)
            return False
        self.odometry.register_foreign_landmark(int(slot), np.asarray(position, dtype=float),
                                                np.asarray(drop_covariance, dtype=float).reshape(2, 2),
                                                np.asarray(normal, dtype=float)[:2])
        self._foreign.add(int(slot))
        return True

    def process(
        self,
        points_sensor: np.ndarray,
        odom_pose: np.ndarray,
        drops: list[tuple[int, np.ndarray]] | None = None,
        detections: list[MarkerDetection] | None = None,
        track_yaw: float | None = None,
    ) -> DetectorOutput | None:
        """One scan in, one signal out. Returns None until registration is possible.

        ``odom_pose`` is the 4x4 pose of the sensor in the odometry frame at this
        scan's stamp. Only differences between consecutive poses are used, so where the
        odometry frame starts does not matter, unless ``start_at_odometry`` places the
        estimate at the first one.

        ``drops`` are ``(slot, offset_sensor)`` for markers mounted since the last
        scan, at the pose of that last scan. ``detections`` are the markers seen in
        this one. Leave both out and the estimator runs on registration alone.

        ``track_yaw`` is the heading of the path at this scan, in the odometry frame,
        which the ``forward`` and ``glance`` gazes steer relative to.
        """
        T_odom = np.asarray(odom_pose, dtype=float)
        if T_odom.shape != (4, 4) or not np.isfinite(T_odom).all():
            raise ValueError(f"odom_pose must be a finite 4x4 pose, got shape {T_odom.shape}")
        if self._gaze is not None and (track_yaw is None or not np.isfinite(track_yaw)):
            raise ValueError(f"gaze {self.config.gaze!r} steers relative to the track heading: pass track_yaw")
        if self._last_odom_pose is None:
            if self.config.start_at_odometry:
                # nothing has been stepped, so placing the estimate here is constructing it here
                self.odometry.T = T_odom.copy()
            delta = np.eye(4)
        else:
            delta = inv_T(self._last_odom_pose) @ T_odom
        self._last_odom_pose = T_odom.copy()

        # a marker is registered from the estimate that held when it was mounted,
        # which is the one still standing here, before this scan moves it
        for slot, offset_sensor in drops or ():
            self.odometry.register_landmark(int(slot), np.asarray(offset_sensor, dtype=float))
            if self.config.share_landmarks:
                self.shared.append(self._share(int(slot)))
                self._shared_at[int(slot)] = self.odometry.landmarks.get(int(slot)).position.copy()

        before = self.odometry.T[:3, 3].copy()
        step = self.odometry.step(
            np.asarray(points_sensor, dtype=float), delta, list(detections or [])
        )
        self._estimated_distance += float(np.linalg.norm(self.odometry.T[:3, 3] - before))
        self._reshare_moved_strips()

        if step.localizability is None:
            return None
        loc = step.localizability
        degenerate = loc.is_degenerate(self.loc_config)
        action = self._action(loc, track_yaw)
        return DetectorOutput(
            eigenvalues=loc.eigenvalues,
            weak_direction=loc.weak_direction,
            ratio=loc.ratio,
            lambda_min_per_point=loc.lambda_min_per_point,
            is_degenerate=degenerate,
            n_points=loc.n_points,
            action=action,
            landmarks_used=int(step.landmarks_used),
        )

    def _action(self, loc, track_yaw: float | None = None) -> str:
        if self._scheduler is not None:
            decision = self._scheduler(_SchedulerInput(loc, self._estimated_distance))
            return "drop_marker" if decision.get("drop_marker") else "none"
        if self._gaze is not None:
            points = normals = None
            if getattr(self._gaze, "needs_map", False):
                from locrec.runner import compute_normals

                points = self.odometry.map.points
                normals = compute_normals(points)
            ctx = _GazeInput(_GazePlatform(PlatformSpec(step_length=self.config.step_length), track_yaw),
                             _GazeSim(self.lidar_spec), self.odometry.T.copy(), loc, points, normals)
            target = float(self._gaze(ctx)["yaw_target"])
            # enough digits that the yaw a vehicle is sent is the yaw run_pass would command
            return f"yaw_to:{np.rad2deg(target):.6f}"
        if not loc.is_degenerate(self.loc_config):
            return "none"
        # look across the weak direction: that is where the structure which
        # would constrain it has to be
        weak = loc.weak_direction
        across = np.array([-weak[1], weak[0]])
        return f"yaw_to:{np.rad2deg(np.arctan2(across[1], across[0])):.1f}"


def identity_pose() -> np.ndarray:
    return make_T(np.eye(3), np.zeros(3))
