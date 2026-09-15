"""Two-stage estimation: a relative frontend and an absolute correction.

Stage A, relative. The motion prior predicts the pose, the new scan is placed
there, ``small_gicp`` registers it against the local map, and the correction is
blended with the prior by their information matrices. The 6x6 Hessian of that
registration is the localizability signal. This stage is self-consistent and it
drifts: the map is built from the estimate, so if the estimate slides along a
tunnel the map slides with it and the registration stays perfectly satisfied.

Stage B, absolute. When a marker is observed, the correction comes from the
predicted minus the measured landmark position, with the gain taken from a
covariance integrated since the last absolute fix and reset on each one. The
correction is applied to the pose **and rigidly to the local map and to every
landmark registered since the last fix**. That last part is not a detail. The
registration constraint is relative, so shifting pose and map together leaves it
satisfied; shifting the pose alone means the next registration drags it straight
back, which is exactly why markers changed nothing before this existed.

Nothing here knows about tunnels or recovery motions. The policies in
``locrec.policies`` layer scheduling on top of the localizability signal.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import small_gicp

from .landmarks import (
    LandmarkBook,
    LandmarkSpec,
    measurement_information,
    predicted_beam_count,
    strip_fit_bias,
    strip_fit_sigma,
)
from .localizability import Localizability, analyse_hessian
from .se3 import make_T, rotz, so3_log, transform_points

__all__ = [
    "MotionPriorSpec",
    "MotionPrior",
    "LocalMap",
    "OdometryConfig",
    "Odometry",
    "OdomStep",
    "prior_information",
    "voxel_downsample",
    "voxel_keys",
]


# ---------------------------------------------------------------------------
# motion prior
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MotionPriorSpec:
    """A dead-reckoning prior with the error structure of IMU + wheel odometry.

    Units are physical, not per-step, so the prior does not silently change when
    the scan spacing changes. Defaults are chosen from hardware, not from
    results, and are justified as follows:

    ``gyro_bias_rad_s = 8.7e-4`` (0.05 deg/s) is the residual attitude rate error
    left after a LiDAR-inertial system has estimated the in-run bias of a
    tactical-grade-adjacent MEMS gyro. Raw uncorrected bias would be an order of
    magnitude worse; assuming it here would make the prior a straw man.

    ``gyro_noise_rad_s = 3e-4`` corresponds to an angle random walk of about
    0.3 deg/sqrt(hr) sampled at the scan rate.

    ``odom_scale_sigma = 0.02`` is a 2 percent distance scale error, typical of
    wheel odometry on loose ground and of a visual-inertial velocity estimate.
    It is constant within a run, which is what makes it the error the tunnel
    axis cannot observe away. This is the parameter the whole study is about, so
    milestone 4 sweeps it rather than trusting one value.
    """

    odom_scale_sigma: float = 0.02
    """1 sigma relative error on translated distance, constant per run."""
    odom_noise_m_per_sqrt_m: float = 0.01
    """1 sigma random-walk translation noise, metres per sqrt(metre) travelled."""
    gyro_bias_rad_s: float = 8.7e-4
    """1 sigma constant residual gyro bias, rad/s, drawn once per run per axis."""
    gyro_noise_rad_s: float = 3e-4
    """1 sigma white gyro noise expressed as a rate, rad/s, scaled by sqrt(dt)."""
    odom_min_sigma_m: float = 2e-3
    """Floor on the per-step translation sigma, metres.

    A random walk expressed per sqrt(metre) goes to zero information as the step
    goes to zero, which would let a stationary or nearly stationary platform claim
    a perfect prior and veto the registration outright. A stopped platform's
    velocity estimate is not exact either, so the floor is physical as well as
    numerical. It only binds below about 4 cm of travel per scan."""


class MotionPrior:
    """Corrupts the true inter-scan motion into a prior the estimator may use."""

    def __init__(self, spec: MotionPriorSpec = MotionPriorSpec(), seed: int = 0, dt: float = 0.5):
        self.spec = spec
        self.dt = float(dt)
        self._rng = np.random.default_rng(seed)
        self.scale = 1.0 + self._rng.normal(0.0, spec.odom_scale_sigma)
        self.gyro_bias = self._rng.normal(0.0, spec.gyro_bias_rad_s, 3) * self.dt

    def predict(self, delta_T_true: np.ndarray) -> np.ndarray:
        """Return a noisy version of the true body-frame motion increment."""
        s = self.spec
        d = float(np.linalg.norm(delta_T_true[:3, 3]))
        dt = delta_T_true[:3, 3] * self.scale + self._rng.normal(
            0.0, s.odom_noise_m_per_sqrt_m * np.sqrt(max(d, 1e-9)), 3
        )
        dw = so3_log(delta_T_true[:3, :3]) + self.gyro_bias + self._rng.normal(
            0.0, s.gyro_noise_rad_s * np.sqrt(self.dt), 3
        )
        # small-angle composition is enough: increments are a fraction of a radian
        theta = np.linalg.norm(dw)
        if theta < 1e-12:
            R = np.eye(3)
        else:
            k = dw / theta
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return make_T(R, dt)


def prior_information(
    spec: MotionPriorSpec,
    dt: float,
    delta_T_prior: np.ndarray,
    R_world_body: np.ndarray | None = None,
) -> np.ndarray:
    """6x6 information of the one-step prior error, ordered [rot, trans], in world frame.

    Computed from the increment the prior actually produced, not from a nominal
    step length, because a recovery policy is free to change speed and a prior
    sized for the nominal step would then be wrong in both directions.

    The translation block is anisotropic. The scale error acts only along the
    direction of travel, the random walk acts on all three axes, so in the body
    frame the covariance is

        sigma_rw^2 I + (scale_sigma * |d|)^2 u u^T

    with u the unit step direction. That is rotated into the frame the
    registration Hessian lives in before the two are combined. Treating it as
    isotropic instead would either over-trust the along-track axis (the one the
    tunnel cannot observe) or under-trust the lateral axes (the ones it can),
    and the whole study turns on telling those apart.

    The scale error is a constant bias rather than white noise, so a per-step
    sigma still understates how it accumulates over a long blind stretch. That
    error is in the conservative direction: it flatters the prior, so any
    measured benefit of a recovery policy is a lower bound.
    """
    sigma_rot = float(
        np.hypot(spec.gyro_bias_rad_s * dt, spec.gyro_noise_rad_s * np.sqrt(max(dt, 1e-12)))
    )

    d = np.asarray(delta_T_prior, dtype=float)[:3, 3]
    dist = float(np.linalg.norm(d))
    var_rw = spec.odom_noise_m_per_sqrt_m**2 * dist + spec.odom_min_sigma_m**2
    cov = np.eye(3) * var_rw
    if dist > 1e-9:
        u = d / dist
        cov = cov + (spec.odom_scale_sigma * dist) ** 2 * np.outer(u, u)
    if R_world_body is not None:
        R = np.asarray(R_world_body, dtype=float)
        cov = R @ cov @ R.T

    info = np.zeros((6, 6))
    info[:3, :3] = np.eye(3) / max(sigma_rot, 1e-12) ** 2
    info[3:, 3:] = np.linalg.inv(cov)
    return info


# ---------------------------------------------------------------------------
# local map
# ---------------------------------------------------------------------------


_VOXEL_BITS = 21
_VOXEL_OFFSET = 1 << (_VOXEL_BITS - 1)
_VOXEL_MASK = (1 << _VOXEL_BITS) - 1


def voxel_keys(points: np.ndarray, resolution: float) -> np.ndarray:
    """Pack 3-D voxel indices into one int64 per point.

    ``np.unique(..., axis=0)`` on an (N, 3) key array costs a lexicographic
    argsort over rows and was the single most expensive call in the pipeline,
    above both the ray casting and the registration. Packing three 21-bit
    indices into one integer turns it into a 1-D unique. The range is plus or
    minus 2^20 voxels, which at a 0.2 m pitch is plus or minus 200 km.
    """
    idx = np.floor(np.asarray(points, dtype=float) / resolution).astype(np.int64)
    if idx.size and (np.abs(idx).max() >= _VOXEL_OFFSET):
        raise ValueError("point outside the packed voxel range; raise the resolution")
    idx = idx + _VOXEL_OFFSET
    return (idx[:, 0] << (2 * _VOXEL_BITS)) | (idx[:, 1] << _VOXEL_BITS) | idx[:, 2]


def voxel_downsample(points: np.ndarray, resolution: float) -> np.ndarray:
    """Deterministic voxel-grid downsample: one centroid per occupied voxel."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return pts.reshape(0, 3)
    _, inverse, counts = np.unique(
        voxel_keys(pts, resolution), return_inverse=True, return_counts=True
    )
    sums = np.zeros((counts.shape[0], 3))
    np.add.at(sums, inverse, pts)
    return sums / counts[:, None]


class LocalMap:
    """Voxel-hashed local map in the estimator's world frame, cropped to a radius.

    The crop radius must exceed the sensor range, otherwise the live scan sticks
    out past the ends of the map and the only axial correspondences left are at
    the map's boundary. Registration then pulls every new scan back toward the
    map centroid, and the estimate stops advancing while the robot keeps moving.
    That failure is silent, it looks like excellent convergence, and it is the
    reason this class crops by distance instead of keeping a fixed scan count.

    A voxel is offered for registration only once ``min_observations`` separate
    scans have landed in it. At the far end of a long-range narrow-FOV scan a
    single sweep leaves beams further apart than the voxel pitch, so a map built
    from one viewpoint is full of one-hit voxels whose nearest neighbour is in the
    wrong place; registering against them drags a forward-looking platform
    backwards for the first few metres of every run. Requiring a second
    observation removes that transient without any warm-up counter: early on the
    confirmed map is empty, registration is skipped, and the prior carries the
    pose until parallax has confirmed enough of the map.
    """

    def __init__(
        self,
        radius: float = 30.0,
        resolution: float = 0.2,
        max_points: int = 80_000,
        min_observations: int = 2,
    ):
        self.radius = float(radius)
        self.resolution = float(resolution)
        self.max_points = int(max_points)
        self.min_observations = int(min_observations)
        self._keys = np.zeros(0, dtype=np.int64)
        self._sums = np.zeros((0, 3))
        self._counts = np.zeros(0, dtype=np.int64)
        # the voxel grid lives in its own frame, and an absolute fix moves that
        # frame rather than the points, so the grid is never rebinned
        self._R = np.eye(3)
        self._t = np.zeros(3)

    def _to_map(self, points_world: np.ndarray) -> np.ndarray:
        return (np.asarray(points_world, dtype=float) - self._t) @ self._R

    def add(self, points_world: np.ndarray, centre: np.ndarray | None = None) -> None:
        new = self._to_map(points_world)
        if new.shape[0] == 0:
            return
        if centre is not None:
            centre = self._to_map(np.asarray(centre, dtype=float).reshape(1, 3))[0]
        # one observation per voxel per scan, so counts mean "how many scans saw it"
        uk, inv = np.unique(voxel_keys(new, self.resolution), return_inverse=True)
        usums = np.zeros((uk.shape[0], 3))
        np.add.at(usums, inv, new)
        ucounts = np.bincount(inv, minlength=uk.shape[0]).astype(np.int64)
        usums = usums / ucounts[:, None]  # voxel centroid for this scan

        keys = np.concatenate([self._keys, uk])
        sums = np.vstack([self._sums, usums])
        counts = np.concatenate([self._counts, np.ones(uk.shape[0], dtype=np.int64)])

        mk, minv = np.unique(keys, return_inverse=True)
        msums = np.zeros((mk.shape[0], 3))
        np.add.at(msums, minv, sums)
        mcounts = np.zeros(mk.shape[0], dtype=np.int64)
        np.add.at(mcounts, minv, counts)

        if centre is not None:
            centroids = msums / mcounts[:, None]
            keep = np.linalg.norm(centroids - centre, axis=1) <= self.radius
            mk, msums, mcounts = mk[keep], msums[keep], mcounts[keep]

        if mk.shape[0] > self.max_points:
            # drop the furthest voxels, not every Nth: a stride thins the near field
            # the registration depends on just as hard as the far field it should shed
            ref = np.asarray(centre, dtype=float) if centre is not None else np.zeros(3)
            order = np.argsort(np.linalg.norm(msums / mcounts[:, None] - ref, axis=1))
            keep = order[: self.max_points]
            mk, msums, mcounts = mk[keep], msums[keep], mcounts[keep]

        self._keys, self._sums, self._counts = mk, msums, mcounts

    def apply_transform(self, R: np.ndarray, t: np.ndarray) -> None:
        """Move the whole map rigidly, by moving its frame.

        An absolute fix moves the pose. The registration constraint is relative, so
        moving the map by the same rigid transform leaves it exactly as satisfied
        as it was. Skip this and the next registration simply pulls the pose back
        to where the stale map says it should be, which is why markers did nothing
        before this existed.

        The shift is applied to the grid's frame rather than to the points, so no
        rebinning happens. Rebinning looked harmless and was not: with a fix on most
        steps the map was re-voxelised hundreds of times per run, and every pass
        snapped its points to fresh voxel centroids. The smearing cost more than the
        markers were worth wherever the registration still had something to work
        with.
        """
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float)
        self._R = R @ self._R
        self._t = R @ self._t + t

    @property
    def points(self) -> np.ndarray:
        """Confirmed voxel centroids: those seen by at least ``min_observations`` scans."""
        if self._counts.shape[0] == 0:
            return np.zeros((0, 3))
        keep = self._counts >= self.min_observations
        if not keep.any():
            return np.zeros((0, 3))
        pts = self._sums[keep] / self._counts[keep][:, None]
        return pts @ self._R.T + self._t

    @property
    def n_voxels(self) -> int:
        """Every voxel, confirmed or not."""
        return int(self._counts.shape[0])

    def __len__(self) -> int:
        return int(self.points.shape[0])


# ---------------------------------------------------------------------------
# odometry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OdometryConfig:
    downsampling_resolution: float = 0.2
    max_correspondence_distance: float = 1.0
    max_iterations: int = 30
    num_threads: int = 4
    registration_type: str = "GICP"
    map_radius: float = 30.0
    """Crop radius of the local map, metres. Must exceed the LiDAR max range;
    ``run_pass`` sets it from the sensor spec."""
    map_resolution: float = 0.2
    map_min_observations: int = 2
    """Scans that must have hit a voxel before it is used for registration."""
    registration_range: float | None = None
    """Only scan points within this range of the sensor are used for registration;
    the full scan still goes into the map.

    This is not an optimisation. A scan reaches the sensor's full range, but the
    map's forward boundary sits one step behind that, so the leading shell of every
    scan has no map support. Those points find their nearest correspondence behind
    themselves and drag the solution backwards by a few centimetres every step. In
    a straight tunnel, where nothing opposes that pull, it compounds into a
    systematic lag of ten percent of distance travelled while the registration
    reports healthy convergence throughout. Measured on a uniform straight tunnel
    with a 10 m sensor and a perfect map and initialisation, the mean axial
    correction is -0.117 m per 0.5 m step at full range and -0.004 m once the
    registration range is cut to the supported shell.

    ``run_pass`` sets this to ``max_range - max_correspondence_distance - step_length``,
    which is the geometric margin the effect requires, not a tuned value.
    """
    min_scan_points: int = 60
    """Below this many returns the registration is skipped and the prior is trusted."""
    fuse_prior: bool = True
    """Fuse the registration correction with the zero-mean one-step prior using both
    information matrices. This is the baseline behaviour, not a recovery policy:
    accepting the raw GICP correction along a direction the Hessian says is
    unobserved is a known way to make odometry worse than dead reckoning, and a
    baseline that does that would be a straw man."""
    calibrate_information: bool = True
    """Rescale the GICP Hessian by (inliers - 6) / residual before fusing, the usual
    Gauss-Newton covariance estimate with an unknown noise scale. Without it the
    two information matrices are not on a common footing."""
    gicp_ratio_floor: float | None = 1.9e-3
    """Absolute guard. Zero the registration's translational information along
    directions whose eigenvalue ratio falls below this.

    The registration's along-track Hessian claims a 4 to 5 mm standard deviation in
    a tunnel with no along-track structure at all. ``experiments/gt_map_diagnostic.py``
    shows it is not real: rebuild the local map from ground-truth poses, so that it
    is a fixed and correct reference, and along-track drift does not vanish. It
    halves, from about 1.0 m to about 0.4 m over 120 m, and what is left is a
    systematic backward pull rather than a lock. So the number is an artefact of
    treating a thousand correlated correspondences as independent evidence, and it
    must not be allowed to outvote an absolute measurement.

    The value is a quantile of the ratio measured on calibration seeds in a
    known-blind tunnel, written to ``results/thresholds.json`` by
    ``experiments/calibrate_thresholds.py``. Anything as degenerate as a tunnel
    that is degenerate by construction is treated as unobserved."""
    relative_anchors: bool = True
    """Weight a marker fix by the odometry accumulated since that marker was
    dropped, rather than by the marker's absolute covariance.

    A dropped marker does not know where it is in the world. What it knows is where
    the robot was when it went on the wall, so the constraint it supplies is
    relative: the residual's covariance is the drop-time observation, plus the
    current observation, plus whatever the odometry accumulated in between. Nothing
    in that grows with distance from the start of the run.

    Weighting by the absolute covariance instead, which is what this did before,
    makes a marker dropped 250 m into a tunnel look worthless precisely where it is
    most needed, because the absolute covariance only ever grows. Set this false to
    recover the old behaviour, which also restores the posterior floor that the old
    behaviour needed to stay stable."""
    yaw_fix: bool = True
    """Let two or more anchors in view correct heading as well as position.

    One anchor gives position only. Two give a bearing, and heading is what limits
    the traverse once the position fix works: an anchor is registered through a
    1.5 m lever arm to the wall and inherits the heading error at drop time, about
    3 cm per leg and systematic."""
    estimate_scale: bool = True
    """Estimate the odometry's distance scale error as a state.

    The prior's scale error is one number per run, not a new random draw per step,
    and modelling it as white noise is what stops a marker chain from behaving like
    a survey traverse: the filter sits at a steady-state lag behind every anchor and
    the next anchor is dropped carrying that lag, so error grows linearly in legs
    rather than as their square root. A marker fix measures exactly this quantity.
    The along-track residual over a leg of length L is an observation of
    ``(s - 1) * L``, so one scalar Kalman step per fix estimates it, and from then
    on the prior increment is divided by the estimate."""
    scale_random_walk_per_m: float = 1e-4
    """Standard deviation the scale estimate is allowed to wander by, per metre.

    Read as a standard deviation rather than a variance: at a variance of 1e-4 per
    metre the random walk would swamp the 2 percent initial uncertainty within four
    metres, which cannot be the intent of a term meant to let a slowly varying bias
    move."""
    correct_strip_bias: bool = False
    """Subtract the strip fit's expected bias from every marker observation.

    **Off by default, on the evidence, although it is exact at the measurement level.** It is
    sensor-specific and only one sensor was ever measured. The Monte Carlo behind it runs the real
    fit on synthetic returns cast as rays, which is what MuJoCo does, and on MuJoCo it removes the
    per-observation bias to within two standard errors at every range. Gazebo's ranges come out of
    a rendered depth texture instead, and measured in situ there, on the detections of a real run,
    the single-column bias is **+0.43 cm with 4.74 cm of scatter**, against MuJoCo's **+3.28 cm with
    2.67 cm**: the same total error, split differently. Subtracting MuJoCo's offset from Gazebo's
    returns therefore adds error, and the declared cross-check says so, scheduler final along-track
    median 1.48 m with it on against 0.85 m with it off, better on 4 of 8 seeds against 5
    (`results/gazebo_crosscheck_ugv_strip_bias_on.csv` against
    `results/gazebo_crosscheck_ugv.csv`). On MuJoCo its own declared grid had already called it a
    null. Turn it on only for a sensor whose strip bias has been measured and matches the model.

    A strip seen along a wall shows its end face, the fit reads it 1 to 2 cm nearer than it is,
    and a chain of strips carries that forward: 1.5 m over 300 m, always short
    (``landmarks.strip_fit_bias``, ``docs/failures.md`` number 31). Needs the strip's thickness,
    which ``Odometry`` is given; with a thickness of zero this does nothing."""
    measure_strip_spread: bool = True
    """Take the spread of a marker observation from the same Monte Carlo as its bias.

    What an observation of a strip is worth used to be modelled by hand, as beam quantisation
    plus the strip's width over root twelve, shared out over the returns. Two things were wrong
    with that. It divided the along-tunnel axis by the number of returns when the returns were
    one column of rings sharing one azimuth, which over-trusted a far detection by a factor of
    six in variance; and where the returns do resolve the strip it charged 2.2 cm against a
    measured 0.63 cm. With this on, ``landmarks.strip_fit_sigma`` runs the real fit at this
    geometry and reports what it actually does, which already contains the range noise, the beam
    comb and the strip's shape (``docs/failures.md`` number 31). Needs the strip's thickness."""
    credit_registration: bool = False
    """Subtract what the registration observed from the odometry a marker fix is
    weighted against.

    Off by the milestone 8 specification, which defines the relative covariance as
    the sum of per-step prior covariances since the drop. That makes a marker keep
    full authority even in a stretch where the geometry is good and the
    registration genuinely does know how far the robot has come, so a marker whose
    own accuracy is tens of centimetres can outvote a frontend that is at
    centimetres. On when this is true, each step's increment is the prior fused
    with the guarded registration information rather than the prior alone, which
    keeps the accumulator additive and therefore keeps every anchor's record a
    difference of two marks on it."""
    resurvey_anchors: bool = False
    """Re-register an anchor whenever the pose is better known than the anchor is.

    Off by default since milestone 9. It is defensible in principle, and in practice
    it re-registers an anchor from an estimate that is itself drifting, so the
    anchor tracks the drift instead of opposing it. Measured on the blind world with
    the scheduler, seeds 0, 2 and 7: 3.65, 1.66 and 2.76 m with it on against 2.01,
    0.97 and 1.29 m with it off, and the same ordering after the detection fix."""
    absolute_fix: bool = True
    """Stage B. Apply marker observations as an absolute correction after the
    relative stage, with the gain from a covariance integrated since the last fix,
    rather than mixing them into the registration's own information."""
    remap_degenerate: bool = False
    """Optional hard version of the above: project the correction out of directions
    whose eigenvalue is below the floor. Off by default; the soft fusion supersedes it."""
    remap_eigenvalue_floor: float = 0.0
    """Absolute translational eigenvalue below which a direction is remapped."""


_STATE = np.array([2, 3, 4])
"""Indices of (yaw, x, y) in the [rot, trans] twist ordering. The whole absolute
stage lives in this subspace: horizontal, and rotation only about the vertical."""


@dataclasses.dataclass
class _AnchorTerm:
    """One observation of one known marker, with everything a fix needs from it."""

    slot: int
    q: np.ndarray
    """Where the observed marker is, under the current estimate."""
    anchor: np.ndarray
    """Where the anchor's record says it is."""
    R_obs: np.ndarray
    """2x2 horizontal measurement covariance: drop-time plus current observation."""
    Q_since: np.ndarray
    """3x3 covariance over (yaw, x, y) of the pose relative to this anchor."""


def _jacobian(q: np.ndarray, sensor_t: np.ndarray, use_yaw: bool) -> np.ndarray:
    """Rows of the 2x3 Jacobian of the predicted marker position in (yaw, x, y).

    Linearised about the current sensor position, a yaw increment moves a world
    point by ``dpsi * z_hat x (q - t)``, so the yaw column is the in-plane
    perpendicular of the lever from sensor to anchor. Anchors close to the sensor
    contribute almost nothing to yaw, which is correct: heading needs a baseline.
    With ``use_yaw`` false the column is zero, which is how a single anchor is kept
    away from attitude.
    """
    J = np.zeros((2, 3))
    if use_yaw:
        lever = np.asarray(q, dtype=float)[:2] - np.asarray(sensor_t, dtype=float)[:2]
        J[:, 0] = np.array([-lever[1], lever[0]])
    J[0, 1] = 1.0
    J[1, 2] = 1.0
    return J


def _psd(M: np.ndarray) -> np.ndarray:
    """Symmetrise and clip negative eigenvalues to zero.

    Differences of covariances appear all over the relative formulation, and a
    difference of two covariances is only positive semidefinite in exact
    arithmetic. Clipping is the honest repair: it never removes uncertainty that
    the arithmetic actually supports.
    """
    M = 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)
    w, V = np.linalg.eigh(M)
    if np.all(w >= 0.0):
        return M
    return V @ np.diag(np.maximum(w, 0.0)) @ V.T


@dataclasses.dataclass
class OdomStep:
    T: np.ndarray
    """Estimated sensor pose in the estimator's world frame."""
    T_prior: np.ndarray
    localizability: Localizability | None
    converged: bool
    num_inliers: int
    error: float
    n_scan_points: int
    registered: bool
    remapped_axes: int = 0
    landmarks_used: int = 0
    yaw_fixed: int = 0


class Odometry:
    def __init__(
        self,
        T0: np.ndarray,
        cfg: OdometryConfig = OdometryConfig(),
        prior_spec: MotionPriorSpec = MotionPriorSpec(),
        dt: float = 0.5,
        lidar_spec=None,
        landmark_spec: LandmarkSpec = LandmarkSpec(),
        marker_height: float = 1.0,
        marker_width: float = 0.15,
        marker_thickness: float = 0.0,
    ):
        self.cfg = cfg
        self.marker_thickness = float(marker_thickness)
        self.T = np.array(T0, dtype=float)
        self.map = LocalMap(
            radius=cfg.map_radius,
            resolution=cfg.map_resolution,
            min_observations=cfg.map_min_observations,
        )
        self.prior_spec = prior_spec
        self.dt = float(dt)
        self.landmarks = LandmarkBook()
        # covariance of the pose error accumulated since the last absolute fix,
        # in the twist frame centred on the current sensor position
        self.P = np.diag([1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6])
        # Running sum of the per-step prior covariance, in the world frame. A
        # difference of two marks on this is the odometry accumulated between them,
        # which is what weights every marker fix. It is the prior only: the
        # registration's own information is deliberately not subtracted, because a
        # scan-to-map registration is relative to a map the estimate built and
        # cannot certify how far the robot has come since a marker went down.
        self._Q_cum = np.zeros((6, 6))
        self._frame_rel = np.zeros((3, 3))
        """Covariance of the pose against the shared frame at this robot's last fix in it."""
        self._frame_q_ref = np.zeros((6, 6))
        """Accumulated odometry covariance at this robot's last fix in the shared frame.

        A marker this robot mounted marks ``_Q_cum`` itself at the drop, because the drop is a
        fix: the anchor inherits the pose exactly. A marker another robot mounted cannot, because
        this robot was somewhere else when it went on, so it starts from here instead, and every
        fix moves it forward."""
        self.scale = 1.0
        """Estimated multiplicative error of the prior's distance scale."""
        self.scale_var = float(prior_spec.odom_scale_sigma) ** 2
        self.travelled = 0.0
        """Distance travelled according to the estimate. The lever arm the scale
        observation is divided by."""
        self._travelled_ref: dict[int, float] = {}
        self._since_fix: list[int] = []
        self.n_fixes = 0
        self.fix_log: list[dict] = []
        """One record per absolute fix: what it moved, and what weighted it.

        Instrumentation rather than state. A fix that is silently small, or in the
        wrong direction, looks exactly like a fix that worked from outside the
        estimator, and the milestone 9 diagnostics needed to tell the two apart."""
        self.landmark_spec = landmark_spec
        self.lidar_spec = lidar_spec
        self.marker_height = marker_height
        self.marker_width = marker_width
        self.n_steps = 0

    # ---- landmarks ------------------------------------------------------

    def register_landmark(self, slot: int, offset_sensor: np.ndarray) -> None:
        """Record where the robot believes it just bolted a marker.

        The offset is in the sensor frame at the moment of the drop, so the
        landmark inherits exactly the pose error the estimate had then, and no
        more. That is the honest accounting: a marker carries the localizability
        the robot had, not the localizability it wishes it had.

        Two records are kept. The absolute position, which is what the residual is
        computed against, and the relative record: the drop pose, the offset in that
        frame, the drop-time observation covariance, and a mark on the accumulated
        odometry covariance. The second is what a fix is weighted by.
        """
        offset = np.asarray(offset_sensor, dtype=float)
        world = self.T[:3, :3] @ offset + self.T[:3, 3]
        self.landmarks.register(
            slot,
            world,
            self.n_steps,
            covariance=self.P.copy(),
            T_drop=self.T.copy(),
            offset_drop=offset,
            R_drop=self._drop_covariance(offset),
            q_ref=self._Q_cum.copy(),
            rel=np.zeros((3, 3)),
        )
        self._travelled_ref[slot] = self.travelled
        # it inherits the error the estimate has now, so it moves with the next fix
        self._since_fix.append(slot)

    def register_foreign_landmark(self, slot: int, position_world: np.ndarray,
                                  R_drop: np.ndarray, normal: np.ndarray) -> None:
        """Record a marker another robot mounted, in that robot's frame.

        This is the whole of what one robot has to tell another for the second to use the first's
        markers: where the strip is, how well the first robot knew that when it bolted it on, and
        which way the face points. Not a map, not a trajectory, not a covariance over the first
        robot's whole run.

        The position is taken as exact, because this robot is localizing **in the other robot's
        frame** and in that frame the strip's coordinates are the definition of it. What is
        uncertain is the observation that placed it, ``R_drop``, and that is carried as the
        measurement noise of every fix from this anchor, exactly as a marker of this robot's own
        carries its own drop-time observation. Weighting instead by the other robot's absolute
        covariance is the mistake ``relative_anchors`` exists to avoid: it would make a strip
        250 m into a tunnel look worthless precisely where it is the only thing worth having.

        The relation this anchor constrains is this robot's pose against the shared frame, so the
        prior on it is this robot's own odometry since it was last fixed in that frame, which is
        ``_frame_q_ref``: zero at the start, and reset at every fix. A marker of this robot's own
        gets that mark at the drop instead, because the drop is a fix.
        """
        R = np.asarray(R_drop, dtype=float)
        if R.shape != (2, 2):
            raise ValueError(f"R_drop must be 2x2 horizontal, got {R.shape}")
        n = np.asarray(normal, dtype=float)[:2]
        if float(np.linalg.norm(n)) < 1e-9:
            raise ValueError("normal must be a non-zero horizontal direction")
        self.landmarks.register(
            slot,
            np.asarray(position_world, dtype=float),
            self.n_steps,
            covariance=None,
            T_drop=None,
            offset_drop=None,
            R_drop=R,
            q_ref=self._frame_q_ref.copy(),
            rel=np.zeros((3, 3)),
            normal=n / float(np.linalg.norm(n)),
            foreign=True,
        )
        self._travelled_ref[slot] = self.travelled

    def _drop_covariance(self, offset_sensor: np.ndarray) -> np.ndarray:
        """2x2 horizontal covariance of registering the strip as it goes on.

        The drop is an observation like any other, at the range the strip is bolted
        on and with the beam count that geometry gives, so it is the same model
        evaluated there. It is the floor on how well any later fix can place the
        robot relative to this anchor: no amount of re-observing a marker beats the
        accuracy with which it was put down.
        """
        if self.lidar_spec is None:
            return np.eye(2) * self.landmark_spec.min_sigma_m**2
        n = predicted_beam_count(
            offset_sensor, self.lidar_spec, self.marker_height, self.marker_width
        )
        omega = measurement_information(
            offset_sensor,
            n,
            self.lidar_spec,
            self.landmark_spec,
            self.marker_height,
            self.marker_width,
        )
        R = self.T[:3, :3]
        return np.linalg.pinv((R @ omega @ R.T)[:2, :2])

    def _relative_covariance(self, lm) -> np.ndarray:
        """Covariance of the current pose relative to one anchor, over (yaw, x, y).

        The odometry accumulated since that anchor's record was last reset, which is
        at the drop and again after every fix computed from it. Nothing here depends
        on how far the run has travelled in total, which is the entire point.
        """
        if lm.foreign:
            # A teammate's marker is not relative to a drop of this robot's, so what separates the
            # pose from it is the odometry since this robot was last fixed in the shared frame,
            # wherever in that frame the fix came from. Keeping a mark per anchor instead would
            # charge a strip first seen at 200 m for every metre since it was told about, although
            # the robot was fixed at 190 m by the strip before it.
            grown = (self._Q_cum - self._frame_q_ref)[np.ix_(_STATE, _STATE)]
            return _psd(self._frame_rel + grown)
        if lm.q_ref is None:
            return np.zeros((3, 3))
        grown = (self._Q_cum - lm.q_ref)[np.ix_(_STATE, _STATE)]
        base = np.zeros((3, 3)) if lm.rel is None else lm.rel
        return _psd(base + grown)

    def _strip_fit(self, lm, sensor_position: np.ndarray, obs=None):
        """What the strip fit is worth for this view of this strip: its expected error as a
        world-frame 3-vector, and its spread along and across the line of sight.

        Both come from one Monte Carlo of the real fit at this geometry. The strip faces the way
        the robot stood when it mounted it: its normal is the horizontal direction from the
        mounting point back to the drop pose."""
        if self.marker_thickness <= 0.0 or self.lidar_spec is None:
            return np.zeros(3), None
        if lm.normal is not None:
            normal = np.asarray(lm.normal, dtype=float)[:2]
        elif lm.T_drop is not None and lm.offset_drop is not None:
            normal = -(lm.T_drop[:3, :3] @ lm.offset_drop)[:2]
        else:
            return np.zeros(3), None
        los = lm.position[:2] - sensor_position[:2]
        distance = float(np.linalg.norm(los))
        if np.linalg.norm(normal) < 1e-9 or distance < 1e-6:
            return np.zeros(3), None
        args = (los, normal, distance, self.lidar_spec, self.marker_width, self.marker_thickness,
                self.marker_height, float(lm.position[2] - sensor_position[2]))
        columns = getattr(obs, "n_columns", None)
        bias = np.zeros(3)
        if self.cfg.correct_strip_bias:
            b = strip_fit_bias(*args, n_columns=columns)
            bias = np.array([b[0], b[1], 0.0])
        sigma = strip_fit_sigma(*args, n_columns=columns) if self.cfg.measure_strip_spread else None
        return bias, sigma

    def _landmark_terms(self, T: np.ndarray, observations: list) -> list:
        """One ``_AnchorTerm`` per observation of a known marker."""
        terms = []
        if not observations or self.lidar_spec is None:
            return terms
        R = T[:3, :3]
        t = T[:3, 3]
        for obs in observations:
            lm = self.landmarks.get(obs.slot)
            if lm is None:
                continue
            q = R @ np.asarray(obs.point_sensor, dtype=float) + t
            bias, fit_sigma = self._strip_fit(lm, t, obs)
            q = q - bias
            omega_sensor = measurement_information(
                obs.point_sensor,
                obs.n_beams,
                self.lidar_spec,
                self.landmark_spec,
                self.marker_height,
                self.marker_width,
                seen_width_m=getattr(obs, "seen_width_m", None),
                n_columns=getattr(obs, "n_columns", None),
                fit_sigma=fit_sigma,
            )
            R_meas = np.linalg.pinv((R @ omega_sensor @ R.T)[:2, :2])
            if self.cfg.relative_anchors:
                R_drop = np.zeros((2, 2)) if lm.R_drop is None else lm.R_drop
                R_obs = R_meas + R_drop
                Q_since = self._relative_covariance(lm)
            else:
                # the old formulation: the anchor's absolute covariance is folded
                # into the measurement, and the pose prior is the absolute one
                R_obs = R_meas
                if lm.covariance is not None:
                    R_obs = R_obs + lm.covariance[np.ix_([3, 4], [3, 4])]
                Q_since = _psd(self.P[np.ix_(_STATE, _STATE)])
            lm.n_observations += 1
            terms.append(_AnchorTerm(obs.slot, q, lm.position, R_obs, Q_since))
        return terms

    def _landmark_system(
        self, T: np.ndarray, observations: list
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Translation-only normal equations, kept for tests and for the one-anchor case.

        A single point landmark fixes position, not attitude. Letting it touch the
        rotation makes the filter unstable: one observation constrains rotation only
        through its lever arm, so the rotational posterior is near-singular, and
        chaining that through the anchor covariances sends the correction to
        infinity within a few fixes. Two anchors are a different matter, and
        ``_yaw_system`` handles that case.
        """
        A = np.zeros((3, 3))
        b = np.zeros(3)
        terms = self._landmark_terms(T, observations)
        for term in terms:
            omega = np.linalg.inv(term.R_obs)
            A[:2, :2] += omega
            b[:2] += -omega @ (term.q - term.anchor)[:2]
        return A, b, len(terms)

    def _yaw_system(
        self, T: np.ndarray, terms: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normal equations over (yaw, x, y) from two or more anchors in view.

        Two separated anchors give a bearing and therefore a heading. That is what
        closes a mine traverse, and here it is what removes the error a
        translation-only fix cannot touch: an anchor is registered through a 1.5 m
        lever arm to the wall, so it inherits the heading error at the moment it was
        dropped, and that error is systematic rather than zero mean.

        Linearised about the current sensor position, a yaw increment moves a world
        point by ``dpsi * z_hat x (q - t)``, so the yaw column of the Jacobian is the
        in-plane perpendicular of the lever from the sensor to the anchor. Anchors
        close to the sensor contribute almost nothing, which is correct: yaw needs a
        baseline.
        """
        A = np.zeros((3, 3))
        b = np.zeros(3)
        t = T[:3, 3]
        for term in terms:
            omega = np.linalg.inv(term.R_obs)
            J = _jacobian(term.q, t, use_yaw=True)
            A += J.T @ omega @ J
            b += -J.T @ omega @ (term.q - term.anchor)[:2]
        return A, b

    def _propagate_covariance(
        self,
        prior_info: np.ndarray,
        translation: np.ndarray,
        registration_info: np.ndarray | None = None,
    ) -> None:
        """P <- F P F^T + Q, then fold in whatever the registration genuinely observed.

        F is the change of twist centre as the sensor moves. The prior always adds
        uncertainty. The registration removes it, but only along the directions that
        survived the absolute guard, which is the point: in a blind tunnel the
        along-track direction is zeroed and P grows there, while in a stretch with
        real structure it survives and P collapses.

        Without this the covariance never shrank except at a fix, so the estimator
        believed it was lost even where the geometry had just told it exactly where
        it was, and a marker anchor carrying a stale drop-time error would win
        against a registration that was locally correct. Markers then made things
        worse in structured tunnels, by 1.3 m against 0.5 m over 300 m.
        """
        F = np.eye(6)
        F[3:, :3] = -_skew(np.asarray(translation, dtype=float))
        Q_step = np.linalg.inv(prior_info)
        if self.cfg.credit_registration and registration_info is not None:
            H = 0.5 * (registration_info + registration_info.T)
            try:
                Q_step = np.linalg.inv(np.linalg.inv(Q_step) + H)
            except np.linalg.LinAlgError:
                pass
            Q_step = 0.5 * (Q_step + Q_step.T)
        # The relative accumulator takes the step covariance as it stands, without
        # the change of twist centre. Over one step the lever arm is the step
        # length, so the coupling it drops is second order in the step, and the
        # translation block, which is the only part a horizontal fix reads, is
        # unaffected by it.
        self._Q_cum = self._Q_cum + 0.5 * (Q_step + Q_step.T)
        P = F @ self.P @ F.T + Q_step
        if registration_info is not None:
            H = 0.5 * (registration_info + registration_info.T)
            try:
                P = np.linalg.inv(np.linalg.inv(P) + H)
            except np.linalg.LinAlgError:
                pass
            P = 0.5 * (P + P.T)
        self.P = P

    def _resurvey(self, observations: list) -> int:
        """Re-register anchors the estimator now knows better than.

        An anchor is a record of where the robot thought it was. If the geometry
        later tells the robot where it is to a centimetre, every anchor carrying a
        stale drop-time error is now the worse source, and left alone it will drag
        the pose back to that error as soon as the next blind stretch begins. So
        when the pose covariance falls below an anchor's own, the anchor is
        re-surveyed from the current estimate, which is what a survey crew does with
        a control point when a better fix becomes available.
        """
        updated = 0
        P_trace = float(np.trace(self.P[np.ix_([3, 4], [3, 4])]))
        for term in self._landmark_terms(self.T, observations):
            lm = self.landmarks.get(term.slot)
            # A marker a teammate mounted is never re-surveyed. Its coordinates are the shared
            # frame's definition; overwriting them with this robot's estimate would quietly
            # redefine the frame the teammate is still using.
            if lm is None or lm.foreign or lm.covariance is None:
                continue
            if P_trace < float(np.trace(lm.covariance[np.ix_([3, 4], [3, 4])])):
                lm.position = term.q
                lm.covariance = self.P.copy()
                # the anchor has been re-registered from the current pose, so its
                # relative record starts again from here
                lm.T_drop = self.T.copy()
                lm.offset_drop = self.T[:3, :3].T @ (term.q - self.T[:3, 3])
                lm.R_drop = np.array(term.R_obs, dtype=float)
                lm.q_ref = self._Q_cum.copy()
                lm.rel = np.zeros((3, 3))
                updated += 1
        return updated

    def _absolute_fix(self, observations: list) -> tuple[int, int]:
        """Stage B. Correct the pose, the map and the recent landmarks together.

        The constraint is relative, not absolute. A marker says where the robot is
        with respect to the pose that dropped it, so the prior on that relation is
        the odometry accumulated since the drop, and the measurement noise is the
        drop-time observation plus the current one. Neither term grows with distance
        from the start of the run, which is what makes a marker deep in a tunnel
        worth as much as one near the portal.

        Horizontal only, and translation only unless two anchors are in view. The
        measurement carries no vertical information by construction, and one point
        landmark constrains position and not attitude: solving in three dimensions
        anyway lets the unconstrained variance grow without bound through the anchor
        chain, and the fix then makes metre-scale corrections in z off numerical
        coupling alone.

        Returns the number of anchors used and whether yaw was part of the fix.
        """
        terms = self._landmark_terms(self.T, observations)
        if not terms:
            return 0, 0

        t_before = self.T[:3, 3].copy()
        distinct = {term.slot for term in terms}
        use_yaw = self.cfg.yaw_fix and len(distinct) >= 2

        # The prior is the pose relative to the best known anchor in view. Every
        # other anchor is then uncertain relative to that one by the difference
        # between their accumulated odometry, and that difference goes into their
        # measurement noise where it belongs. With one anchor the extra term is
        # zero and this reduces to the textbook update.
        best = min(terms, key=lambda t: float(np.trace(t.Q_since[1:, 1:])))
        P_rel = _psd(best.Q_since)
        # the leg this fix is measured over, captured before the anchor records are
        # reset below, because that reset is what the leg is measured from
        leg = self.travelled - self._travelled_ref.get(best.slot, self.travelled)

        A = np.zeros((3, 3))
        b = np.zeros(3)
        extras = {}
        for term in terms:
            extra = _psd(term.Q_since[1:, 1:] - P_rel[1:, 1:])
            extras[term.slot] = extra
            W = np.linalg.inv(term.R_obs + extra)
            J = _jacobian(term.q, self.T[:3, 3], use_yaw=use_yaw)
            A += J.T @ W @ J
            b += -J.T @ W @ (term.q - term.anchor)[:2]

        # A ridge rather than a pseudo-inverse: right at the drop the relative
        # covariance is zero, which means the pose is already exactly where the
        # anchor says, and the correction must be zero rather than unconstrained.
        S = np.linalg.inv(P_rel + 1e-10 * np.eye(3)) + A
        try:
            x = np.linalg.solve(S, b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(S, b, rcond=None)[0]

        dpsi = float(x[0]) if use_yaw else 0.0
        dx, dy = float(x[1]), float(x[2])
        R_corr = rotz(dpsi)
        t_corr = np.array([dx, dy, 0.0])
        # the yaw acts about the current sensor position, not about the world origin
        offset = t_before - R_corr @ t_before + t_corr

        self.T = make_T(R_corr @ self.T[:3, :3], R_corr @ self.T[:3, 3] + offset)
        self.map.apply_transform(R_corr, offset)

        # Landmarks registered since the last fix drifted with the estimate, so they
        # move with the correction. The ones this fix was computed from do not: they
        # are the anchor, and moving them with the pose would make the whole thing
        # circular, leaving a marker that tracks the drift instead of opposing it.
        for slot in self._since_fix:
            if slot in distinct:
                continue
            lm = self.landmarks.get(slot)
            if lm is not None:
                lm.position = R_corr @ lm.position + offset
        self._since_fix = [slot for slot in self._since_fix if slot in distinct]

        posterior = _psd(np.linalg.inv(S))
        best_lm = self.landmarks.get(best.slot)
        heading = self.T[:3, 0].copy()
        q_along = float(heading[:2] @ P_rel[1:, 1:] @ heading[:2])
        r_along = float(heading[:2] @ best.R_obs @ heading[:2])
        self.fix_log.append(
            {
                "step": self.n_steps,
                "n_terms": len(terms),
                "slots": sorted(distinct),
                "dx": dx,
                "dy": dy,
                "dpsi": dpsi,
                "along": float(heading[:2] @ np.array([dx, dy])),
                "residual_along": float(
                    heading[:2] @ (best.q - best.anchor)[:2]
                ),
                "range_m": float(np.linalg.norm((best.anchor - t_before)[:2])),
                "q_since_along": q_along,
                "r_along": r_along,
                "gain_along": q_along / max(q_along + r_along, 1e-18),
                "used_yaw": int(use_yaw),
            }
        )
        if self.cfg.relative_anchors:
            # This robot has just been fixed in the shared frame, so its pose against that frame
            # is the posterior of this fix, and grows from here by its own odometry alone. Markers
            # a teammate mounted are held against this rather than against a mark of their own:
            # they were all put down in one frame, so being fixed against any of them is being
            # fixed against all of them, including the ones still out of sight.
            self._frame_rel = posterior.copy()
            self._frame_q_ref = self._Q_cum.copy()
        if self.cfg.relative_anchors:
            # Each anchor's relative record restarts here. There is no floor and no
            # reset to an absolute number: the relative covariance cannot fall below
            # R_drop + R_meas because those terms are inside S, which is the correct
            # bound and the reason repeated observations of one strip can be treated
            # as independent without compounding into a confidence nothing supports.
            for term in terms:
                lm = self.landmarks.get(term.slot)
                if lm is None:
                    continue
                rel = posterior.copy()
                rel[1:, 1:] = rel[1:, 1:] + extras[term.slot]
                lm.rel = rel
                lm.q_ref = self._Q_cum.copy()
                self._travelled_ref[term.slot] = self.travelled

            # Absolute covariance is bookkeeping: where the whole chain might be
            # with respect to the start of the run. Reported, never used in a gain.
            if best_lm is not None and best_lm.covariance is not None:
                P_abs = self.P.copy()
                P_abs[np.ix_(_STATE, _STATE)] = (
                    best_lm.covariance[np.ix_(_STATE, _STATE)] + posterior
                )
                self.P = _psd(P_abs)
        else:
            # The old formulation drove the gain from the absolute covariance, and
            # needed a floor at the anchor's own covariance to stop repeated views
            # of one strip compounding into a confidence the measurement could not
            # support. Kept so the ablation can measure what the change was worth.
            P_meas = posterior.copy()
            if best_lm is not None and best_lm.covariance is not None:
                P_meas[1:, 1:] = _loewner_floor(
                    P_meas[1:, 1:], best_lm.covariance[np.ix_([3, 4], [3, 4])]
                )
            self.P[np.ix_(_STATE, _STATE)] = P_meas
        if self.cfg.estimate_scale:
            self._update_scale(
                leg, float(heading[:2] @ (best.q - best.anchor)[:2]), q_along, r_along
            )
        self.n_fixes += 1
        return len(terms), int(use_yaw)

    def _update_scale(
        self, leg: float, residual_along: float, q_along: float, r_along: float
    ) -> None:
        """One scalar Kalman step on the prior's distance scale.

        The residual along the direction of travel, accumulated over a leg of length
        L, is an observation of ``(s - 1) * L``. Short legs carry almost no
        information about a multiplicative error, which the lever arm handles on its
        own: with L near zero the gain is near zero.

        The residual is used twice, once for the pose and once here. A jointly
        augmented state would handle the correlation properly; sequentially it makes
        the scale estimate slightly over-confident, which the random walk works
        against. The alternative is not to estimate it at all, and the alternative
        costs linear growth over every leg of the traverse.
        """
        L = float(leg)
        if L <= 1e-6:
            return
        z = float(residual_along)
        S = L * L * self.scale_var + max(q_along + r_along, 1e-12)
        K = L * self.scale_var / S
        self.scale += K * (z - L * (self.scale - 1.0))
        self.scale_var = max((1.0 - K * L) * self.scale_var, 1e-10)
        # a sanity bound far outside anything the prior spec allows, so a pathological
        # fix cannot run the prior away
        self.scale = float(np.clip(self.scale, 0.8, 1.2))

    def step(
        self,
        scan_points_sensor: np.ndarray,
        delta_T_prior: np.ndarray,
        marker_observations: list | None = None,
    ) -> OdomStep:
        cfg = self.cfg
        T_before = self.T
        delta_T_prior = np.asarray(delta_T_prior, dtype=float)
        if cfg.estimate_scale and abs(self.scale - 1.0) > 1e-12:
            delta_T_prior = delta_T_prior.copy()
            delta_T_prior[:3, 3] = delta_T_prior[:3, 3] / self.scale
        step_distance = float(np.linalg.norm(delta_T_prior[:3, 3]))
        self.travelled += step_distance
        self.scale_var += (cfg.scale_random_walk_per_m**2) * step_distance
        T_pred = self.T @ delta_T_prior
        pts = np.asarray(scan_points_sensor, dtype=float)
        reg_pts = pts
        if cfg.registration_range is not None and pts.shape[0] > 0:
            reg_pts = pts[np.linalg.norm(pts, axis=1) <= cfg.registration_range]

        prior_info = prior_information(
            self.prior_spec, self.dt, delta_T_prior, R_world_body=self.T[:3, :3]
        )

        if reg_pts.shape[0] < cfg.min_scan_points or len(self.map) < cfg.min_scan_points:
            self.T = T_pred
            self._propagate_covariance(prior_info, T_pred[:3, 3] - T_before[:3, 3])
            if pts.shape[0] > 0:
                self.map.add(transform_points(self.T, pts), centre=self.T[:3, 3])
            n_lm, yawed = (
                self._absolute_fix(marker_observations or []) if cfg.absolute_fix else (0, 0)
            )
            self.n_steps += 1
            return OdomStep(
                T=self.T.copy(),
                T_prior=T_pred,
                localizability=None,
                converged=False,
                num_inliers=0,
                error=float("nan"),
                n_scan_points=int(pts.shape[0]),
                registered=False,
                landmarks_used=n_lm,
                yaw_fixed=yawed,
            )

        src_world = transform_points(T_pred, reg_pts)
        res = small_gicp.align(
            self.map.points,
            src_world,
            registration_type=cfg.registration_type,
            downsampling_resolution=cfg.downsampling_resolution,
            max_correspondence_distance=cfg.max_correspondence_distance,
            max_iterations=cfg.max_iterations,
            num_threads=cfg.num_threads,
        )
        H = np.array(res.H, dtype=float)
        loc = analyse_hessian(H, int(res.num_inliers))
        T_corr = np.array(res.T_target_source, dtype=float)

        # ---- stage A: relative -------------------------------------------
        H_guarded = None
        if cfg.fuse_prior:
            T_corr, H_guarded = _fuse(
                T_corr,
                H,
                prior_info,
                sensor_position=T_pred[:3, 3],
                n_inliers=int(res.num_inliers),
                residual=float(res.error),
                calibrate=cfg.calibrate_information,
                gicp_ratio_floor=cfg.gicp_ratio_floor,
            )

        remapped = 0
        if cfg.remap_degenerate and cfg.remap_eigenvalue_floor > 0.0:
            T_corr, remapped = _remap_correction(T_corr, loc, cfg.remap_eigenvalue_floor)

        self.T = T_corr @ T_pred
        self._propagate_covariance(
            prior_info, self.T[:3, 3] - T_before[:3, 3], registration_info=H_guarded
        )
        self.map.add(transform_points(self.T, pts), centre=self.T[:3, 3])

        # ---- stage B: absolute -------------------------------------------
        n_lm, yawed = (0, 0)
        if cfg.absolute_fix:
            obs = marker_observations or []
            if cfg.resurvey_anchors:
                self._resurvey(obs)
            n_lm, yawed = self._absolute_fix(obs)
        self.n_steps += 1
        return OdomStep(
            T=self.T.copy(),
            T_prior=T_pred,
            localizability=loc,
            converged=bool(res.converged),
            num_inliers=int(res.num_inliers),
            error=float(res.error),
            n_scan_points=int(reg_pts.shape[0]),
            registered=True,
            remapped_axes=remapped,
            landmarks_used=n_lm,
            yaw_fixed=yawed,
        )


def _loewner_floor(P: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Lift ``P`` until it dominates ``floor`` in the Loewner order.

    Both matrices are projected onto the floor's eigenbasis, which is the sensible
    common basis here because the floor is the anchor's own covariance and its
    shape is the thing being imposed. Only deficient directions are lifted, so
    ``P`` is never made larger than it has to be, and the result dominates the
    floor along every eigendirection of the floor.

    A diagonal maximum would be cheaper and is what the two reduce to when they are
    near aligned, but they are not reliably near aligned: the anchor's covariance
    is elongated along the tunnel while the posterior is elongated along whatever
    the registration last failed to observe, and on a curve those differ by tens of
    degrees.
    """
    P = 0.5 * (np.asarray(P, dtype=float) + np.asarray(P, dtype=float).T)
    floor = 0.5 * (np.asarray(floor, dtype=float) + np.asarray(floor, dtype=float).T)
    d_floor, V = np.linalg.eigh(floor)
    d_post = np.einsum("ij,jk,ki->i", V.T, P, V)
    deficit = np.maximum(0.0, d_floor - d_post)
    if not np.any(deficit > 0.0):
        return P
    return P + V @ np.diag(deficit) @ V.T


def _so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _shift_twist_frame(t: np.ndarray) -> np.ndarray:
    """6x6 map from a world-origin twist to one about the point ``t``.

    ``small_gicp`` linearises with a left perturbation, so its Hessian is written
    in coordinates where a rotation increment turns the world about the origin.
    A hundred and sixty metres down a tunnel that makes the rotation block
    enormous and couples it hard into translation, which is fine for GICP alone
    but wrong to combine with a prior written about the sensor. Everything is
    therefore moved to a parameterisation centred on the current sensor position
    before the three information matrices meet, and moved back afterwards.
    """
    M = np.eye(6)
    M[3:, :3] = -_skew(np.asarray(t, dtype=float))
    return M


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _fuse(
    T_corr: np.ndarray,
    H: np.ndarray,
    prior_info: np.ndarray,
    sensor_position: np.ndarray,
    n_inliers: int,
    residual: float,
    calibrate: bool,
    landmark_info: np.ndarray | None = None,
    landmark_rhs: np.ndarray | None = None,
    gicp_ratio_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine registration, prior and landmark measurements into one correction.

    The prior's mean correction is zero by construction (the scan was already
    placed at the predicted pose), so the MAP correction solves

        (H_cal + H_prior + H_landmark) x = H_cal x_gicp + b_landmark

    which passes well-observed directions through untouched, collapses
    unobserved ones onto the prior, and lets a marker in view override both along
    the direction it constrains.

    This is a blend applied after an unconstrained GICP has already converged, not
    a factor inside its Gauss-Newton loop. Along a degenerate direction the two
    differ: a factor would steer the iterations, whereas this only rescales where
    they ended up. A tightly coupled formulation would be the stronger design and
    is deliberately out of scope here, because ``small_gicp`` does not expose its
    iteration. The consequence is one-sided, since the unconstrained solve can
    wander further before the blend reins it in, so the baseline is if anything
    understated.

    Returns the corrected transform and the calibrated, guarded registration
    information in the sensor-centred frame.
    """
    x_world = np.concatenate([so3_log(T_corr[:3, :3]), T_corr[:3, 3]])
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    if calibrate:
        dof = max(n_inliers - 6, 1)
        scale = dof / residual if np.isfinite(residual) and residual > 1e-12 else 1.0
        H = H * scale

    if gicp_ratio_floor is not None:
        Ht = 0.5 * (H[3:, 3:] + H[3:, 3:].T)
        ev, V = np.linalg.eigh(Ht)
        ev = np.maximum(ev, 0.0)
        keep = ev >= gicp_ratio_floor * max(ev[-1], 1e-30)
        H = H.copy()
        H[3:, 3:] = V @ np.diag(np.where(keep, ev, 0.0)) @ V.T
        H[:3, 3:] = 0.0
        H[3:, :3] = 0.0

    M = _shift_twist_frame(sensor_position)
    Minv = np.eye(6)
    Minv[3:, :3] = -M[3:, :3]

    H_s = Minv.T @ H @ Minv
    x_s = M @ x_world

    A = H_s + np.asarray(prior_info, dtype=float)
    rhs = H_s @ x_s
    if landmark_info is not None:
        A = A + landmark_info
        rhs = rhs + landmark_rhs

    try:
        x_fused_s = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        x_fused_s = np.linalg.lstsq(A, rhs, rcond=None)[0]

    x_fused = Minv @ x_fused_s
    return make_T(_so3_exp(x_fused[:3]), x_fused[3:]), H_s


def _remap_correction(
    T_corr: np.ndarray, loc: Localizability, eigenvalue_floor: float
) -> tuple[np.ndarray, int]:
    """Zero the translational correction along directions the Hessian says are weak.

    This is the classical solution-remapping guard (Zhang, Kaess and Singh 2016).
    It is off by default in milestone 1; milestone 2 and 3 switch it on so the
    prior, not the registration, carries the degenerate axes.
    """
    weak = loc.eigenvalues < eigenvalue_floor
    if not weak.any():
        return T_corr, 0
    V = loc.eigenvectors
    keep = V[:, ~weak] @ V[:, ~weak].T
    out = T_corr.copy()
    out[:3, 3] = keep @ T_corr[:3, 3]
    return out, int(weak.sum())
