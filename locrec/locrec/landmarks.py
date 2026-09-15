"""Retroreflective markers as landmarks.

A marker is a vertical strip bolted to the tunnel wall. The robot drops it while
it still knows where it is, registers its position in the estimator's own frame at
that moment, and then, for as long as the strip stays in view, every scan that
picks it up ties the pose back to that registered position. That is the "carry it,
leave it behind" half of the claim: a marker exports the localizability the robot
had at drop time into the blind stretch ahead.

Nothing here removes accumulated drift. A marker anchors the pose to wherever the
robot *thought* it was when the marker went down, so drift is bounded by the error
at drop time plus whatever accrues after the marker leaves view, not driven to
zero. The measured curve in milestone 2 is that bound, not a loop closure.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .lidar import LidarSpec

__all__ = [
    "LandmarkSpec",
    "LandmarkBook",
    "measurement_information",
    "predicted_beam_count",
    "fit_strip_centre",
    "strip_fit_bias",
    "strip_fit_sigma",
    "beam_columns",
]


@dataclasses.dataclass(frozen=True)
class LandmarkSpec:
    """Noise model for one marker observation.

    The three error sources are measured, not guessed at:

    * range noise, straight from the LiDAR spec;
    * beam quantisation, the returned centroid can sit anywhere inside the beam
      spacing, so a uniform distribution of width ``r * beam_step`` with standard
      deviation ``r * beam_step / sqrt(12)``;
    * target extent, the centroid of a handful of returns on a 1 m strip is not
      the strip's centre, and with ``n`` returns that residual scatters like
      ``extent / sqrt(12 n)``.

    All three shrink the same way with more returns except the range term, which
    averages as ``1 / sqrt(n)``.
    """

    use_beam_quantisation: bool = True
    use_target_extent: bool = True
    use_vertical: bool = False
    """Whether the measurement constrains the vertical at all. It should not.

    The returns land wherever the elevation channels happen to cross a 1 m strip,
    so the vertical component of their centroid slides by a good fraction of the
    strip's height as the geometry changes. That is a geometry-dependent bias, not
    zero-mean noise, and believing it is expensive: forcing the covariance to 1 mm
    and leaving the vertical in took drift from about 1 m to about 40 m, because
    the pose was dragged along behind a wandering centroid. The strip is a
    horizontal-plane landmark. Its height is a detection aid, nothing more."""
    min_sigma_m: float = 0.01
    """Floor on any axis, so a lucky geometry cannot claim more than a centimetre."""


@dataclasses.dataclass
class Landmark:
    slot: int
    position: np.ndarray
    """Position in the estimator's world frame, fixed at registration time."""
    registered_at_step: int
    covariance: np.ndarray | None = None
    """Absolute pose covariance at the moment the marker was registered.

    Bookkeeping only. It says how far from the start of the run the whole chain
    might be, which is a reasonable thing to report and a wrong thing to weight a
    fix with: a marker constrains where the robot is *relative to the drop*, and
    that relation is only as uncertain as the odometry accumulated since. Weighting
    by this instead makes every marker deep in a tunnel look worthless, because the
    absolute covariance grows with distance from the start and never comes back."""
    n_observations: int = 0

    # ---- the relative record, which is what a fix is actually weighted by ----
    T_drop: np.ndarray | None = None
    """Estimated sensor pose when the strip went on the wall."""
    offset_drop: np.ndarray | None = None
    """Marker position in that drop frame. ``T_drop`` composed with this is
    ``position``, and the pair is what makes the constraint relative."""
    R_drop: np.ndarray | None = None
    """2x2 horizontal covariance of the drop-time observation itself."""
    q_ref: np.ndarray | None = None
    """Accumulated prior covariance at the last reset of this anchor's relative
    record: at the drop, and again after every fix computed from it."""
    rel: np.ndarray | None = None
    """Covariance of the pose relative to this anchor at that reset, over
    (yaw, x, y). Zero at the drop, the posterior after a fix."""
    normal: np.ndarray | None = None
    """Horizontal outward normal of the strip's face, in the world frame.

    For a marker this robot mounted itself, the face points back at the pose that mounted it, so
    the normal is implied by ``T_drop`` and ``offset_drop`` and this stays ``None``. A marker
    another robot mounted arrives without either, and its normal has to be carried explicitly:
    the fit's bias is mirrored between a strip approached from ahead and one left behind, so a
    vehicle that reads a strip off a teammate and guesses the facing gets the correction exactly
    backwards."""
    foreign: bool = False
    """True when another robot mounted this one and this robot is using it on trust.

    It changes what the record means. There is no drop pose of this robot's to be relative to, so
    the covariance of the relation is this robot's own odometry since its last fix in the shared
    frame, and the measurement noise is the other robot's drop-time observation plus this one."""


class LandmarkBook:
    """Markers the robot has dropped, with the positions it believed at the time."""

    def __init__(self) -> None:
        self._by_slot: dict[int, Landmark] = {}

    def register(
        self,
        slot: int,
        position_world: np.ndarray,
        step: int,
        covariance: np.ndarray | None = None,
        **relative,
    ) -> Landmark:
        lm = Landmark(
            slot=slot,
            position=np.asarray(position_world, dtype=float),
            registered_at_step=step,
            covariance=None if covariance is None else np.array(covariance, dtype=float),
            **relative,
        )
        self._by_slot[slot] = lm
        return lm

    def get(self, slot: int) -> Landmark | None:
        return self._by_slot.get(slot)

    def __len__(self) -> int:
        return len(self._by_slot)

    @property
    def slots(self) -> list[int]:
        return sorted(self._by_slot)


def predicted_beam_count(
    point_sensor: np.ndarray,
    lidar: LidarSpec,
    marker_height: float,
    marker_width: float,
) -> int:
    """How many returns a strip at this offset should produce.

    Needed at the moment a marker is bolted on, where there is no observation to
    count beams from but the drop-time measurement covariance still has to be
    recorded. It is the strip's angular extent divided by the beam spacing in each
    axis, floored at one return, which is the same geometry the detector's range
    calibration uses.
    """
    p = np.asarray(point_sensor, dtype=float)
    r = float(np.linalg.norm(p))
    if r < 1e-6:
        return 1
    across = marker_width / max(r * lidar.azimuth_step_rad, 1e-9)
    down = marker_height / max(r * lidar.elevation_step_rad, 1e-9)
    return max(int(across * down), 1)


def measurement_information(
    point_sensor: np.ndarray,
    n_beams: int,
    lidar: LidarSpec,
    spec: LandmarkSpec,
    marker_height: float,
    marker_width: float,
    seen_width_m: float | None = None,
    n_columns: int | None = None,
    fit_sigma: tuple[float, float] | None = None,
) -> np.ndarray:
    """3x3 information of one marker observation, expressed in the sensor frame.

    The covariance is diagonal in the ray frame (radial, horizontal tangential,
    vertical tangential) and then rotated into the sensor frame. The horizontal
    tangential axis is the one that matters: a strip on the side wall of a tunnel
    sits roughly abeam, so that direction is the tunnel axis, and that is exactly
    the direction the odometry cannot observe on its own.

    With ``use_vertical`` false, which is the default, the returned matrix is rank
    two and spans the horizontal plane. The strip constrains where the robot is
    along and across the tunnel, and says nothing about its height.

    ``n_columns`` is how many distinct azimuths the returns came from, which is what
    the horizontal axis averages over. Without it the strip's own angular extent is
    used, which is what the number of returns is predicted from at drop time.

    ``fit_sigma`` is the measured spread of the strip fit at this geometry, radial and
    tangential, from ``strip_fit_sigma``. It is the whole error of the fit, so it replaces
    the modelled terms below rather than joining them. The modelled terms stay for the
    drop, where nothing is fitted, and for a detector that does not fit the strip.
    """
    p = np.asarray(point_sensor, dtype=float)
    r = float(np.linalg.norm(p))
    if r < 1e-6:
        return np.eye(3) / spec.min_sigma_m**2
    n = max(int(n_beams), 1)

    # The basis is built in the horizontal plane, not in the ray frame, so that
    # world z is an exact null direction of the result. That matters: the point the
    # robot registered when it bolted the strip on and the centroid of the beams
    # that later land on it are at different heights, by up to half the strip. With
    # a basis tilted by the ray's elevation, that vertical mismatch leaks into the
    # horizontal residual as its sine, which at a 10 degree elevation turns a 0.4 m
    # height difference into 7 cm of fictitious along-track error, larger than the
    # measurement itself.
    horizontal = np.array([p[0], p[1], 0.0])
    nh = np.linalg.norm(horizontal)
    if nh < 1e-6:  # straight up or down: the strip says nothing about position
        return np.zeros((3, 3))
    e_r = horizontal / nh
    e_h = np.cross(np.array([0.0, 0.0, 1.0]), e_r)
    e_v = np.array([0.0, 0.0, 1.0])

    var_r = lidar.range_sigma**2 / n
    var_h = 0.0
    var_v = 0.0
    if fit_sigma is not None and min(fit_sigma) > 0.0:
        var_r, var_h = float(fit_sigma[0]) ** 2, float(fit_sigma[1]) ** 2
        floor = spec.min_sigma_m**2
        info = np.outer(e_r, e_r) / max(var_r, floor) + np.outer(e_h, e_h) / max(var_h, floor)
        if spec.use_vertical:
            var_v = (r * lidar.elevation_step_rad) ** 2 / 12.0 + marker_height**2 / 12.0
            info = info + np.outer(e_v, e_v) / max(var_v, floor)
        return info

    # How many independent samples each axis actually has. The returns arrive in
    # columns: every ring in a column shares one azimuth, so it is the columns that
    # say where the strip sits across the line of sight and the rings that say where
    # it sits up and down. Averaging the horizontal over all n returns, as this did,
    # calls a strip crossed by a single column of six rings six measurements along
    # the tunnel when it is one. At 4.4 m that claims 1.9 cm where the truth is the
    # strip's own width over root twelve, 4.3 cm, and the estimator then lets far
    # detections pull six times harder than the evidence allows
    # (docs/failures.md number 31). Range is different: the rings do give n
    # independent looks at it, so var_r keeps dividing by n.
    columns = float(n_columns) if n_columns else marker_width / max(r * lidar.azimuth_step_rad, 1e-9)
    columns = min(max(columns, 1.0), float(n))
    rings = max(n / columns, 1.0)

    # Foreshortening. A strip seen end on hides its own far half, so what comes
    # back is its near edge and the range is short by up to half a width. The
    # geometric fit removes most of that, and the rest has to be paid for here, or
    # the filter treats a systematic offset as independent evidence and follows it.
    # The detector reports how much of the strip's width it actually saw, so the
    # shortfall against the known width is the foreshortening, and the strip's
    # height stays out of the horizontal measurement where it belongs.
    if seen_width_m is not None and marker_width > 0:
        hidden = np.clip(1.0 - float(seen_width_m) / marker_width, 0.0, 1.0)
        var_r += (0.5 * marker_width * hidden) ** 2
    if spec.use_beam_quantisation:
        var_h += (r * lidar.azimuth_step_rad) ** 2 / (12.0 * columns)
        var_v += (r * lidar.elevation_step_rad) ** 2 / (12.0 * rings)
    if spec.use_target_extent:
        var_h += marker_width**2 / (12.0 * columns)
        var_v += marker_height**2 / (12.0 * rings)

    floor = spec.min_sigma_m**2
    info = np.outer(e_r, e_r) / max(var_r, floor) + np.outer(e_h, e_h) / max(var_h, floor)
    if spec.use_vertical:
        info = info + np.outer(e_v, e_v) / max(var_v, floor)
    return info


def beam_columns(points_sensor: np.ndarray, lidar: LidarSpec) -> int:
    """Distinct beam azimuths among a strip's returns.

    Every ring shares the scanner's azimuths, so returns come in columns. How many there are is
    what says whether the strip's position between two beams was observed at all: with one
    column it was not, and the fit has to assume where in the beam the strip sat.
    """
    pts = np.asarray(points_sensor, dtype=float)
    if pts.shape[0] == 0:
        return 0
    step = lidar.azimuth_step_rad
    if step <= 0:
        return 1
    az = np.arctan2(pts[:, 1], pts[:, 0])
    return int(np.unique(np.round(np.unwrap(np.sort(az)) / step).astype(int)).size)


_BIAS_CACHE: dict = {}
BIAS_GRID_M = 0.25
BIAS_GRID_RAD = float(np.deg2rad(2.5))
BIAS_SAMPLES = 1024
BIAS_MIN_MATCHES = 24


def strip_fit_bias(
    los: np.ndarray,
    normal: np.ndarray,
    distance: float,
    lidar: LidarSpec,
    marker_width: float,
    marker_thickness: float,
    marker_height: float = 1.0,
    dz: float = 0.5,
    n_columns: int | None = None,
) -> np.ndarray:
    """Expected horizontal error of ``fit_strip_centre``, fit minus true centre, for a strip
    whose facing is known. 2-vector in the frame ``los`` and ``normal`` are given in.

    ``los`` is the horizontal unit vector from the sensor to the strip, ``normal`` the strip's
    outward face normal, ``dz`` the height of the strip's centre above the sensor. The estimator
    knows all three: it mounted the strip.

    A strip has an end face as well as a front face. Seen along a tunnel wall the front is
    foreshortened to ``width * |los . normal|`` while the end toward the sensor presents
    ``thickness * |los . axis|``, which at 7 m beside a 3.2 m tunnel is 3.2 cm against 2.0 cm:
    four returns in ten land on the near end, 7.5 cm short of the centre. Measured on 300 to 400
    poses per range, the fit reads the strip 1.4 to 2.2 cm nearer than it is from 5.5 m out, a
    1 mm strip reads 0.0 cm, and down a 300 m chain of twenty strips the bias was worth 1.5 m,
    always short (``docs/failures.md`` number 31).

    Nothing here models the fit by hand. An analytic model of the returns was right from 4 m out
    and 0.5 to 0.8 cm wrong at 2 to 3 m, where the fit takes its other branch. So the fit itself
    is run on synthetic returns from the known box (front face and the end toward the sensor,
    the azimuth comb at a random phase, every ring that crosses the strip, the sensor's range
    noise) and the mean is cached on a grid of distance and incidence angle. The generator is
    seeded by the grid cell, so the same geometry always gives the same correction.

    ``n_columns``, the number of distinct beam azimuths the detection actually had, conditions
    that average on what was observed, and it is the difference between a correction that works
    and one that does not. Averaged over where the strip sits between two beams, the bias at
    4 to 5 m is under a centimetre. A robot that drops a strip beside itself and then steps
    0.5 m at a time does not sample that average: it meets the same handful of beam phases at
    every strip, and more than half its detections are a single column, worth +3.2 cm. Measured
    on the lattice those poses make, the average over phase misses +4.3 cm of it
    (``docs/failures.md`` number 31).
    """
    los = np.asarray(los, dtype=float)[:2]
    normal = np.asarray(normal, dtype=float)[:2]
    los = los / max(np.linalg.norm(los), 1e-12)
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    axis = np.array([-normal[1], normal[0]])
    # incidence angle, signed: 0 is face on, +90 degrees is looking along +axis
    ang = float(np.arctan2(los @ axis, -(los @ normal)))
    if marker_thickness <= 0.0 or abs(ang) >= 0.5 * np.pi:
        return np.zeros(2)
    along, out, _, _ = _strip_fit_moments(ang, distance, lidar, marker_width, marker_thickness,
                                          marker_height, dz, n_columns)
    return (along if ang >= 0 else -along) * axis + out * normal


def _strip_fit_moments(ang, distance, lidar, marker_width, marker_thickness, marker_height, dz,
                       n_columns):
    """Cached (mean along the width axis, mean along the normal, sd radial, sd tangential)."""
    columns = None if n_columns is None else min(int(n_columns), 4)
    key = (int(round(distance / BIAS_GRID_M)), int(round(abs(ang) / BIAS_GRID_RAD)), lidar,
           round(marker_width, 4), round(marker_thickness, 4), round(marker_height, 3), round(dz, 1),
           columns)
    if key not in _BIAS_CACHE:
        _BIAS_CACHE[key] = _simulate_fit_bias(key[0] * BIAS_GRID_M, key[1] * BIAS_GRID_RAD, lidar,
                                              marker_width, marker_thickness, marker_height, round(dz, 1),
                                              columns)
    return _BIAS_CACHE[key]


def strip_fit_sigma(
    los: np.ndarray,
    normal: np.ndarray,
    distance: float,
    lidar: LidarSpec,
    marker_width: float,
    marker_thickness: float,
    marker_height: float = 1.0,
    dz: float = 0.5,
    n_columns: int | None = None,
) -> tuple[float, float]:
    """Spread of the strip fit for this view, along the line of sight and across it, metres.

    The same Monte Carlo that gives ``strip_fit_bias`` its mean. Taking both from one measurement
    is the point: the mean was measured and the spread used to be a hand-written term,
    ``marker_width**2 / 12`` shared out over the returns, and the two disagreed by a factor of
    three and a half where the returns resolve the strip. Measured in six runs, a four-column
    detection at 2 m has a fit spread of 0.63 cm along the tunnel where that term charged 2.2 cm,
    while a single-column detection at 4.4 m has 2.61 cm of spread about a 3.17 cm offset, which
    the strip's own width over root twelve, 4.33 cm, happens to cover almost exactly.

    This is what an observation of a strip is worth, and it already contains the range noise, the
    beam comb and the strip's geometry, so it replaces them rather than adding to them.
    """
    los = np.asarray(los, dtype=float)[:2]
    normal = np.asarray(normal, dtype=float)[:2]
    n = float(np.linalg.norm(normal))
    if n < 1e-9 or distance < 1e-6 or marker_width <= 0.0:
        return 0.0, 0.0
    normal = normal / n
    axis = np.array([-normal[1], normal[0]])
    ang = float(np.arctan2(float(los @ axis), -float(los @ normal)))
    _, _, sd_r, sd_h = _strip_fit_moments(ang, distance, lidar, marker_width, marker_thickness,
                                          marker_height, dz, n_columns)
    return sd_r, sd_h


def _simulate_fit_bias(distance, ang, lidar, width, thickness, height, dz, n_columns=None):
    """Mean of (fit - centre) in the strip's frame: (along the width axis, along the normal),
    for a sensor ``distance`` away whose line of sight makes ``ang`` >= 0 with the face normal.

    With ``n_columns`` the mean is taken over the beam phases that would have produced a
    detection with that many columns, which is the information the detection carries. Phases
    that match are rare for some geometries, so the unconditioned mean is used when fewer than
    ``BIAS_MIN_MATCHES`` of them do."""
    if distance < 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    rng = np.random.default_rng([int(round(distance * 1000)), int(round(ang * 1e6)), 7])
    h, t = 0.5 * width, 0.5 * thickness
    # strip frame: x along the width axis, y the outward normal. Sensor on the -x, +y side.
    sensor = np.array([-distance * np.sin(ang), distance * np.cos(ang)])
    step = lidar.azimuth_step_rad
    centre_bearing = float(np.arctan2(-sensor[1], -sensor[0]))
    half_span = float(np.arctan2(np.hypot(h, t), distance)) + step
    el = (np.linspace(-0.5, 0.5, lidar.n_elevation) * np.deg2rad(lidar.fov_elevation_deg)
          if lidar.n_elevation > 1 else np.zeros(1))
    errs, every = [], []
    for _ in range(BIAS_SAMPLES):
        phase = rng.uniform(0.0, step)
        k0 = np.ceil((centre_bearing - half_span - phase) / step)
        az = phase + step * np.arange(k0, k0 + np.ceil(2 * half_span / step) + 1)
        d = np.stack([np.cos(az), np.sin(az)], axis=1)
        hits = []
        for di in d:
            best = np.inf
            if di[1] < -1e-12:  # the front face, y = +t
                r = (t - sensor[1]) / di[1]
                if r > 0 and abs(sensor[0] + r * di[0]) <= h:
                    best = r
            if di[0] > 1e-12:  # the end toward the sensor, x = -h
                r = (-h - sensor[0]) / di[0]
                if r > 0 and abs(sensor[1] + r * di[1]) <= t:
                    best = min(best, r)
            if np.isfinite(best):
                hits.append((di, best))
        pts = []
        for di, r in hits:
            z = r * np.tan(el)
            for e in el[(z >= dz - 0.5 * height) & (z <= dz + 0.5 * height)]:
                slant = r / np.cos(e)
                if not (lidar.min_range <= slant <= lidar.max_range):
                    continue
                slant = slant + rng.normal(0.0, lidar.range_sigma)
                pts.append([slant * np.cos(e) * di[0], slant * np.cos(e) * di[1], slant * np.sin(e)])
        if not pts:
            continue
        pts = np.array(pts)
        fit, _ = fit_strip_centre(pts, lidar, width, thickness)
        # the fit is relative to the sensor; the true centre is at -sensor from it
        err = fit[:2] + sensor
        every.append(err)
        if n_columns is not None and min(beam_columns(pts, lidar), 4) != n_columns:
            continue
        errs.append(err)
    if len(errs) < BIAS_MIN_MATCHES:
        errs = every  # this geometry hardly ever gives that many columns
    if not errs:
        return 0.0, 0.0, 0.0, 0.0
    e = np.array(errs)
    mean = e.mean(axis=0)
    # the spread, resolved the way the measurement model wants it: along the line of sight to the
    # strip's centre and across it
    e_r = -sensor / np.linalg.norm(sensor)
    e_h = np.array([-e_r[1], e_r[0]])
    sd_r = float((e @ e_r).std())
    sd_h = float((e @ e_h).std())
    return float(mean[0]), float(mean[1]), sd_r, sd_h


def fit_strip_centre(
    points_sensor: np.ndarray,
    lidar: LidarSpec,
    marker_width: float,
    marker_thickness: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Estimate the mounting point of a strip from the returns that landed on it.

    The centroid of the returns is not that point, and the difference is a bias
    rather than noise. Two geometric effects produce it, both of which the robot
    knows enough to undo:

    * At grazing incidence a strip of known width occludes its own far half, so the
      returns cluster on the near edge. The centroid then sits up to half a width
      short. Measured on a 0.15 m strip at 4.3 m, that is 3 cm, against a modelled
      standard deviation of 1.7 cm.
    * The returns land on the face, which stands proud of the wall by half the
      strip's thickness, while the point registered when the strip went on is the
      mounting point on the wall itself.

    So: project the returns onto the horizontal plane, find the strip's width axis
    from their spread, and if the observed extent plus one beam spacing covers the
    known width, take the midpoint of the extremes, whose quantisation errors
    cancel. If it does not, the far edge is not being seen, so anchor to the near
    edge and step half a width toward the far one. Then push the result back by
    half the thickness along the line of sight.

    Returns the fitted point and how much of the strip's width was actually seen,
    which is what tells the measurement model how foreshortened the view was.

    Nothing here is tuned. Both corrections come from the strip's known dimensions
    and the beam spacing, and the residual distributions before and after are in
    ``experiments/strip_fit.py``.
    """
    pts = np.asarray(points_sensor, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros(3), 0.0
    centroid = pts.mean(axis=0)
    p2 = pts[:, :2]
    c2 = centroid[:2]
    r = float(np.linalg.norm(c2))
    if r < 1e-6 or pts.shape[0] < 2:
        return centroid, 0.0
    u = c2 / r

    spread = p2 - c2
    # the strip's width axis: the principal direction of the horizontal spread,
    # falling back to the perpendicular of the line of sight when the returns are
    # too few or too collinear with it to say
    cov = spread.T @ spread
    w_eig, v_eig = np.linalg.eigh(cov)
    axis = v_eig[:, -1]
    if w_eig[-1] < 1e-8:
        axis = np.array([-u[1], u[0]])
    if axis @ np.array([-u[1], u[0]]) < 0:
        axis = -axis

    s = spread @ axis
    beam = max(r * lidar.azimuth_step_rad, 1e-9)
    extent = float(s.max() - s.min()) + beam
    if extent >= 0.8 * marker_width:
        offset = 0.5 * (float(s.max()) + float(s.min()))
    else:
        near = np.argmin(np.linalg.norm(p2, axis=1))
        s_near = float(s[near])
        toward_far = 1.0 if s_near <= float(np.median(s)) else -1.0
        # the near edge sits half a beam spacing outside the nearest return
        offset = s_near - toward_far * 0.5 * beam + toward_far * 0.5 * marker_width

    centre2 = c2 + axis * offset + u * (0.5 * marker_thickness)
    seen = float(min(extent, marker_width))
    return np.array([centre2[0], centre2[1], centroid[2]]), seen
