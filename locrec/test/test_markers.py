"""Marker geometry, detection and the landmark measurement model."""
import numpy as np
import pytest

from locrec.landmarks import LandmarkBook, LandmarkSpec, measurement_information
from locrec.lidar import SPINNING_360
from locrec.odometry import (
    MotionPriorSpec,
    Odometry,
    OdometryConfig,
    _fuse,
    prior_information,
)
from locrec.policies import (
    DropOnFailure,
    LocalizabilityScheduler,
    NoMarkers,
    UniformSpacing,
    _blind_stretch_targets,
)
from locrec.se3 import make_T
from locrec.sim import TunnelSim
from locrec.worlds import WorldSpec, build_world

TUBE = WorldSpec(
    length=40.0,
    n_curves=0,
    n_junctions=0,
    n_niches=0,
    straight_lead_in=40.0,
    width_min=3.2,
    width_max=3.2,
    width_mean=3.2,
    n_marker_slots=8,
)


def _sim():
    return TunnelSim(0, TUBE, SPINNING_360)


# ---- geometry -------------------------------------------------------------


def test_strip_is_mounted_flush_on_the_wall():
    sim = _sim()
    T = make_T(np.eye(3), [10.0, 0.0, 0.7])
    slot, offset = sim.drop_marker_on_wall(T)
    assert slot == 0
    pos = sim.marker_positions()[0]
    half_width = sim.world.width[10] * 0.5
    assert abs(abs(pos[1]) - half_width) < 0.1, pos
    assert abs(pos[0] - 10.0) < 1e-6
    np.testing.assert_allclose(T[:3, :3] @ offset + T[:3, 3], pos, atol=1e-9)


def test_a_vertical_strip_is_seen_out_to_several_metres_and_not_beyond():
    """The detection range is a measured property of the sensor and the target, not
    an assumption. A 1 m strip viewed along a tunnel is nearly edge on, so the
    useful window is far shorter than the sensor's 10 m range, and the marker
    spacing a scheduler can get away with follows from that, not from the datasheet.
    """
    sim = _sim()
    sim.drop_marker_on_wall(make_T(np.eye(3), [20.0, 0.0, 0.7]))
    seen = {}
    for d in (1.0, 3.0, 5.0, 7.0, 12.0):
        scan = sim.scan(make_T(np.eye(3), [20.0 - d, 0.0, 0.7]))
        seen[d] = sum(x.n_beams for x in sim.marker_detections(scan))
    assert seen[1.0] > seen[5.0] > 0
    assert seen[12.0] == 0
    assert seen[3.0] > 0


def test_detections_are_grouped_per_marker():
    sim = _sim()
    sim.drop_marker_on_wall(make_T(np.eye(3), [12.0, 0.0, 0.7]))
    sim.drop_marker_on_wall(make_T(np.eye(3), [16.0, 0.0, 0.7]))
    dets = sim.marker_detections(sim.scan(make_T(np.eye(3), [14.0, 0.0, 0.7])))
    assert {d.slot for d in dets} == {0, 1}
    for d in dets:
        assert d.n_beams >= 1
        assert 1.0 < d.range_m < 4.0


def test_nothing_is_detected_before_anything_is_dropped():
    sim = _sim()
    assert sim.marker_detections(sim.scan(make_T(np.eye(3), [10.0, 0.0, 0.7]))) == []


# ---- measurement model ----------------------------------------------------


def test_measurement_is_sharpest_along_the_tunnel_axis_for_an_abeam_strip():
    """A strip abeam of the robot puts its horizontal tangential axis along the
    tunnel, which is precisely the direction the odometry cannot observe."""
    p = np.array([0.0, 4.0, 0.0])  # abeam, to the left, level with the sensor
    info = measurement_information(p, 5, SPINNING_360, LandmarkSpec(), 1.0, 0.15)
    sigma_along = 1.0 / np.sqrt(info[0, 0])  # x is along the tunnel here
    assert sigma_along < 0.05
    assert info[2, 2] == pytest.approx(0.0, abs=1e-9), "no vertical constraint"


def test_more_returns_mean_a_tighter_measurement():
    p = np.array([0.0, 4.0, 0.5])
    few = measurement_information(p, 1, SPINNING_360, LandmarkSpec(), 1.0, 0.15)
    many = measurement_information(p, 16, SPINNING_360, LandmarkSpec(), 1.0, 0.15)
    # rank two by design: the vertical is not constrained at all
    assert np.linalg.eigvalsh(few)[0] == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.eigvalsh(many)[2] > np.linalg.eigvalsh(few)[2]


def test_only_a_wider_spread_of_azimuths_tightens_the_along_tunnel_axis():
    """Returns stacked in one column are one look along the tunnel, not n of them.

    The rings in a column share an azimuth, so they average down the range noise and
    say nothing new about where the strip sits across the line of sight. Counting them
    there let a strip crossed by a single column claim several centimetres more
    precision than it had, and the far fixes then pulled far harder than the evidence
    allowed (docs/failures.md number 31).
    """
    p = np.array([0.0, 4.0, 0.5])  # abeam: the horizontal tangential axis is x, along the tunnel
    one_column = measurement_information(p, 1, SPINNING_360, LandmarkSpec(), 1.0, 0.15, n_columns=1)
    more_rings = measurement_information(p, 6, SPINNING_360, LandmarkSpec(), 1.0, 0.15, n_columns=1)
    three = measurement_information(p, 6, SPINNING_360, LandmarkSpec(), 1.0, 0.15, n_columns=3)

    assert more_rings[0, 0] == pytest.approx(one_column[0, 0], rel=1e-12), "rings say nothing along the tunnel"
    assert more_rings[1, 1] > one_column[1, 1], "but they do average the range down"
    assert three[0, 0] > one_column[0, 0], "more azimuths do tighten the along-tunnel axis"

    # and a single column is held to the strip's own width over root twelve
    sigma = 1.0 / np.sqrt(one_column[0, 0])
    assert sigma > 0.15 / np.sqrt(12.0), sigma


def test_measurement_information_is_symmetric_and_rank_two_by_default():
    info = measurement_information(
        np.array([2.0, 3.0, -0.4]), 3, SPINNING_360, LandmarkSpec(), 1.0, 0.15
    )
    np.testing.assert_allclose(info, info.T, atol=1e-9)
    ev = np.linalg.eigvalsh(info)
    assert ev[0] == pytest.approx(0.0, abs=1e-9), "the vertical must carry no weight"
    assert ev[1] > 0


def test_the_vertical_can_be_switched_back_on():
    p = np.array([0.0, 4.0, 0.5])
    with_v = measurement_information(
        p, 4, SPINNING_360, LandmarkSpec(use_vertical=True), 1.0, 0.15
    )
    assert np.linalg.eigvalsh(with_v).min() > 0


def test_the_strip_height_never_enters_the_horizontal_measurement():
    """Fitting the strip means its 1 m height is a detection aid, not a position."""
    p = np.array([0.0, 4.0, 0.5])
    short = measurement_information(p, 4, SPINNING_360, LandmarkSpec(), 0.3, 0.15)
    tall = measurement_information(p, 4, SPINNING_360, LandmarkSpec(), 3.0, 0.15)
    np.testing.assert_allclose(short, tall, atol=1e-9)


def test_range_grows_the_tangential_uncertainty():
    near = measurement_information(
        np.array([0.0, 2.0, 0.0]), 4, SPINNING_360, LandmarkSpec(), 1.0, 0.15
    )
    far = measurement_information(
        np.array([0.0, 9.0, 0.0]), 4, SPINNING_360, LandmarkSpec(), 1.0, 0.15
    )
    assert near[0, 0] > far[0, 0]


# ---- the landmark book and the correction ---------------------------------


def test_landmark_is_registered_at_the_estimated_pose_not_the_true_one():
    """A marker inherits exactly the error the estimate had when it went down."""
    book = LandmarkBook()
    odom = Odometry(make_T(np.eye(3), [5.0, 0.0, 0.7]), lidar_spec=SPINNING_360)
    odom.T[0, 3] += 0.4  # the estimate is 40 cm ahead of the truth
    odom.register_landmark(0, np.array([0.0, 1.5, 0.5]))
    lm = odom.landmarks.get(0)
    assert lm is not None
    np.testing.assert_allclose(lm.position, [5.4, 1.5, 1.2], atol=1e-9)
    assert len(book) == 0


def _odom_with_marker(lag: float = 0.5):
    """Estimator holding one registered marker, with the pose lagging by ``lag``."""
    sim = _sim()
    T0 = make_T(np.eye(3), [10.0, 0.0, 0.7])
    odom = Odometry(T0.copy(), OdometryConfig(), lidar_spec=SPINNING_360)
    slot, offset = sim.drop_marker_on_wall(T0)
    odom.register_landmark(slot, offset)

    T_true = make_T(np.eye(3), [10.0 + lag * 2, 0.0, 0.7])
    dets = sim.marker_detections(sim.scan(T_true))
    assert dets, "the marker must still be visible"
    odom.T = make_T(np.eye(3), [T_true[0, 3] - lag, 0.0, 0.7])
    # A metre of travel has accumulated since the drop. What weights the fix is the
    # odometry since that drop, not the absolute covariance, so the accumulator is
    # what has to grow; the absolute number is set too because it is still reported.
    odom._Q_cum[3:, 3:] += np.diag([0.04, 0.04, 0.04])
    odom.P = np.diag([1e-6, 1e-6, 1e-6, 0.04, 0.04, 0.04])
    return sim, odom, dets, T_true


def test_an_absolute_fix_pulls_the_estimate_back_toward_the_marker():
    sim, odom, dets, T_true = _odom_with_marker()
    before = odom.T[0, 3]
    used, _ = odom._absolute_fix(dets)
    assert used == 1
    after = odom.T[0, 3]
    assert after > before, "the fix must move the estimate forward"
    assert abs(after - T_true[0, 3]) < 0.5 * abs(before - T_true[0, 3])


def test_the_fix_moves_the_local_map_with_the_pose():
    """Without this the next registration drags the pose straight back."""
    sim, odom, dets, _ = _odom_with_marker()
    odom.map.add(np.array([[9.0, 1.5, 1.0], [11.0, -1.5, 1.0]]), centre=odom.T[:3, 3])
    odom.map.add(np.array([[9.0, 1.5, 1.0], [11.0, -1.5, 1.0]]), centre=odom.T[:3, 3])
    before_map = odom.map.points.copy()
    before_pose = odom.T[:3, 3].copy()

    odom._absolute_fix(dets)

    shift = odom.T[:3, 3] - before_pose
    assert np.linalg.norm(shift) > 1e-3
    after_map = odom.map.points
    assert after_map.shape == before_map.shape
    moved = np.sort(after_map, axis=0) - np.sort(before_map, axis=0)
    np.testing.assert_allclose(moved, np.tile(shift, (moved.shape[0], 1)), atol=0.25)


def test_the_fix_moves_recent_landmarks_but_never_its_own_anchor():
    """The marker the fix was computed from defines the correction. Moving it too
    would make the whole thing circular and leave a marker that tracks the drift."""
    sim, odom, dets, _ = _odom_with_marker()
    odom.landmarks.register(5, np.array([1.0, 2.0, 3.0]), step=0)  # dropped since
    odom.landmarks.register(6, np.array([4.0, 5.0, 6.0]), step=0)  # an older anchor
    odom._since_fix = [0, 5]
    anchor_before = odom.landmarks.get(0).position.copy()
    recent_before = odom.landmarks.get(5).position.copy()
    old_before = odom.landmarks.get(6).position.copy()

    odom._absolute_fix(dets)

    np.testing.assert_allclose(odom.landmarks.get(0).position, anchor_before, atol=1e-12)
    np.testing.assert_allclose(odom.landmarks.get(6).position, old_before, atol=1e-12)
    assert np.linalg.norm(odom.landmarks.get(5).position - recent_before) > 1e-3
    assert odom._since_fix == [0]


def test_the_fix_never_touches_height():
    sim, odom, dets, _ = _odom_with_marker()
    z_before = odom.T[2, 3]
    odom._absolute_fix(dets)
    assert odom.T[2, 3] == pytest.approx(z_before, abs=1e-12)


def test_the_covariance_shrinks_on_a_fix_and_grows_between_them():
    sim, odom, dets, _ = _odom_with_marker()
    before = np.trace(odom.P[3:, 3:])
    odom._absolute_fix(dets)
    after = np.trace(odom.P[3:, 3:])
    assert after < before
    odom._propagate_covariance(
        prior_information(odom.prior_spec, 0.5, make_T(np.eye(3), [0.5, 0.0, 0.0])),
        np.array([0.5, 0.0, 0.0]),
    )
    assert np.trace(odom.P[3:, 3:]) > after


def test_a_fix_with_a_tiny_covariance_barely_moves_anything():
    """Right after a fix the estimator should not be re-corrected by the same view."""
    sim, odom, dets, _ = _odom_with_marker()
    odom._Q_cum = np.zeros((6, 6))
    odom.landmarks.get(0).q_ref = np.zeros((6, 6))
    odom.landmarks.get(0).rel = np.zeros((3, 3))
    odom.P = np.diag([1e-10, 1e-10, 1e-10, 1e-8, 1e-8, 1e-8])
    before = odom.T[:3, 3].copy()
    odom._absolute_fix(dets)
    assert np.linalg.norm(odom.T[:3, 3] - before) < 1e-3


def test_the_absolute_guard_zeros_degenerate_registration_information():
    H = np.diag([1e8, 1e8, 1e8, 1.0, 1e6, 1e6])  # weak in x translation
    prior = prior_information(
        MotionPriorSpec(), dt=0.5, delta_T_prior=make_T(np.eye(3), [0.5, 0.0, 0.0])
    )
    T_corr = make_T(np.eye(3), [0.3, 0.0, 0.0])
    guarded, _ = _fuse(
        T_corr, H, prior, sensor_position=np.zeros(3), n_inliers=1000,
        residual=994.0, calibrate=False, gicp_ratio_floor=1e-3,
    )
    unguarded, _ = _fuse(
        T_corr, H, prior, sensor_position=np.zeros(3), n_inliers=1000,
        residual=994.0, calibrate=False, gicp_ratio_floor=None,
    )
    assert abs(guarded[0, 3]) < abs(unguarded[0, 3])
    assert abs(guarded[0, 3]) < 1e-6


def test_an_unregistered_marker_is_ignored():
    sim = _sim()
    T0 = make_T(np.eye(3), [10.0, 0.0, 0.7])
    odom = Odometry(T0.copy(), OdometryConfig(), lidar_spec=SPINNING_360)
    sim.drop_marker_on_wall(T0)  # placed, but never registered with the estimator
    dets = sim.marker_detections(sim.scan(make_T(np.eye(3), [11.0, 0.0, 0.7])))
    _, _, n = odom._landmark_system(T0, dets)
    assert n == 0


# ---- policies -------------------------------------------------------------


class _Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def marker_in_view(self):
        return len(self.detections) > 0


class _Loc:
    def __init__(self, ratio):
        self.ratio = ratio


class _Reg:
    def __init__(self, converged=True, num_inliers=1000, registered=True):
        self.converged = converged
        self.num_inliers = num_inliers
        self.registered = registered


def _ctx(distance, ratio=1e-3, detections=(), reg=None):
    return _Ctx(
        estimated_distance=distance,
        localizability=_Loc(ratio),
        detections=list(detections),
        registration=reg or _Reg(),
    )


def test_no_markers_never_drops():
    p = NoMarkers()
    assert p(_ctx(0.0)) == {}
    assert p(_ctx(100.0)) == {}


def test_uniform_spacing_drops_on_estimated_distance():
    p = UniformSpacing(spacing=10.0)
    drops = [d for d in np.arange(0.0, 35.0, 0.5) if p(_ctx(d)).get("drop_marker")]
    np.testing.assert_allclose(drops, [0.0, 10.0, 20.0, 30.0])


def _sched(**kw):
    return LocalizabilityScheduler(
        ratio_threshold=0.01, reliable_range_m=7.0, margin_m=1.5, **kw
    )


def test_scheduler_spaces_by_range_not_by_every_dip():
    """In a stretch that is degenerate throughout, the ratio is below threshold at
    every step. Spacing must come from the sensor, not from the trigger rate."""
    p = _sched()
    drops = [
        d
        for d in np.arange(0.0, 20.0, 0.5)
        if p(_ctx(d, ratio=1e-3, detections=[object()])).get("drop_marker")
    ]
    # twice the 7 m detection range, less the 1.5 m margin
    np.testing.assert_allclose(drops, [0.0, 12.5])


def test_scheduler_drops_on_the_falling_edge():
    """The valuable place to leave a marker is the last place you still know where
    you are, which is one step into the blind stretch."""
    p = _sched()
    for d in np.arange(0.0, 5.0, 0.5):  # structured: nothing spent
        assert not p(_ctx(d, ratio=0.5)).get("drop_marker")
    assert p(_ctx(5.0, ratio=1e-3)).get("drop_marker"), "must fire on the falling edge"
    assert not p(_ctx(5.5, ratio=1e-3, detections=[object()])).get("drop_marker")


def test_scheduler_fires_once_per_falling_edge_not_once_per_degenerate_step():
    p = _sched()
    n = 0
    for d in np.arange(0.0, 40.0, 0.5):
        ratio = 0.5 if (d // 10) % 2 == 0 else 1e-3  # alternating 10 m stretches
        if p(_ctx(d, ratio=ratio, detections=[object()])).get("drop_marker"):
            n += 1
    # two blind stretches, each an edge drop plus at most one range drop
    assert 2 <= n <= 4, n


def test_scheduler_spends_nothing_where_the_geometry_carries_the_estimate():
    p = _sched()
    assert not any(
        p(_ctx(d, ratio=0.5)).get("drop_marker") for d in np.arange(0.0, 30.0, 0.5)
    )


def test_scheduler_keeps_its_chain_across_a_structured_stretch():
    """The detector does not enter a blind stretch once. On a 300 m blind run the
    ratio crosses the threshold 77 times, so a policy that forgets its chain
    whenever the ratio recovers drops on all 77 of those edges and spends 64
    markers where the spacing rule wanted 24. The chain therefore survives."""
    p = _sched()
    assert p(_ctx(0.0, ratio=1e-3, detections=[object()])).get("drop_marker")
    for d in np.arange(0.5, 10.0, 0.5):  # a structured stretch, nothing spent
        assert not p(_ctx(d, ratio=0.5)).get("drop_marker")
    # blind again, but the last anchor is only 10 m back and reach is 12.5 m
    assert not p(_ctx(10.0, ratio=1e-3, detections=[object()])).get("drop_marker")
    assert p(_ctx(13.0, ratio=1e-3, detections=[object()])).get("drop_marker")


def test_scheduler_does_not_chase_an_intermittent_detection():
    """Detection is intermittent by design: the reliable range was calibrated at a
    0.4 per-scan rate, because the estimator needs a fix every 1.25 m, not every
    scan. A gap of a few steps is normal and must not trigger a drop."""
    p = _sched()
    assert p(_ctx(0.0, ratio=1e-3, detections=[object()])).get("drop_marker")
    for d in (0.5, 1.0, 1.5, 2.0):
        assert not p(_ctx(d, ratio=1e-3, detections=[])).get("drop_marker")


def test_drop_on_failure_ignores_a_healthy_but_degenerate_registration():
    """The point of the contrast: a degenerate tunnel produces a converged,
    inlier-rich registration, so a symptom watcher sees nothing wrong."""
    p = DropOnFailure(min_inliers=100)
    assert not p(_ctx(0.0, ratio=1e-6, reg=_Reg(converged=True, num_inliers=1000))).get(
        "drop_marker"
    )
    assert p(_ctx(10.0, reg=_Reg(converged=False))).get("drop_marker")


def test_oracle_targets_land_inside_the_blind_stretches():
    world = build_world(0, WorldSpec(length=120.0, n_junctions=1, n_niches=2))
    targets = _blind_stretch_targets(world, 6)
    assert 0 < len(targets) <= 6
    blind = ~world.feature_mask & (np.abs(world.curvature) <= 1e-6)
    for t in targets:
        i = int(np.clip(round(t / world.spec.ds), 0, len(blind) - 1))
        assert blind[i], t


def test_oracle_with_no_budget_places_nothing():
    world = build_world(0, WorldSpec(length=60.0))
    assert _blind_stretch_targets(world, 0) == []


def test_marker_capacity_is_respected_by_the_runner(tmp_path):
    from locrec import RunConfig, run_pass

    spec = dataclasses_replace(TUBE, n_marker_slots=3)
    r = run_pass(
        RunConfig(seed=0, platform="ugv", world=spec), UniformSpacing(spacing=2.0)
    )
    assert r.markers_placed == 3


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


@pytest.mark.parametrize("spacing", [5.0, 12.0])
def test_uniform_spacing_scales_the_marker_count(spacing):
    from locrec import RunConfig, run_pass

    r = run_pass(
        RunConfig(seed=0, platform="ugv", world=TUBE), UniformSpacing(spacing=spacing)
    )
    assert r.markers_placed == pytest.approx(TUBE.length / spacing, abs=2)


# ---- the strip fit --------------------------------------------------------


def _strip_returns(centre_x, centre_y, n, width=0.15, seen=1.0, rng=None):
    """Returns along the visible fraction of a strip lying along the x axis."""
    rng = rng or np.random.default_rng(0)
    s = np.linspace(-width / 2, -width / 2 + width * seen, n)
    return np.stack(
        [centre_x + s, np.full(n, centre_y), rng.uniform(0.8, 1.4, n)], axis=1
    )


def test_the_fit_finds_the_centre_of_a_fully_seen_strip():
    from locrec.landmarks import fit_strip_centre

    pts = _strip_returns(0.0, 1.5, 9)
    got, seen = fit_strip_centre(pts, SPINNING_360, 0.15, 0.02)
    assert got[0] == pytest.approx(0.0, abs=0.01)
    assert got[1] == pytest.approx(1.51, abs=0.005)  # half the thickness, pushed back
    assert seen == pytest.approx(0.15, abs=0.02)


def test_the_fit_uses_the_known_width_when_the_far_half_is_hidden():
    """The failure this exists for. At grazing incidence a strip occludes its own
    far half, so the returns pile up on the near edge and the centroid sits short
    of the centre by up to half a width."""
    from locrec.landmarks import fit_strip_centre

    # strip ahead and to the left, width along x, only its near third returning
    pts = np.stack(
        [np.linspace(3.925, 3.975, 4), np.full(4, 1.5), np.linspace(0.9, 1.3, 4)], axis=1
    )
    centroid = pts.mean(axis=0)
    got, seen = fit_strip_centre(pts, SPINNING_360, 0.15, 0.0)
    assert seen < 0.15
    assert got[0] > centroid[0], (got[0], centroid[0])
    assert abs(got[0] - 4.0) < abs(centroid[0] - 4.0)


def test_the_fit_survives_a_single_return():
    from locrec.landmarks import fit_strip_centre

    pts = _strip_returns(0.0, 1.5, 1)
    got, seen = fit_strip_centre(pts, SPINNING_360, 0.15, 0.02)
    assert np.all(np.isfinite(got))
    assert seen == 0.0


def test_foreshortening_is_charged_to_the_range_axis():
    """A strip returning a fraction of its width is being seen end on, and what
    comes back is its near edge rather than its centre."""
    p = np.array([0.0, 4.0, 0.3])
    square = measurement_information(
        p, 8, SPINNING_360, LandmarkSpec(), 1.0, 0.15, seen_width_m=0.15
    )
    grazing = measurement_information(
        p, 8, SPINNING_360, LandmarkSpec(), 1.0, 0.15, seen_width_m=0.03
    )
    var_square = float(np.linalg.pinv(square)[1, 1])
    var_grazing = float(np.linalg.pinv(grazing)[1, 1])
    assert var_grazing > 4.0 * var_square, (var_square, var_grazing)
