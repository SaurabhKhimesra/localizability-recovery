"""Odometry tests on motions whose answer is known exactly."""
import numpy as np
import pytest

from locrec.lidar import LidarSpec
from locrec.odometry import (
    LocalMap,
    MotionPrior,
    MotionPriorSpec,
    Odometry,
    OdometryConfig,
    prior_information,
    voxel_downsample,
)
from locrec.se3 import euler_zyx, inv_T, make_T, pose_error, so3_log
from locrec.sim import TunnelSim, UGV
from locrec.worlds import WorldSpec

STRAIGHT = WorldSpec(
    length=40.0,
    n_curves=0,
    n_junctions=0,
    n_niches=2,
    straight_lead_in=6.0,
    n_marker_slots=4,
)
TEST_LIDAR = LidarSpec(
    n_azimuth=360,
    n_elevation=16,
    fov_azimuth_deg=360.0,
    fov_elevation_deg=30.0,
    max_range=10.0,
    range_sigma=0.0,
)
CFG = OdometryConfig(registration_range=6.0)
# The focused registration tests below prime the map with a single scan, so they
# opt out of the two-observation confirmation rule; it has its own test.
CFG_1OBS = OdometryConfig(registration_range=6.0, map_min_observations=1)


# ---- building blocks ------------------------------------------------------


def test_voxel_downsample_collapses_a_voxel_to_its_centroid():
    pts = np.array([[0.01, 0.01, 0.01], [0.09, 0.09, 0.09], [5.0, 5.0, 5.0]])
    out = voxel_downsample(pts, 0.2)
    assert out.shape == (2, 3)
    assert np.isclose(out, [0.05, 0.05, 0.05]).all(axis=1).any()


def test_voxel_downsample_handles_empty():
    assert voxel_downsample(np.zeros((0, 3)), 0.2).shape == (0, 3)


def test_local_map_crops_to_its_radius():
    m = LocalMap(radius=5.0, resolution=0.5, min_observations=1)
    pts = np.stack([np.arange(0.0, 20.0, 0.5), np.zeros(40), np.zeros(40)], axis=1)
    m.add(pts, centre=np.zeros(3))
    assert len(m) > 0
    assert np.linalg.norm(m.points, axis=1).max() <= 5.0 + 1e-9


def test_local_map_accumulates_across_adds():
    m = LocalMap(radius=100.0, resolution=0.5, min_observations=1)
    m.add(np.array([[0.0, 0.0, 0.0]]))
    m.add(np.array([[10.0, 0.0, 0.0]]))
    assert len(m) == 2


def test_local_map_withholds_voxels_until_they_are_confirmed():
    m = LocalMap(radius=100.0, resolution=0.5, min_observations=2)
    m.add(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    assert m.n_voxels == 2
    assert len(m) == 0, "a voxel seen once must not be offered for registration"
    m.add(np.array([[0.02, 0.0, 0.0]]))
    assert len(m) == 1
    np.testing.assert_allclose(m.points[0], [0.01, 0.0, 0.0], atol=1e-9)


def test_local_map_counts_scans_not_points():
    m = LocalMap(radius=100.0, resolution=0.5, min_observations=2)
    m.add(np.random.default_rng(0).uniform(0.0, 0.4, (500, 3)))
    assert len(m) == 0, "500 points from one scan are still one observation"


# ---- motion prior ---------------------------------------------------------


def test_motion_prior_is_unbiased_over_many_draws():
    spec = MotionPriorSpec()
    true = make_T(np.eye(3), [0.5, 0.0, 0.0])
    means = []
    for seed in range(400):
        means.append(MotionPrior(spec, seed=seed, dt=0.5).predict(true)[0, 3])
    # sigma of the scale error is 2 percent of 0.5 m, so the mean of 400 draws
    # sits well inside 4 sigma / sqrt(400)
    assert abs(np.mean(means) - 0.5) < 4.0 * (0.02 * 0.5) / np.sqrt(400)


def test_motion_prior_scale_error_is_constant_within_a_run():
    prior = MotionPrior(MotionPriorSpec(odom_noise_m_per_sqrt_m=0.0), seed=3, dt=0.5)
    true = make_T(np.eye(3), [0.5, 0.0, 0.0])
    a = prior.predict(true)[0, 3]
    b = prior.predict(true)[0, 3]
    assert a == pytest.approx(b)
    assert a != pytest.approx(0.5)


def test_motion_prior_is_deterministic_in_seed():
    true = make_T(euler_zyx(0.01), [0.5, 0.0, 0.0])
    a = MotionPrior(seed=11, dt=0.5).predict(true)
    b = MotionPrior(seed=11, dt=0.5).predict(true)
    np.testing.assert_allclose(a, b)


def test_prior_information_is_weakest_along_the_step_direction():
    """The scale error only acts along travel, so that axis must carry the least
    information and the two lateral axes must be equal to each other."""
    spec = MotionPriorSpec()
    step = make_T(np.eye(3), [0.5, 0.0, 0.0])
    info = prior_information(spec, dt=0.5, delta_T_prior=step)
    assert info.shape == (6, 6)
    ev, evec = np.linalg.eigh(info[3:, 3:])
    assert abs(evec[0, 0]) > 0.999, "the weak eigenvector must be the step direction"
    assert ev[1] == pytest.approx(ev[2], rel=1e-9), "lateral axes must match"
    assert ev[0] < 0.5 * ev[2], "the along-track axis must be the least certain"


def test_prior_information_rotates_into_the_given_frame():
    spec = MotionPriorSpec()
    step = make_T(np.eye(3), [0.5, 0.0, 0.0])
    R = euler_zyx(np.deg2rad(40.0))
    body = prior_information(spec, dt=0.5, delta_T_prior=step)
    world = prior_information(spec, dt=0.5, delta_T_prior=step, R_world_body=R)
    np.testing.assert_allclose(world[3:, 3:], R @ body[3:, 3:] @ R.T, atol=1e-9)
    np.testing.assert_allclose(np.linalg.eigvalsh(world[3:, 3:]), np.linalg.eigvalsh(body[3:, 3:]))


def test_prior_information_tracks_the_actual_step_not_a_nominal_one():
    """A policy that halves the speed must get a tighter prior, per step."""
    spec = MotionPriorSpec()
    short = prior_information(spec, dt=0.5, delta_T_prior=make_T(np.eye(3), [0.25, 0.0, 0.0]))
    long = prior_information(spec, dt=0.5, delta_T_prior=make_T(np.eye(3), [1.0, 0.0, 0.0]))
    assert short[3, 3] > long[3, 3]


def test_prior_information_handles_a_zero_step():
    """A stationary step must not claim infinite certainty and veto registration."""
    spec = MotionPriorSpec()
    info = prior_information(spec, dt=0.5, delta_T_prior=np.eye(4))
    assert np.all(np.isfinite(info))
    assert info[3, 3] == pytest.approx(1.0 / spec.odom_min_sigma_m**2)


# ---- registration on known motions ---------------------------------------


def _sim_and_platform():
    sim = TunnelSim(1, STRAIGHT, TEST_LIDAR)
    return sim, UGV(sim.world)


def test_registration_of_a_known_translation_is_exact_off_the_tunnel_axis():
    """Feed the true motion as the prior and look at where the residual lands.

    In a straight tunnel the observed axes (across and up) must come back to
    millimetres, while the axis along the tunnel is unobserved and is free to
    wander by centimetres. Asserting a single scalar bound would hide that split,
    which is the whole phenomenon this project is about.
    """
    sim, plat = _sim_and_platform()
    plat.s = 10.0
    T0 = plat.pose()
    odom = Odometry(T0, CFG_1OBS)
    odom.step(sim.scan(T0).points, np.eye(4))

    plat.s = 12.0
    T1 = plat.pose()
    out = odom.step(sim.scan(T1).points, inv_T(T0) @ T1)
    assert out.registered

    residual_body = (inv_T(T1) @ out.T)[:3, 3]
    assert abs(residual_body[1]) < 0.005, residual_body
    assert abs(residual_body[2]) < 0.005, residual_body
    assert abs(residual_body[0]) < 0.25, residual_body
    assert pose_error(out.T, T1)[1] < np.deg2rad(1.0)


def test_registration_corrects_a_deliberately_wrong_prior():
    """The prior is off by 25 cm across the tunnel. The walls observe that axis,
    so the correction must remove most of the error."""
    sim, plat = _sim_and_platform()
    plat.s = 10.0
    T0 = plat.pose()
    odom = Odometry(T0, CFG_1OBS)
    odom.step(sim.scan(T0).points, np.eye(4))

    plat.s = 11.0
    T1 = plat.pose()
    delta_true = inv_T(T0) @ T1
    bad = delta_true.copy()
    bad[:3, 3] = bad[:3, 3] + np.array([0.0, 0.25, 0.0])

    out = odom.step(sim.scan(T1).points, bad)
    lateral_before = 0.25
    lateral_after = abs(float((inv_T(T1) @ out.T)[1, 3]))
    assert lateral_after < 0.4 * lateral_before, (lateral_before, lateral_after)


def test_registration_recovers_a_known_rotation():
    sim, plat = _sim_and_platform()
    plat.s = 10.0
    T0 = plat.pose()
    odom = Odometry(T0, CFG_1OBS)
    odom.step(sim.scan(T0).points, np.eye(4))

    plat.s = 10.5
    T1 = make_T(plat.pose()[:3, :3] @ euler_zyx(np.deg2rad(4.0)), plat.pose()[:3, 3])
    bad = inv_T(T0) @ T1
    bad[:3, :3] = (inv_T(T0) @ plat.pose())[:3, :3]  # prior misses the 4 deg turn
    out = odom.step(sim.scan(T1).points, bad)
    err = np.linalg.norm(so3_log((inv_T(T1) @ out.T)[:3, :3]))
    assert err < np.deg2rad(1.0), np.rad2deg(err)


def test_the_prior_carries_the_pose_when_the_scan_is_empty():
    sim, plat = _sim_and_platform()
    T0 = plat.pose()
    odom = Odometry(T0, CFG_1OBS)
    delta = make_T(np.eye(3), [0.5, 0.0, 0.0])
    out = odom.step(np.zeros((0, 3)), delta)
    assert not out.registered
    np.testing.assert_allclose(out.T, T0 @ delta)


def test_prior_fusion_damps_the_correction_on_an_unobserved_axis():
    """Along a direction the Hessian calls unobserved, the fused correction must be
    a small fraction of the raw one; along an observed direction it must survive."""
    from locrec.odometry import _fuse

    H = np.diag([1e8, 1e8, 1e8, 1e2, 1e8, 1e8])  # weak in x translation
    prior = prior_information(
        MotionPriorSpec(), dt=0.5, delta_T_prior=make_T(np.eye(3), [0.0, 0.5, 0.0])
    )
    T_corr = make_T(np.eye(3), [0.3, 0.3, 0.0])
    fused, _ = _fuse(
        T_corr,
        H,
        prior,
        sensor_position=np.zeros(3),
        n_inliers=1000,
        residual=994.0,
        calibrate=False,
    )
    assert abs(fused[0, 3]) < 0.1 * 0.3
    assert abs(fused[1, 3]) > 0.9 * 0.3


def test_odometry_tracks_a_short_drive_with_a_noisy_prior():
    sim, plat = _sim_and_platform()
    prior = MotionPrior(MotionPriorSpec(), seed=2, dt=0.5)
    T = plat.pose()
    odom = Odometry(T.copy(), CFG, prior_spec=MotionPriorSpec(), dt=0.5)
    odom.step(sim.scan(T).points, np.eye(4))
    while not plat.finished:
        T_prev = T
        plat.advance()
        T = plat.pose()
        odom.step(sim.scan(T).points, prior.predict(inv_T(T_prev) @ T))
    te, _ = pose_error(odom.T, T)
    assert te < 0.05 * plat.path_length, f"{te:.2f} m over {plat.path_length:.0f} m"


def test_detector_fires_along_the_tunnel_axis_during_a_drive():
    """The weak direction the detector reports must line up with the tunnel, and the
    ratio must be far below what the same pipeline reports in a niche."""
    sim, plat = _sim_and_platform()
    T = plat.pose()
    odom = Odometry(T.copy(), CFG)
    odom.step(sim.scan(T).points, np.eye(4))

    aligned = []
    while not plat.finished:
        T_prev = T
        plat.advance()
        T = plat.pose()
        out = odom.step(sim.scan(T).points, inv_T(T_prev) @ T)
        if out.localizability is None:
            continue
        axis = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
        aligned.append(abs(float(out.localizability.weak_direction @ axis)))
    assert len(aligned) > 30
    assert np.median(aligned) > 0.9, np.median(aligned)


def test_detector_ranks_a_featureless_tunnel_below_a_featured_one():
    """Same estimator, same seed, two worlds that differ only in their features."""
    blind = WorldSpec(
        length=40.0,
        n_curves=0,
        n_junctions=0,
        n_niches=0,
        straight_lead_in=40.0,
        width_min=3.2,
        width_max=3.2,
        width_mean=3.2,
        n_marker_slots=2,
    )
    ratios = {}
    for name, ws in (("blind", blind), ("featured", STRAIGHT)):
        sim = TunnelSim(1, ws, TEST_LIDAR)
        plat = UGV(sim.world)
        plat.s = 12.0
        T = plat.pose()
        odom = Odometry(T.copy(), CFG)
        odom.step(sim.scan(T).points, np.eye(4))
        vals = []
        for _ in range(30):
            T_prev = T
            plat.advance()
            T = plat.pose()
            out = odom.step(sim.scan(T).points, inv_T(T_prev) @ T)
            if out.localizability is not None:
                vals.append(out.localizability.ratio)
        ratios[name] = float(np.median(vals))
    assert ratios["blind"] < ratios["featured"], ratios
