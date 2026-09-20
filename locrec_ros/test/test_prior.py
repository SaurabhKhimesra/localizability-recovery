"""The motion prior and the registration range, the two things the node got wrong.

Both run without a ROS graph or a ROS install.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locrec_ros.conversions import odometry_to_pose, stamp_to_ns  # noqa: E402
from locrec_ros.detector import DetectorConfig, LocalizabilityDetector  # noqa: E402
from locrec_ros.prior import PriorPairer  # noqa: E402

from locrec.lidar import LIMITED_FOV, SPINNING_360  # noqa: E402
from locrec.odometry import OdometryConfig  # noqa: E402
from locrec.runner import default_registration_range  # noqa: E402
from locrec.se3 import euler_zyx, inv_T, make_T  # noqa: E402
from locrec.sim import PlatformSpec  # noqa: E402

SEC = 1_000_000_000


def _pose(x, yaw=0.0):
    return make_T(euler_zyx(yaw), [x, 0.0, 0.0])


def _odometry(position, quaternion):
    """Duck-typed nav_msgs/Odometry, as the PointCloud2 tests do it."""
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
                orientation=SimpleNamespace(
                    x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
                ),
            )
        )
    )


# ---- the detector will not run without a prior -----------------------------

# these tests exercise pairing, ranges and message shapes, none of which depend
# on where the threshold sits, so any finite value will do
ANY_THRESHOLD = 2.0e-3
CAL = dict(ratio_threshold=ANY_THRESHOLD, marker_reliable_range_m=7.0, scheduler_margin_m=1.5)



def test_a_scan_without_a_prior_is_refused_not_assumed_still():
    """No prior used to mean no motion, and the estimate stopped following the robot."""
    det = LocalizabilityDetector(DetectorConfig(**CAL))
    points = np.zeros((100, 3))
    with pytest.raises(TypeError):
        det.process(points)
    for bad in (None, np.eye(3), np.full((4, 4), np.nan)):
        with pytest.raises(ValueError):
            det.process(points, bad)


def test_the_estimate_moves_by_the_increment_between_odometry_poses():
    """Until the map is confirmed the prior carries the pose on its own, so after two
    scans the estimate is exactly the increment between the two odometry poses,
    wherever the odometry frame started."""
    det = LocalizabilityDetector(DetectorConfig(**CAL))
    T0 = make_T(euler_zyx(0.7), [12.0, -3.0, 0.5])
    T1 = T0 @ make_T(euler_zyx(0.05), [0.5, 0.02, 0.0])
    points = np.random.default_rng(0).uniform(-5.0, 5.0, (2000, 3))
    assert det.process(points, T0) is None
    np.testing.assert_allclose(det.odometry.T, np.eye(4), atol=1e-12)
    assert det.process(points, T1) is None
    np.testing.assert_allclose(det.odometry.T, inv_T(T0) @ T1, atol=1e-12)


# ---- the registration range the threshold was calibrated with ---------------


@pytest.mark.parametrize(
    "lidar, params",
    [
        (SPINNING_360, dict(max_range=10.0, fov_azimuth_deg=360.0, azimuth_beams=360)),
        (LIMITED_FOV, dict(max_range=30.0, fov_azimuth_deg=90.0, azimuth_beams=180)),
    ],
    ids=["ugv", "drone"],
)
def test_registration_range_is_the_one_the_threshold_was_calibrated_with(lidar, params):
    det = LocalizabilityDetector(DetectorConfig(**CAL, **params))
    calibrated = default_registration_range(lidar, OdometryConfig(), PlatformSpec().step_length)
    assert det.odometry.cfg.registration_range == calibrated
    # for both study sensors that is the sampling limit, and it sits well inside the
    # fixed 0.85 of max range the detector used to register out to
    assert calibrated == lidar.nyquist_range(OdometryConfig().map_resolution)
    assert calibrated < 0.85 * lidar.max_range


@pytest.mark.parametrize("max_range, expected", [(10.0, 30.0), (30.0, 60.0)], ids=["ugv", "drone"])
def test_the_local_map_reaches_further_than_the_sensor(max_range, expected):
    """``run_pass`` floors the crop radius at twice the sensor range. The UGV's 30 m
    default already clears its 10 m sensor; the drone's 30 m sensor does not, and a
    map shorter than the sensor is `docs/failures.md` number 1."""
    det = LocalizabilityDetector(DetectorConfig(**CAL, max_range=max_range))
    assert det.odometry.cfg.map_radius == max(DetectorConfig(**CAL).map_radius, 2.0 * max_range)
    assert det.odometry.cfg.map_radius == expected


def test_fov_and_beam_count_both_change_the_range():
    """The fov parameter used to be accepted and never read."""
    base = LocalizabilityDetector(DetectorConfig(**CAL, azimuth_beams=360))
    more_beams = LocalizabilityDetector(DetectorConfig(**CAL, azimuth_beams=720))
    narrower = LocalizabilityDetector(DetectorConfig(**CAL, fov_azimuth_deg=180.0, azimuth_beams=360))
    r = [d.odometry.cfg.registration_range for d in (base, more_beams, narrower)]
    assert r[0] < r[1] and r[0] < r[2], r


# ---- odometry on the wire --------------------------------------------------


def test_odometry_pose_reads_position_and_orientation():
    h = np.sqrt(0.5)
    T = odometry_to_pose(_odometry((1.0, 2.0, 3.0), (0.0, 0.0, h, h)))  # 90 deg about z
    np.testing.assert_allclose(T[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])


def test_quaternion_is_normalised_and_a_zero_or_nan_pose_is_refused():
    h = np.sqrt(0.5)
    np.testing.assert_allclose(
        odometry_to_pose(_odometry((0, 0, 0), (0.0, 0.0, 3 * h, 3 * h))),
        odometry_to_pose(_odometry((0, 0, 0), (0.0, 0.0, h, h))),
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        odometry_to_pose(_odometry((0, 0, 0), (0.0, 0.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        odometry_to_pose(_odometry((float("nan"), 0, 0), (0.0, 0.0, 0.0, 1.0)))


def test_stamps_compare_exactly_in_nanoseconds():
    assert stamp_to_ns(SimpleNamespace(sec=1789470853, nanosec=168117399)) == 1789470853168117399


# ---- pairing a scan with the odometry at its own stamp ----------------------


def test_a_scan_published_before_its_odometry_waits_for_it():
    """The simulator's order: the scan goes out first, then the odometry with its stamp."""
    p = PriorPairer()
    assert p.add_scan(5 * SEC, "scan") == []
    assert p.waiting == 1
    ((stamp, payload, pose, _side),) = p.add_odometry(5 * SEC, _pose(2.0))
    assert (stamp, payload) == (5 * SEC, "scan")
    np.testing.assert_allclose(pose, _pose(2.0))


def test_a_scan_published_after_its_odometry_is_released_at_once():
    p = PriorPairer()
    assert p.add_odometry(5 * SEC, _pose(2.0)) == []
    ((_, _, pose, _side),) = p.add_scan(5 * SEC, "scan")
    np.testing.assert_allclose(pose, _pose(2.0))


def test_between_two_samples_the_pose_is_interpolated_not_the_latest():
    p = PriorPairer()
    p.add_odometry(4 * SEC, _pose(1.0, yaw=0.0))
    p.add_odometry(6 * SEC, _pose(3.0, yaw=0.4))
    ((_, _, pose, _side),) = p.add_scan(int(4.5 * SEC), "scan")
    np.testing.assert_allclose(pose, _pose(1.5, yaw=0.1), atol=1e-12)


def test_a_scan_newer_than_every_sample_waits_instead_of_extrapolating():
    p = PriorPairer()
    p.add_odometry(4 * SEC, _pose(1.0))
    assert p.add_scan(5 * SEC, "scan") == []
    ((_, _, pose, _side),) = p.add_odometry(6 * SEC, _pose(3.0))
    np.testing.assert_allclose(pose, _pose(2.0), atol=1e-12)


def test_a_scan_older_than_the_odometry_is_dropped_not_guessed():
    p = PriorPairer()
    p.add_odometry(6 * SEC, _pose(3.0))
    assert p.add_scan(5 * SEC, "scan") == []
    assert (p.dropped_stale, p.waiting) == (1, 0)


def test_scans_leave_in_stamp_order_and_the_wait_is_bounded():
    p = PriorPairer(max_pending=3)
    for k in range(5):
        assert p.add_scan((k + 1) * SEC, k) == []
    assert p.dropped_overflow == 2
    out = p.add_odometry(0, _pose(0.0)) + p.add_odometry(10 * SEC, _pose(10.0))
    assert [item.payload for item in out] == [2, 3, 4]
    np.testing.assert_allclose([item.pose[0, 3] for item in out], [3.0, 4.0, 5.0], atol=1e-12)


def test_a_scan_stamped_before_one_already_released_is_dropped():
    p = PriorPairer()
    p.add_odometry(0, _pose(0.0))
    p.add_odometry(10 * SEC, _pose(10.0))
    assert len(p.add_scan(5 * SEC, "a")) == 1
    assert p.add_scan(4 * SEC, "b") == []
    assert p.dropped_out_of_order == 1


def test_odometry_history_keeps_only_what_later_scans_can_need():
    p = PriorPairer(max_history=50)
    for k in range(1000):  # 10 s of odometry at 100 Hz
        p.add_odometry(k * 10_000_000, _pose(k / 100))
    assert len(p._odom_stamps) == 50  # no scan yet, so bounded by max_history alone
    ((_, _, pose, _side),) = p.add_scan(9_955_000_000, "scan")
    np.testing.assert_allclose(pose[0, 3], 9.955, atol=1e-9)
    # every later scan is stamped after 9.955 s, so 9.95 s is the oldest sample needed
    assert len(p._odom_stamps) == 5


def test_with_start_at_odometry_the_estimate_starts_at_the_first_odometry_pose():
    """What run_pass does: its estimate starts at the platform's first pose. The drone's gaze
    policies need it, because the track heading they steer by is in the odometry frame."""
    det = LocalizabilityDetector(DetectorConfig(**CAL, start_at_odometry=True))
    T0 = make_T(euler_zyx(0.7), [12.0, -3.0, 0.5])
    T1 = T0 @ make_T(euler_zyx(0.05), [0.5, 0.02, 0.0])
    points = np.random.default_rng(0).uniform(-5.0, 5.0, (2000, 3))
    assert det.process(points, T0) is None
    np.testing.assert_allclose(det.odometry.T, T0, atol=1e-12)
    assert det.process(points, T1) is None
    np.testing.assert_allclose(det.odometry.T, T1, atol=1e-12)


def test_a_gaze_policy_refuses_to_run_in_a_frame_other_than_the_odometry_one():
    with pytest.raises(ValueError, match="start_at_odometry"):
        LocalizabilityDetector(DetectorConfig(ratio_threshold=2.9e-3, platform="drone", gaze="glance",
                                              max_range=30.0, fov_azimuth_deg=90.0, azimuth_beams=180))
    det = LocalizabilityDetector(DetectorConfig(ratio_threshold=2.9e-3, platform="drone", gaze="glance",
                                                max_range=30.0, fov_azimuth_deg=90.0, azimuth_beams=180,
                                                elevation_beams=112, fov_elevation_deg=60.0,
                                                start_at_odometry=True))
    with pytest.raises(ValueError, match="track_yaw"):
        det.process(np.zeros((10, 3)), np.eye(4))
