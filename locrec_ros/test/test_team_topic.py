"""The team half over ROS: what a vehicle sends about a strip, and what a teammate does with it.

Runs without a ROS graph. The message round trip needs the generated interfaces, so the two tests
that build a ``SharedLandmark`` skip where they are not installed; everything else is the
detector's own behaviour and always runs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locrec_ros.detector import DetectorConfig, LocalizabilityDetector  # noqa: E402

from locrec.sim import MarkerDetection  # noqa: E402

CAL = dict(ratio_threshold=2.0e-3, marker_reliable_range_m=7.0, scheduler_margin_m=1.5)
POSITION = np.array([5.0, 1.53, 1.2])
NORMAL = np.array([0.0, -1.0])
R_DROP = np.eye(2) * 1e-4


def follower(**kw):
    cfg = dict(platform="drone", gaze="across", use_shared_landmarks=True, **CAL)
    cfg.pop("marker_reliable_range_m"), cfg.pop("scheduler_margin_m")
    return LocalizabilityDetector(DetectorConfig(**cfg, **kw))


def test_a_shared_strip_is_registered_as_foreign():
    d = follower()
    assert d.add_shared_landmark(0, POSITION, R_DROP, NORMAL)
    lm = d.odometry.landmarks.get(0)
    assert lm.foreign, "a teammate's strip has to be marked, or it would be re-surveyed"
    np.testing.assert_allclose(lm.position, POSITION)
    np.testing.assert_allclose(lm.normal, NORMAL)
    assert lm.T_drop is None and lm.offset_drop is None, "there is no drop of this vehicle's"


def test_a_republished_strip_moves_but_is_not_registered_again():
    """A teammate republishes a strip when its own estimate of it moves, so the later position
    has to win. What must not happen is a second registration: that would reset the anchor's
    relative record and hand this vehicle a fix it has not earned."""
    d = follower()
    assert d.add_shared_landmark(3, POSITION, R_DROP, NORMAL)
    lm = d.odometry.landmarks.get(3)
    q_ref, rel, normal = lm.q_ref.copy(), lm.rel.copy(), lm.normal.copy()

    assert not d.add_shared_landmark(3, POSITION + 0.2, R_DROP * 4.0, NORMAL)
    lm = d.odometry.landmarks.get(3)
    np.testing.assert_allclose(lm.position, POSITION + 0.2, err_msg="the later answer wins")
    np.testing.assert_allclose(lm.R_drop, R_DROP * 4.0)
    np.testing.assert_allclose(lm.q_ref, q_ref, atol=1e-12)
    np.testing.assert_allclose(lm.rel, rel, atol=1e-12)
    np.testing.assert_allclose(lm.normal, normal, atol=1e-12)
    assert len(d.odometry.landmarks) == 1


def test_a_strip_is_sent_again_when_the_mounting_vehicle_moves_it():
    """A strip mounted since the last fix drifts with the estimate, so the next fix moves it. The
    vehicle then knows it better than when it bolted it on, and the team has to be told."""
    d = LocalizabilityDetector(DetectorConfig(platform="ugv", share_landmarks=True, **CAL))
    d.process(np.zeros((0, 3)), np.eye(4), drops=[(0, np.array([0.0, 1.53, 0.5]))])
    assert len(d.shared) == 1
    d.shared.clear()

    # a fix has moved it: exactly what _absolute_fix does to a landmark registered since
    d.odometry.landmarks.get(0).position += np.array([0.3, 0.0, 0.0])
    d.process(np.zeros((0, 3)), np.eye(4))
    assert len(d.shared) == 1, "the move has to be sent"
    np.testing.assert_allclose(d.shared[0]["position"], d.odometry.landmarks.get(0).position)

    # and a strip that has not moved is not sent again every scan
    d.shared.clear()
    d.process(np.zeros((0, 3)), np.eye(4))
    assert d.shared == []


def test_a_team_vehicle_uses_the_strips_it_sees():
    """It mounts nothing, so it has no drops. It still has to read its own detections, or it holds
    a teammate's strips and never fixes on them, and its team estimate comes out identical to its
    solo one (docs/failures.md number 33)."""
    d = follower()
    d.add_shared_landmark(0, POSITION, R_DROP, NORMAL)
    sensor = POSITION - np.array([1.0, 0.0, 0.7])
    det = MarkerDetection(slot=0, point_sensor=sensor, n_beams=8,
                          range_m=float(np.linalg.norm(sensor)), n_columns=2)
    before = d.odometry.T[:3, 3].copy()
    d.process(np.zeros((0, 3)), np.eye(4), detections=[det], track_yaw=0.0)
    d.process(np.zeros((0, 3)), np.eye(4), detections=[det], track_yaw=0.0)
    assert d.odometry.fix_log, "a known strip in view has to produce a fix"
    assert not np.allclose(d.odometry.T[:3, 3], before), "and the fix has to move the estimate"


def test_a_vehicle_that_is_not_in_the_team_ignores_the_topic():
    d = LocalizabilityDetector(DetectorConfig(platform="drone", gaze="across",
                                              ratio_threshold=CAL["ratio_threshold"]))
    assert not d.add_shared_landmark(0, POSITION, R_DROP, NORMAL)
    assert len(d.odometry.landmarks) == 0


def test_a_degenerate_normal_is_refused_rather_than_guessed():
    """The facing decides the sign of the fit's bias correction, so a vehicle that cannot be told
    which way the strip faces must not assume one."""
    d = follower()
    with pytest.raises(ValueError):
        d.add_shared_landmark(0, POSITION, R_DROP, np.zeros(2))


def test_a_mounting_vehicle_records_exactly_what_the_message_carries():
    """Slot, position, the 2x2 that placed it, the face normal. No map, no trajectory, no
    covariance over the sender's own run."""
    d = LocalizabilityDetector(DetectorConfig(platform="ugv", share_landmarks=True, **CAL))
    d.odometry.register_landmark(0, np.array([0.0, 1.53, 0.5]))
    rec = d._share(0)
    assert set(rec) == {"slot", "position", "drop_covariance", "normal"}
    assert rec["drop_covariance"].shape == (2, 2)
    assert float(np.linalg.norm(rec["normal"])) == pytest.approx(1.0)
    # the face points back at the pose that mounted it, which is across the tunnel
    np.testing.assert_allclose(rec["normal"], [0.0, -1.0], atol=1e-9)


def test_a_mounted_strip_is_queued_for_the_team_only_when_sharing_is_on():
    for share in (False, True):
        d = LocalizabilityDetector(DetectorConfig(platform="ugv", share_landmarks=share, **CAL))
        d.process(np.zeros((0, 3)), np.eye(4), drops=[(0, np.array([0.0, 1.53, 0.5]))])
        assert len(d.shared) == int(share), (share, d.shared)


def test_the_message_carries_the_record_unchanged():
    msgs = pytest.importorskip("locrec_msgs.msg")
    from std_msgs.msg import Header

    from locrec_ros.node import build_shared

    rec = {"slot": 7, "position": POSITION, "drop_covariance": R_DROP, "normal": NORMAL}
    msg = build_shared(Header(), rec)
    assert isinstance(msg, msgs.SharedLandmark)
    assert msg.slot == 7
    np.testing.assert_allclose([msg.position.x, msg.position.y, msg.position.z], POSITION)
    np.testing.assert_allclose(np.array(msg.drop_covariance).reshape(2, 2), R_DROP)
    np.testing.assert_allclose([msg.normal.x, msg.normal.y], NORMAL)
    assert msg.normal.z == 0.0, "the facing is horizontal by construction"


def test_the_round_trip_puts_back_what_was_sent():
    pytest.importorskip("locrec_msgs.msg")
    from std_msgs.msg import Header

    from locrec_ros.node import build_shared

    sender = LocalizabilityDetector(DetectorConfig(platform="ugv", share_landmarks=True, **CAL))
    sender.process(np.zeros((0, 3)), np.eye(4), drops=[(0, np.array([0.0, 1.53, 0.5]))])
    msg = build_shared(Header(), sender.shared.pop(0))

    receiver = follower()
    receiver.add_shared_landmark(msg.slot,
                                 np.array([msg.position.x, msg.position.y, msg.position.z]),
                                 np.array(msg.drop_covariance),
                                 np.array([msg.normal.x, msg.normal.y, msg.normal.z]))
    theirs = sender.odometry.landmarks.get(0)
    ours = receiver.odometry.landmarks.get(0)
    np.testing.assert_allclose(ours.position, theirs.position)
    np.testing.assert_allclose(ours.R_drop, theirs.R_drop)
