"""The closed loop: the report join, where the calibrated numbers come from, the marker rule.

All of it runs without a ROS graph or a ROS install.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locrec_ros.calibration import resolve_calibration  # noqa: E402
from locrec_ros.detector import DetectorConfig, LocalizabilityDetector  # noqa: E402
from locrec_ros.prior import PriorPairer  # noqa: E402

from locrec.policies import LocalizabilityScheduler  # noqa: E402

SEC = 1_000_000_000
POSE = np.eye(4)
CAL = dict(ratio_threshold=2.0e-3, marker_reliable_range_m=7.0, scheduler_margin_m=1.5)


# ---- a scan waits for the report that carries its stamp ---------------------


def test_a_scan_waits_for_its_report_whichever_arrives_first():
    for order in ("report first", "scan first"):
        p = PriorPairer(require_side=True)
        p.add_odometry(0, POSE)
        p.add_odometry(10 * SEC, POSE)
        if order == "report first":
            assert p.add_side(5 * SEC, "report") == []
            (item,) = p.add_scan(5 * SEC, "scan")
        else:
            assert p.add_scan(5 * SEC, "scan") == []
            (item,) = p.add_side(5 * SEC, "report")
        assert (item.payload, item.side) == ("scan", "report"), order


def test_the_join_is_exact_never_the_nearest_report():
    p = PriorPairer(require_side=True)
    p.add_odometry(0, POSE)
    p.add_odometry(10 * SEC, POSE)
    p.add_side(5 * SEC - 1, "one nanosecond early")
    assert p.add_scan(5 * SEC, "scan") == []


def test_a_scan_whose_report_can_no_longer_come_is_dropped_and_counted():
    p = PriorPairer(require_side=True)
    p.add_odometry(0, POSE)
    p.add_odometry(10 * SEC, POSE)
    assert p.add_scan(5 * SEC, "orphan") == []
    # a later report has arrived, so the one for 5 s never will
    assert p.add_side(6 * SEC, "report") == []
    assert (p.dropped_no_side, p.waiting) == (1, 0)
    (item,) = p.add_scan(6 * SEC, "scan")
    assert item.side == "report"


def test_without_require_side_nothing_waits_for_a_report():
    p = PriorPairer()
    p.add_odometry(0, POSE)
    p.add_odometry(10 * SEC, POSE)
    (item,) = p.add_scan(5 * SEC, "scan")
    assert item.side is None


# ---- calibrated numbers have one source -------------------------------------


def _thresholds(tmp_path, **values) -> str:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(values))
    return str(path)


def test_values_come_from_the_file_and_the_drone_reads_its_own_threshold(tmp_path):
    f = _thresholds(tmp_path, ratio_threshold=2e-3, drone_ratio_threshold=3e-3,
                    marker_reliable_range_m=7.0, scheduler_margin_m=1.5)
    assert resolve_calibration("ugv", {}, f) == {
        "ratio_threshold": 2e-3, "marker_reliable_range_m": 7.0, "scheduler_margin_m": 1.5}
    assert resolve_calibration("drone", {}, f) == {"ratio_threshold": 3e-3}


def test_an_explicit_parameter_wins_over_the_file(tmp_path):
    f = _thresholds(tmp_path, ratio_threshold=2e-3, marker_reliable_range_m=7.0,
                    scheduler_margin_m=1.5)
    got = resolve_calibration("ugv", {"ratio_threshold": 1.8e-3}, f)
    assert got["ratio_threshold"] == 1.8e-3 and got["marker_reliable_range_m"] == 7.0


def test_nothing_is_defaulted_and_the_error_names_what_is_missing(tmp_path):
    with pytest.raises(ValueError, match="ratio_threshold"):
        resolve_calibration("ugv", {}, None)
    f = _thresholds(tmp_path, ratio_threshold=2e-3)
    with pytest.raises(ValueError, match="marker_reliable_range_m"):
        resolve_calibration("ugv", {}, f)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_calibration("ugv", {}, str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="marker_reliable_range_m"):
        LocalizabilityDetector(DetectorConfig(ratio_threshold=2e-3))


# ---- the marker rule is the core's scheduler, not a copy ---------------------


class _Loc:
    def __init__(self, ratio):
        self.ratio, self.weak_direction = ratio, np.array([1.0, 0.0, 0.0])

    def is_degenerate(self, cfg):
        return self.ratio < cfg.ratio_threshold


def test_a_recovery_does_not_buy_another_marker():
    """`docs/failures.md` 17, which the wrapper's own rule had reintroduced: the ratio
    chatters across the threshold, and every re-entry used to cost a marker."""
    det = LocalizabilityDetector(DetectorConfig(**CAL))
    actions = []
    for k in range(60):  # 30 m of tunnel, blind throughout but for a blip every 2 m
        det._estimated_distance = 0.5 * k
        actions.append(det._action(_Loc(5e-3 if k % 4 == 3 else 1.7e-3)))
    drops = [0.5 * k for k, a in enumerate(actions) if a == "drop_marker"]
    # twice the reliable range less the margin is 12.5 m, so 0, 12.5 and 25 m
    assert drops == [0.0, 12.5, 25.0], drops


def test_the_wrapper_decides_exactly_what_the_scheduler_decides():
    rng = np.random.default_rng(0)
    ratios = np.where(rng.random(400) < 0.7, 1.7e-3, 6e-3)
    det = LocalizabilityDetector(DetectorConfig(**CAL))
    core = LocalizabilityScheduler(ratio_threshold=CAL["ratio_threshold"],
                                   reliable_range_m=7.0, margin_m=1.5)

    class Ctx:
        pass

    for k, ratio in enumerate(ratios):
        det._estimated_distance = 0.5 * k
        ctx = Ctx()
        ctx.localizability, ctx.estimated_distance = _Loc(ratio), 0.5 * k
        expected = "drop_marker" if core(ctx).get("drop_marker") else "none"
        assert det._action(_Loc(ratio)) == expected, k


def test_the_estimator_is_given_the_sensor_model_markers_need():
    """Without it the core returns no marker terms at all, silently."""
    det = LocalizabilityDetector(DetectorConfig(**CAL))
    assert det.odometry.lidar_spec is not None
