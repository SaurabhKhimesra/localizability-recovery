"""Conversion and action tests that run without a ROS graph or a ROS install."""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locrec_ros.conversions import pointcloud2_to_xyz  # noqa: E402
from locrec_ros.detector import DetectorConfig, LocalizabilityDetector  # noqa: E402


class _Field:
    def __init__(self, name, offset, datatype=7, count=1):
        self.name = name
        self.offset = offset
        self.datatype = datatype
        self.count = count


class _Cloud:
    """Duck-typed PointCloud2. The conversion is written against the wire format,
    so a plain object with the right attributes exercises exactly what the driver
    will hand the node."""

    def __init__(self, points, extra_bytes=0, bigendian=False, row_pad=0, dtype="f"):
        self.fields = [_Field("x", 0), _Field("y", 4), _Field("z", 8)]
        if dtype == "d":
            self.fields = [
                _Field("x", 0, 8),
                _Field("y", 8, 8),
                _Field("z", 16, 8),
            ]
        item = 8 if dtype == "d" else 4
        self.point_step = 3 * item + extra_bytes
        self.height = 1
        self.width = len(points)
        self.row_step = self.width * self.point_step + row_pad
        self.is_bigendian = bigendian
        order = ">" if bigendian else "<"
        blob = b""
        for p in points:
            blob += struct.pack(f"{order}3{dtype}", *p) + b"\x00" * extra_bytes
        self.data = blob + b"\x00" * row_pad


def test_round_trip_float32():
    pts = [(1.0, 2.0, 3.0), (-4.5, 0.25, 7.0)]
    out = pointcloud2_to_xyz(_Cloud(pts))
    np.testing.assert_allclose(out, pts, rtol=1e-6)


def test_extra_fields_between_xyz_and_the_next_point_are_skipped():
    """Every real driver packs intensity, ring and time after xyz. A hard-coded
    stride of twelve bytes is the classic way to get a plausible wrong cloud."""
    pts = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    out = pointcloud2_to_xyz(_Cloud(pts, extra_bytes=20))
    np.testing.assert_allclose(out, pts, rtol=1e-6)


def test_row_padding_is_respected():
    pts = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    out = pointcloud2_to_xyz(_Cloud(pts, row_pad=16))
    np.testing.assert_allclose(out, pts, rtol=1e-6)


def test_big_endian():
    pts = [(1.0, 2.0, 3.0)]
    out = pointcloud2_to_xyz(_Cloud(pts, bigendian=True))
    np.testing.assert_allclose(out, pts, rtol=1e-6)


def test_float64_fields():
    pts = [(1.5, 2.5, 3.5)]
    out = pointcloud2_to_xyz(_Cloud(pts, dtype="d"))
    np.testing.assert_allclose(out, pts, rtol=1e-12)


def test_non_finite_points_are_dropped():
    """A NaN reaching the registration turns the whole Hessian into NaN, which
    shows up as a detector that silently never fires."""
    pts = [(1.0, 2.0, 3.0), (float("nan"), 0.0, 0.0), (float("inf"), 1.0, 1.0)]
    out = pointcloud2_to_xyz(_Cloud(pts))
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], (1.0, 2.0, 3.0), rtol=1e-6)


def test_missing_field_is_an_error_not_a_silent_zero():
    cloud = _Cloud([(1.0, 2.0, 3.0)])
    cloud.fields = [_Field("x", 0), _Field("y", 4)]
    with pytest.raises(ValueError, match="z"):
        pointcloud2_to_xyz(cloud)


def test_truncated_data_is_an_error():
    cloud = _Cloud([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    cloud.data = cloud.data[:-8]
    with pytest.raises(ValueError, match="bytes"):
        pointcloud2_to_xyz(cloud)


def test_max_points_subsamples():
    pts = [(float(i), 0.0, 0.0) for i in range(100)]
    out = pointcloud2_to_xyz(_Cloud(pts), max_points=10)
    assert out.shape[0] <= 10


# ---- detector and action rule ---------------------------------------------


def _corridor(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-8.0, 8.0, n)
    z = rng.uniform(-0.6, 1.8, n)
    y = np.where(rng.random(n) < 0.5, -1.6, 1.6)
    walls = np.stack([x, y, z], axis=1)
    floor = np.stack(
        [rng.uniform(-8, 8, n), rng.uniform(-1.6, 1.6, n), np.full(n, -0.7)], axis=1
    )
    return np.vstack([walls, floor])


# the same corridor on every scan is a sensor standing still, so its odometry pose
# is the identity every time, and that is the true prior rather than a stand-in
STILL = np.eye(4)

# these tests exercise pairing, ranges and message shapes, none of which depend
# on where the threshold sits, so any finite value will do
ANY_THRESHOLD = 2.0e-3
CAL = dict(ratio_threshold=ANY_THRESHOLD, marker_reliable_range_m=7.0, scheduler_margin_m=1.5)



def test_detector_returns_nothing_until_it_can_register():
    det = LocalizabilityDetector(DetectorConfig(**CAL))
    assert det.process(_corridor(), STILL) is None


def test_detector_fires_in_a_corridor_and_names_the_weak_direction():
    det = LocalizabilityDetector(DetectorConfig(marker_reliable_range_m=7.0, scheduler_margin_m=1.5, ratio_threshold=0.05))
    out = None
    for _ in range(4):
        out = det.process(_corridor(), STILL)
    assert out is not None
    assert out.is_degenerate
    assert abs(out.weak_direction[0]) > 0.9, out.weak_direction
    assert out.n_points > 0
    assert out.eigenvalues[0] <= out.eigenvalues[2]


def test_ugv_recommends_a_marker_on_the_falling_edge():
    det = LocalizabilityDetector(DetectorConfig(marker_reliable_range_m=7.0, scheduler_margin_m=1.5, ratio_threshold=0.05, platform="ugv"))
    actions = [det.process(_corridor(), STILL) for _ in range(4)]
    actions = [a.action for a in actions if a is not None]
    assert actions[0] == "drop_marker", actions
    assert actions[1] == "none", "one edge, one marker, not one per scan"


def test_drone_recommends_a_yaw_across_the_weak_direction():
    det = LocalizabilityDetector(DetectorConfig(marker_reliable_range_m=7.0, scheduler_margin_m=1.5, ratio_threshold=0.05, platform="drone"))
    out = None
    for _ in range(4):
        out = det.process(_corridor(), STILL)
    assert out.action.startswith("yaw_to:")
    deg = float(out.action.split(":")[1])
    # the corridor runs along x, so the useful direction to look is across it
    assert min(abs(deg - 90.0), abs(deg + 90.0)) < 20.0, deg


def test_message_assembly_without_ros():
    """build_message is imported lazily so this file runs with no ROS installed."""
    pytest.importorskip("rclpy")
    from locrec_ros.node import build_message

    det = LocalizabilityDetector(DetectorConfig(marker_reliable_range_m=7.0, scheduler_margin_m=1.5, ratio_threshold=0.05))
    out = None
    for _ in range(4):
        out = det.process(_corridor(), STILL)
    msg = build_message(None, out)
    assert len(msg.eigenvalues) == 3
    assert len(msg.weak_direction) == 3
    assert msg.n_points == out.n_points
