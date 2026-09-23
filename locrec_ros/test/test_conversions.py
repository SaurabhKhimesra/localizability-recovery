"""Conversion tests that run without a ROS graph or a ROS install.

These cover the Python side, which the viewer uses. The estimator's own conversions are
C++ and are tested in locrec_estimator/test/test_conversions.cpp.
"""
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locrec_ros.conversions import (  # noqa: E402
    odometry_to_pose,
    pointcloud2_to_xyz,
    stamp_to_ns,
)


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


# ---- odometry and stamps ---------------------------------------------------


class _Odometry:
    """Duck-typed nav_msgs/Odometry, for the same reason _Cloud is duck typed."""

    def __init__(self, position, quaternion):
        self.pose = SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
                orientation=SimpleNamespace(
                    x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
                ),
            )
        )


def test_odometry_pose_is_read_and_the_quaternion_normalised():
    half = np.sqrt(0.5)
    T = odometry_to_pose(_Odometry((1.0, 2.0, 3.0), (0.0, 0.0, 2 * half, 2 * half)))
    np.testing.assert_allclose(T[:3, 3], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-12)


def test_a_zero_quaternion_is_an_error_not_an_identity():
    """An identity rotation is a plausible prior that is wrong."""
    with pytest.raises(ValueError):
        odometry_to_pose(_Odometry((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))


def test_stamps_compare_exactly_as_nanoseconds():
    assert stamp_to_ns(SimpleNamespace(sec=3, nanosec=250_000_000)) == 3_250_000_000
