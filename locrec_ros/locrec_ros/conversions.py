"""PointCloud2 and Odometry to numpy, with no ROS import anywhere in this file.

The node needs rclpy. This does not, which is the point: the conversion is the
part most likely to be wrong and the part hardest to test inside a ROS graph, so
it is written against the wire format rather than against the Python class. Any
object with the message's attributes works, including a plain namespace built in
a test.

Field layout is read from the message rather than assumed. A Velodyne driver, an
Ouster driver and a rosbag replay all describe x, y and z with the same names but
at different offsets and sometimes different datatypes, and a hard-coded stride is
the classic way to get a point cloud that looks plausible and is wrong.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["DATATYPES", "odometry_to_pose", "pointcloud2_to_xyz", "stamp_to_ns"]

# sensor_msgs/PointField datatype constants, repeated here so this module does not
# import ROS. The numbers are part of the message definition and do not change.
DATATYPES = {
    1: np.dtype(np.int8),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int16),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int32),
    6: np.dtype(np.uint32),
    7: np.dtype(np.float32),
    8: np.dtype(np.float64),
}


def pointcloud2_to_xyz(msg, max_points: int | None = None) -> np.ndarray:
    """Return an (N, 3) float64 array of finite xyz points.

    ``msg`` needs ``height``, ``width``, ``point_step``, ``row_step``, ``fields``,
    ``is_bigendian`` and ``data``. Rows are unpacked with ``row_step`` rather than
    ``width * point_step`` because organised clouds pad their rows.

    Non-finite points are dropped. Most drivers mark an invalid return with NaN,
    and a NaN reaching the registration turns the whole Hessian into NaN, which
    surfaces as a detector that silently never fires.
    """
    names = {f.name: f for f in msg.fields}
    missing = {"x", "y", "z"} - set(names)
    if missing:
        raise ValueError(f"point cloud has no {sorted(missing)} field")

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n_rows = max(int(msg.height), 1)
    row_step = int(msg.row_step) or int(msg.width) * int(msg.point_step)
    expected = n_rows * row_step
    if raw.size < expected:
        raise ValueError(f"point cloud data is {raw.size} bytes, expected {expected}")
    raw = raw[:expected].reshape(n_rows, row_step)
    point_step = int(msg.point_step)
    usable = (row_step // point_step) * point_step
    raw = raw[:, :usable].reshape(-1, point_step)

    n_points = int(msg.width) * n_rows
    raw = raw[:n_points]

    out = np.empty((raw.shape[0], 3), dtype=np.float64)
    for i, axis in enumerate("xyz"):
        field = names[axis]
        dtype = DATATYPES[int(field.datatype)]
        dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
        offset = int(field.offset)
        chunk = raw[:, offset : offset + dtype.itemsize]
        out[:, i] = np.ascontiguousarray(chunk).view(dtype).reshape(-1)

    out = out[np.isfinite(out).all(axis=1)]
    if max_points is not None and out.shape[0] > max_points:
        stride = int(np.ceil(out.shape[0] / max_points))
        out = out[::stride]
    return out


def stamp_to_ns(stamp) -> int:
    """A ``builtin_interfaces/Time`` as integer nanoseconds, so stamps compare exactly."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def odometry_to_pose(msg) -> np.ndarray:
    """The pose in a ``nav_msgs/Odometry`` as a 4x4 matrix: child frame in odometry frame.

    ``msg`` needs ``pose.pose.position`` and ``pose.pose.orientation``. The quaternion
    is normalised. A zero or non-finite one is an error rather than an identity, for
    the same reason a missing point field is: an identity rotation is a plausible
    prior that is wrong.
    """
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    t = np.array([p.x, p.y, p.z], dtype=float)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(t).all() or not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("odometry pose has a non-finite position or a zero quaternion")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat / norm).as_matrix()
    T[:3, 3] = t
    return T
