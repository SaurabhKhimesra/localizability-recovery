"""Minimal SE(3) helpers. Right-handed, column-vector convention.

A pose T is a 4x4 homogeneous matrix mapping body/sensor coordinates into the
world (map) frame:  p_world = T @ [p_body; 1].
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rotz",
    "euler_zyx",
    "make_T",
    "inv_T",
    "transform_points",
    "so3_log",
    "se3_log",
    "pose_error",
]


def rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_zyx(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of R, numerically safe near 0 and pi."""
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if np.pi - theta < 1e-6:
        # near pi: use the symmetric part
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 0:
            axis = A[:, k] / axis[k]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def se3_log(T: np.ndarray) -> np.ndarray:
    """(rotation vector, translation) stacked as a length-6 vector.

    Translation is the raw translation column, not the twist; this is only used
    for reporting errors, never for optimisation.
    """
    return np.concatenate([so3_log(T[:3, :3]), T[:3, 3]])


def pose_error(T_est: np.ndarray, T_true: np.ndarray) -> tuple[float, float]:
    """(translation error in m, rotation error in rad)."""
    dT = inv_T(T_true) @ T_est
    return float(np.linalg.norm(dT[:3, 3])), float(np.linalg.norm(so3_log(dT[:3, :3])))
