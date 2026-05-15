"""Rigid transforms (SE(3)) for object-centric Gaussian warping."""

from __future__ import annotations

import numpy as np


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] -> 3x3 rotation matrix."""

    q = np.asarray(quat, dtype=np.float64).reshape(4)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion [qx, qy, qz, qw]."""

    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_multiply_xyzw(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Hamilton product; quaternions in [qx, qy, qz, qw] order (supports broadcasting)."""

    left = np.asarray(q_left, dtype=np.float64)
    right = np.asarray(q_right, dtype=np.float64)
    x1, y1, z1, w1 = left[..., 0], left[..., 1], left[..., 2], left[..., 3]
    x2, y2, z2, w2 = right[..., 0], right[..., 1], right[..., 2], right[..., 3]
    out = np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )
    return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)


def pose_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build 4x4 body-to-world transform."""

    rot = quat_xyzw_to_matrix(quat_xyzw)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return mat


def invert_pose(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""

    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rot = m[:3, :3]
    trans = m[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def relative_pose(ref_to_world: np.ndarray, target_to_world: np.ndarray) -> np.ndarray:
    """Delta that maps points from ref body frame to target body frame in world coords.

    p_world = target @ (inv(ref) @ p_world_at_ref)  <=>  apply (target @ inv(ref)) to
    points expressed in the canonical (ref) world placement.
    """

    return target_to_world @ invert_pose(ref_to_world)


def apply_pose_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""

    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts, ones], axis=1)
    out = (transform @ hom.T).T
    return out[:, :3]


def apply_pose_to_gaussian_rotations(
    rot_wxyz: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Compose rigid rotation (from transform) with per-Gaussian quaternions (w,x,y,z)."""

    rot_world = transform[:3, :3]
    q_delta_xyzw = matrix_to_quat_xyzw(rot_world)
    q_gauss_wxyz = np.asarray(rot_wxyz, dtype=np.float64)
    q_gauss_xyzw = np.stack(
        [q_gauss_wxyz[:, 1], q_gauss_wxyz[:, 2], q_gauss_wxyz[:, 3], q_gauss_wxyz[:, 0]],
        axis=1,
    )
    q_delta_batch = np.broadcast_to(q_delta_xyzw, q_gauss_xyzw.shape)
    composed_xyzw = quat_multiply_xyzw(q_delta_batch, q_gauss_xyzw)
    return np.stack(
        [
            composed_xyzw[:, 3],
            composed_xyzw[:, 0],
            composed_xyzw[:, 1],
            composed_xyzw[:, 2],
        ],
        axis=1,
    )
