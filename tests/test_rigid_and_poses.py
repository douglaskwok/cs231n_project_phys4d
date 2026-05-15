import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phys4d.gaussian_ply import GaussianCloud, warp_gaussians_to_frame
from phys4d.poses import ObjectPose, ObjectPoseTrajectory
from phys4d.rigid import (
    apply_pose_to_points,
    invert_pose,
    pose_matrix,
    quat_multiply_xyzw,
    relative_pose,
)


class RigidPoseTest(unittest.TestCase):
    def test_pure_translation_warp(self) -> None:
        ref = ObjectPose(
            frame=0,
            time_s=0.0,
            position=np.array([0.0, 0.0, 1.6]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            linear_velocity=np.zeros(3),
        )
        target = ObjectPose(
            frame=1,
            time_s=0.016,
            position=np.array([0.0, 0.0, 1.5]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            linear_velocity=np.zeros(3),
        )
        traj = ObjectPoseTrajectory(poses=[ref, target])

        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
        verts = np.zeros(2, dtype=dtype)
        verts["x"] = [0.0, 0.05]
        verts["y"] = 0.0
        verts["z"] = [1.6, 1.65]
        verts["rot_0"] = 1.0
        verts["rot_1"] = 0.0
        verts["rot_2"] = 0.0
        verts["rot_3"] = 0.0
        cloud = GaussianCloud(vertices=verts)

        warped = warp_gaussians_to_frame(cloud, traj, ref_frame=0, target_frame=1)
        np.testing.assert_allclose(warped.xyz[:, 2], [1.5, 1.55], rtol=0, atol=1e-5)

    def test_relative_pose_inverse(self) -> None:
        pos = np.array([0.1, -0.2, 1.3])
        quat = np.array([0.1, 0.2, 0.3, 0.9])
        quat = quat / np.linalg.norm(quat)
        mat = pose_matrix(pos, quat)
        delta = relative_pose(mat, mat)
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(apply_pose_to_points(pts, delta), pts, atol=1e-6)

    def test_quat_multiply_identity(self) -> None:
        q_id = np.array([0.0, 0.0, 0.0, 1.0])
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        out = quat_multiply_xyzw(q_id, q)
        np.testing.assert_allclose(out, q, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
