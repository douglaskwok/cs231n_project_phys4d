import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phys4d.camera_project import (  # noqa: E402
    project_gaussian_splatting_points,
    project_pybullet_points,
    load_gs_cameras,
    load_cameras,
)


class CameraProjectTest(unittest.TestCase):
    def test_pybullet_pose_projects_near_mask_at_frame0(self) -> None:
        cameras_json = REPO_ROOT / "outputs/sphere_bounce_m2/cameras.json"
        poses_csv = REPO_ROOT / "outputs/sphere_bounce_m2/object_poses.csv"
        if not cameras_json.is_file() or not poses_csv.is_file():
            self.skipTest("dataset not generated locally")

        import csv
        import imageio.v2 as imageio

        with cameras_json.open() as f:
            cam = json.load(f)["cameras"][0]
        with poses_csv.open() as f:
            row = next(csv.DictReader(f))
        # Empirically, z slightly below pose center matches segmentation at frame 0.
        z = float(row["z_m"]) - 0.05
        point = np.array([[0.0, 0.0, z]], dtype=np.float64)

        uv, visible = project_pybullet_points(point, cam)
        self.assertTrue(visible[0])
        mask = imageio.imread(REPO_ROOT / "outputs/sphere_bounce_m2/masks/cam00/frame00000.png")
        ys, xs = np.where(mask > 0)
        u_exp, v_exp = float(xs.mean()), float(ys.mean())
        self.assertLess(abs(uv[0, 0] - u_exp), 12.0)
        self.assertLess(abs(uv[0, 1] - v_exp), 12.0)

    def test_gs_projection_hits_mask_at_frame30(self) -> None:
        gs_cam_json = REPO_ROOT / "gs_sphere_bounce/cameras.json"
        ply_path = REPO_ROOT / "gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply"
        if not gs_cam_json.is_file() or not ply_path.is_file():
            self.skipTest("trained gaussians not downloaded")

        try:
            import imageio.v2 as imageio
            from plyfile import PlyData
        except ImportError:
            self.skipTest("imageio/plyfile missing")

        cam = load_gs_cameras(gs_cam_json)[0]
        v = np.asarray(PlyData.read(str(ply_path))["vertex"].data)
        top = np.argsort(v["opacity"])[-2000:]
        pts = np.stack([v["x"][top], v["y"][top], v["z"][top]], axis=1)

        mask = imageio.imread(
            REPO_ROOT / "outputs/sphere_bounce_m2/masks/cam00/frame00030.png"
        )
        uv, visible = project_gaussian_splatting_points(pts, cam)
        hits = 0
        for i in np.where(visible)[0]:
            ui = int(round(uv[i, 0]))
            vi = int(round(uv[i, 1]))
            if 0 <= ui < 256 and 0 <= vi < 256 and mask[vi, ui] > 0:
                hits += 1
        self.assertGreater(hits, 50)


if __name__ == "__main__":
    unittest.main()
