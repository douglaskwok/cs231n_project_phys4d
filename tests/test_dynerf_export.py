"""Tests for DyNeRF / 4DGS dataset export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.dynerf_export import export_dynerf_dataset  # noqa: E402


class TestDynerfExport(unittest.TestCase):
    def test_export_counts_and_time_field(self) -> None:
        m2 = REPO_ROOT / "outputs" / "sphere_bounce_m2"
        rgb = m2 / "rgb"
        cams = m2 / "cameras.json"
        if not rgb.is_dir() or not cams.is_file():
            self.skipTest("Run generate_sphere_bounce_dataset.py first")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dynerf"
            meta = export_dynerf_dataset(
                rgb_root=rgb,
                cameras_json=cams,
                out_dir=out,
                train_cameras=[0, 1, 2, 3],
                test_cameras=[4, 5],
                train_frame_range=(0, 59),
                test_frame_range=(60, 89),
                fps=60.0,
                copy_images=False,
            )
            self.assertEqual(meta["num_train_views"], 4 * 60)
            self.assertEqual(meta["num_test_views"], 2 * 30)

            with (out / "transforms_train.json").open(encoding="utf-8") as f:
                train = json.load(f)
            self.assertEqual(len(train["frames"]), 240)
            self.assertIn("time", train["frames"][0])
            self.assertAlmostEqual(train["frames"][0]["time"], 0.0)
            self.assertAlmostEqual(train["frames"][-1]["time"], 59 / 60.0)

            img = list((out / "images").glob("cam00_*.png"))
            self.assertGreater(len(img), 0)


if __name__ == "__main__":
    unittest.main()
