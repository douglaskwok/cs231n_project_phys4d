"""Unit tests for Step 6 metrics (no GPU)."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phys4d.bounce.metrics import _read_predicted_csv, _r2_pearson, run_step6


def _write_poses(path: Path, frames: list[int], positions: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "time_s",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
                "vx_m_s",
                "vy_m_s",
                "vz_m_s",
            ],
        )
        w.writeheader()
        for i, fr in enumerate(frames):
            p = positions[i]
            w.writerow(
                {
                    "frame": fr,
                    "time_s": fr / 60.0,
                    "x_m": p[0],
                    "y_m": p[1],
                    "z_m": p[2],
                    "qx": 0,
                    "qy": 0,
                    "qz": 0,
                    "qw": 1,
                    "vx_m_s": 0,
                    "vy_m_s": 0,
                    "vz_m_s": 0,
                }
            )


class TestStep6Metrics(unittest.TestCase):
    def test_r2_perfect_match(self) -> None:
        x = np.linspace(0, 1, 20).reshape(-1, 1)
        self.assertGreater(_r2_pearson(x, x), 0.99)

    def test_run_step6_trajectory_only(self) -> None:
        frames = [10, 11, 12]
        pred = np.stack(
            [np.linspace(0, 0.1, 3), np.zeros(3), np.linspace(1, 0.9, 3)],
            axis=1,
        )
        gt = pred + np.array([0.01, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_csv = root / "pred.csv"
            with pred_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "t_sec", "x", "y", "z"])
                w.writeheader()
                for fr, p in zip(frames, pred):
                    w.writerow({"frame": fr, "t_sec": fr / 60, "x": p[0], "y": p[1], "z": p[2]})

            poses = root / "poses.csv"
            _write_poses(poses, frames, gt)

            renders = root / "renders"
            renders.mkdir()
            (renders / "0010_cam0.png").touch()

            rgb = root / "rgb" / "cam00"
            rgb.mkdir(parents=True)
            (rgb / "frame00010.png").touch()

            result = run_step6(
                predicted_csv=pred_csv,
                gt_poses_csv=poses,
                rendered_dir=renders,
                gt_rgb_root=root / "rgb",
                out_dir=root / "step6",
                skip_render_metrics=True,
            )
            self.assertTrue(result.metrics_json.is_file())
            blob = json.loads(result.metrics_json.read_text())
            self.assertIn("trajectory", blob)
            self.assertIsNone(blob["render"])

    def test_read_predicted_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.csv"
            with p.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "t_sec", "x", "y", "z"])
                w.writeheader()
                w.writerow({"frame": 5, "t_sec": 0.1, "x": 1, "y": 2, "z": 3})
            fr, xyz = _read_predicted_csv(p)
            self.assertEqual(int(fr[0]), 5)
            np.testing.assert_allclose(xyz[0], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
