"""Step 5 overlay fallback smoke test (no Wu CUDA)."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phys4d.bounce.render_compose import run_step5
from tests.test_step5_gaussian_ply import _make_vertices

_HAS_PLYFILE = importlib.util.find_spec("plyfile") is not None


class TestStep5Overlay(unittest.TestCase):
    def test_force_overlay_renders_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = _make_vertices(3)
            bg = _make_vertices(2)
            (root / "obj.ply").touch()
            (root / "bg.ply").touch()

            pred = root / "pred.csv"
            with pred.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "t_sec", "x", "y", "z"])
                w.writeheader()
                w.writerow({"frame": 60, "t_sec": 1.0, "x": 0.0, "y": 0.0, "z": 0.5})

            ref = root / "ref.csv"
            with ref.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "t_sec", "x", "y", "z"])
                w.writeheader()
                w.writerow({"frame": 0, "t_sec": 0.0, "x": 0.0, "y": 0.0, "z": 0.5})

            export = root / "export"
            export.mkdir()
            c2w = np.eye(4).tolist()
            transforms = {
                "w": 64,
                "h": 64,
                "fl_x": 32.0,
                "fl_y": 32.0,
                "cx": 32.0,
                "cy": 32.0,
                "frames": [
                    {
                        "file_path": "images/cam00_00060",
                        "transform_matrix": c2w,
                        "time": 1.0,
                    }
                ],
            }
            (export / "transforms_test.json").write_text(json.dumps(transforms), encoding="utf-8")

            with patch(
                "phys4d.bounce.render_compose.load_gaussian_vertices",
                side_effect=[obj, bg],
            ):
                result = run_step5(
                    canonical_ply=root / "obj.ply",
                    bg_ply=root / "bg.ply",
                    predicted_csv=pred,
                    ref_traj_csv=ref,
                    dynerf_export=export,
                    out_dir=root / "step5",
                    force_overlay=True,
                )
            self.assertEqual(result.render_mode, "overlay_fallback")
            self.assertEqual(result.num_rendered, 1)
            png = result.renders_dir / "0060_cam0.png"
            self.assertTrue(png.is_file())


if __name__ == "__main__":
    unittest.main()
