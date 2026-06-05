"""Unit tests for Step 4a timestamp / CSV helpers (no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phys4d.bounce.extract import load_train_timestamps, smooth_trajectory_savgol
from phys4d.bounce.load_4dgs import load_cfg_args, opacity_weighted_centroid


class TestStep4aTimestamps(unittest.TestCase):
    def test_frame_map_train_times_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_map = {
                "train": [
                    {"kept_index": 0, "original_frame": 10, "original_time_s": 0.2},
                    {"kept_index": 1, "original_frame": 20, "original_time_s": 0.0},
                    {"kept_index": 2, "original_frame": 30, "original_time_s": 0.1},
                ]
            }
            (root / "frame_map.json").write_text(json.dumps(frame_map), encoding="utf-8")
            ts = load_train_timestamps(root)
            self.assertEqual(len(ts), 3)
            times = [t.t_sec for t in ts]
            self.assertEqual(times, sorted(times))
            self.assertAlmostEqual(ts[0].t_norm, 0.0)
            self.assertAlmostEqual(ts[-1].t_norm, 1.0)

    def test_transforms_train_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transforms = {
                "frames": [
                    {"time": 0.5, "file_path": "images/cam00_00005"},
                    {"time": 0.0, "file_path": "images/cam00_00000"},
                    {"time": 0.5, "file_path": "images/cam01_00005"},
                ]
            }
            (root / "transforms_train.json").write_text(json.dumps(transforms), encoding="utf-8")
            ts = load_train_timestamps(root)
            self.assertEqual(len(ts), 2)
            self.assertEqual([t.t_sec for t in ts], [0.0, 0.5])

    def test_opacity_weighted_centroid(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        w = np.array([1.0, 1.0])
        c = opacity_weighted_centroid(pts, w)
        np.testing.assert_allclose(c, [1.0, 0.0, 0.0])

    def test_savgol_short_series(self) -> None:
        pos = np.random.randn(2, 3)
        out = smooth_trajectory_savgol(pos, window_length=5, polyorder=2)
        np.testing.assert_array_equal(out, pos)

    def test_cfg_args_rejects_all_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg_args"
            path.write_text("Namespace(all_train=True)\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_cfg_args(path)


if __name__ == "__main__":
    unittest.main()
