"""Unit tests for Step 4c extrapolation (no GPU)."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phys4d.bounce.extrapolate import run_step4c, _build_rollout_timeline, _read_frame_map
from phys4d.bounce.physics import fit_physics_params, simulate_trajectory


def _write_traj(path: Path, frames: np.ndarray, times: np.ndarray, pos: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "t_sec", "x", "y", "z"])
        w.writeheader()
        for fr, t, p in zip(frames, times, pos):
            w.writerow({"frame": int(fr), "t_sec": float(t), "x": p[0], "y": p[1], "z": p[2]})


class TestStep4cExtrapolate(unittest.TestCase):
    def test_read_frame_map_and_timeline(self) -> None:
        train = [{"original_frame": 0, "original_time_s": 0.0}, {"original_frame": 1, "original_time_s": 1 / 60}]
        test = [{"original_frame": 2, "original_time_s": 2 / 60}]
        with tempfile.TemporaryDirectory() as tmp:
            fm = Path(tmp) / "frame_map.json"
            fm.write_text(json.dumps({"train": train, "test": test, "fps": 60}), encoding="utf-8")
            tr, te = _read_frame_map(fm)
            self.assertEqual(len(tr), 2)
            self.assertEqual(len(te), 1)
            frames = np.array([0, 1], dtype=np.int32)
            times = np.array([0.0, 1 / 60], dtype=np.float64)
            obs = np.zeros((2, 3))
            f, t, o, tm, xm = _build_rollout_timeline(
                tr, te, frames_traj=frames, times_traj=times, obs_traj=obs, fps=60.0
            )
            self.assertEqual(f.shape[0], 3)
            self.assertTrue(xm[2])

    def test_run_step4c_end_to_end(self) -> None:
        fps = 60.0
        train_frames = np.arange(0, 40, dtype=np.int32)
        test_frames = np.arange(40, 50, dtype=np.int32)
        times_train = train_frames.astype(np.float64) / fps
        obs_train = simulate_trajectory(
            times_train,
            p0=np.array([0.0, 0.0, 1.0]),
            v0=np.zeros(3),
            gravity_z=-9.81,
            restitution=0.8,
            ground_z=0.0,
        )
        params, _, _ = fit_physics_params(times_train, obs_train, fix_gravity=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traj = root / "traj.csv"
            _write_traj(traj, train_frames, times_train, obs_train)

            step4b = root / "step4b"
            step4b.mkdir()
            (step4b / "physics_params.json").write_text(
                json.dumps(
                    {
                        "params": {
                            "p0": params.p0.tolist(),
                            "v0": params.v0.tolist(),
                            "gravity_z_m_s2": params.gravity_z,
                            "restitution": params.restitution,
                            "ground_z_m": params.ground_z,
                        }
                    }
                ),
                encoding="utf-8",
            )

            train_rows = [
                {"original_frame": int(f), "original_time_s": float(f) / fps}
                for f in train_frames
            ]
            test_rows = [
                {"original_frame": int(f), "original_time_s": float(f) / fps}
                for f in test_frames
            ]
            fm = root / "frame_map.json"
            fm.write_text(json.dumps({"train": train_rows, "test": test_rows, "fps": fps}), encoding="utf-8")

            result = run_step4c(
                params_json=step4b / "physics_params.json",
                frame_map_path=fm,
                out_dir=root / "step4c",
                trajectory_csv=traj,
            )
            self.assertEqual(result.num_test_frames, len(test_frames))
            self.assertTrue(result.predicted_csv.is_file())
            self.assertTrue(np.isfinite(result.boundary_gap_m))
            self.assertGreaterEqual(result.min_clearance_m, -1e-4)


if __name__ == "__main__":
    unittest.main()
