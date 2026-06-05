"""Unit tests for Step 4b physics fit (no GPU)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phys4d.bounce.physics import (
    detect_bounces,
    fit_physics_params,
    load_physics_params_json,
    load_trajectory_csv,
    run_step4b,
    simulate_trajectory,
)


def _write_traj(path: Path, times: np.ndarray, pos: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "t_sec", "x", "y", "z"])
        w.writeheader()
        for i, (t, p) in enumerate(zip(times, pos)):
            w.writerow(
                {
                    "frame": i,
                    "t_sec": float(t),
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                }
            )


class TestStep4bPhysics(unittest.TestCase):
    def test_simulate_single_bounce(self) -> None:
        times = np.linspace(0.0, 1.0, 61)
        p0 = np.array([0.0, 0.0, 1.0])
        v0 = np.array([0.1, 0.0, 0.0])
        traj = simulate_trajectory(
            times,
            p0=p0,
            v0=v0,
            gravity_z=-9.81,
            restitution=0.8,
            ground_z=0.0,
        )
        self.assertGreater(traj[:, 2].max(), 0.0)
        self.assertGreaterEqual(traj[:, 2].min(), -1e-6)

    def test_fit_recovers_synthetic_ball_drop(self) -> None:
        times = np.linspace(0.0, 0.8, 49)
        true_p0 = np.array([0.02, -0.01, 1.2])
        true_v0 = np.array([0.15, 0.05, 0.2])
        obs = simulate_trajectory(
            times,
            p0=true_p0,
            v0=true_v0,
            gravity_z=-9.81,
            restitution=0.75,
            ground_z=0.1,
        )
        obs = obs + np.random.default_rng(0).normal(0.0, 0.003, obs.shape)

        params, pred, mse = fit_physics_params(times, obs, fix_gravity=True)
        self.assertLess(mse, 0.01)
        self.assertAlmostEqual(params.gravity_z, -9.81, places=2)
        self.assertGreaterEqual(params.restitution, 0.3)
        self.assertLessEqual(params.restitution, 0.95)

    def test_detect_bounces_on_clean_sim(self) -> None:
        times = np.linspace(0.0, 1.2, 73)
        obs = simulate_trajectory(
            times,
            p0=np.array([0.0, 0.0, 1.0]),
            v0=np.zeros(3),
            gravity_z=-9.81,
            restitution=0.8,
            ground_z=0.0,
        )
        bounces = detect_bounces(times, obs[:, 2])
        self.assertGreaterEqual(len(bounces), 1)

    def test_run_step4b_writes_json(self) -> None:
        times = np.linspace(0.0, 0.6, 37)
        obs = simulate_trajectory(
            times,
            p0=np.array([0.0, 0.0, 0.9]),
            v0=np.array([0.0, 0.0, 0.0]),
            gravity_z=-9.81,
            restitution=0.85,
            ground_z=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traj = root / "trajectory_smoothed.csv"
            out = root / "step4b"
            _write_traj(traj, times, obs)
            result = run_step4b(trajectory_csv=traj, out_dir=out)
            self.assertTrue(result.params_json.is_file())
            self.assertTrue(result.fit_plot_png.is_file())
            loaded = load_physics_params_json(result.params_json)
            self.assertEqual(loaded.p0.shape, (3,))
            frames, t2, pos2 = load_trajectory_csv(traj)
            self.assertEqual(len(frames), len(times))

    def test_load_roundtrip(self) -> None:
        times = np.linspace(0.0, 0.4, 25)
        obs = simulate_trajectory(
            times,
            p0=np.array([0.0, 0.0, 0.5]),
            v0=np.zeros(3),
            gravity_z=-9.81,
            restitution=0.9,
            ground_z=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traj = root / "t.csv"
            _write_traj(traj, times, obs)
            run_step4b(trajectory_csv=traj, out_dir=root / "out")
            p = load_physics_params_json(root / "out" / "physics_params.json")
            self.assertAlmostEqual(p.gravity_z, -9.81)


if __name__ == "__main__":
    unittest.main()
