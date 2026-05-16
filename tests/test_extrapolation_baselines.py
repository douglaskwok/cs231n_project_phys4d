"""Tests for extrapolation baseline ordering on synthetic bounce."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d import BounceConfig  # noqa: E402
from phys4d.extrapolation_baselines import (  # noqa: E402
    constant_hold_baseline,
    linear_delta_baseline,
    physics_bounce_baseline,
)
from phys4d.poses import load_object_poses_csv  # noqa: E402


class ExtrapolationBaselinesTest(unittest.TestCase):
    def setUp(self) -> None:
        poses = REPO_ROOT / "outputs" / "sphere_bounce_m2" / "object_poses.csv"
        if not poses.is_file():
            self.skipTest("Run generate_sphere_bounce_dataset.py first")
        self.traj = load_object_poses_csv(poses)
        self.z0 = 1.6
        self.train_end = 59
        self.bounce_cfg = BounceConfig(steps=90, z0=self.z0)

    def test_physics_beats_naive_on_test_z(self) -> None:
        const = constant_hold_baseline(self.traj, train_end_frame=self.train_end, z0=self.z0)
        linear = linear_delta_baseline(self.traj, train_end_frame=self.train_end, z0=self.z0)
        phys = physics_bounce_baseline(
            self.traj, bounce_cfg=self.bounce_cfg, train_end_frame=self.train_end, z0=self.z0
        )
        self.assertLess(phys.z_test_mse, const.z_test_mse)
        self.assertLess(phys.z_test_mse, linear.z_test_mse)


if __name__ == "__main__":
    unittest.main()
