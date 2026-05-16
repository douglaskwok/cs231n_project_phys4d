"""Shape and loss tests for visual dynamics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.visual_dynamics.dynamics import ObjectTokenDynamics  # noqa: E402
from phys4d.visual_dynamics.losses import quat_geodesic_loss, state_loss  # noqa: E402


class TestVisualDynamics(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = ObjectTokenDynamics(history=4, visual_dim=32, use_visual=True)
        hist = torch.randn(2, 1, 4, 10)
        vis = torch.randn(2, 1, 32)
        delta = model(hist, vis)
        self.assertEqual(delta.shape, (2, 1, 10))
        nxt = model.predict_next(hist, vis)
        self.assertEqual(nxt.shape, (2, 1, 10))

    def test_states_only(self) -> None:
        model = ObjectTokenDynamics(history=4, visual_dim=32, use_visual=False)
        hist = torch.randn(2, 1, 4, 10)
        vis = torch.zeros(2, 1, 32)
        nxt = model.predict_next(hist, vis)
        self.assertEqual(nxt.shape, (2, 1, 10))

    def test_quat_loss(self) -> None:
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        self.assertLess(float(quat_geodesic_loss(q, q)), 1e-6)


if __name__ == "__main__":
    unittest.main()
