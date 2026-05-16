"""Shape tests for visual dynamics modules."""

from __future__ import annotations

import unittest

import torch

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d.visual_dynamics.dynamics_head import VisualDynamicsModel  # noqa: E402


class VisualDynamicsShapesTest(unittest.TestCase):
    def test_forward(self) -> None:
        model = VisualDynamicsModel(history=4, state_dim=10, visual_dim=32)
        hist = torch.randn(2, 4, 10)
        img = torch.randn(2, 3, 64, 64)
        delta = model(hist, img)
        self.assertEqual(delta.shape, (2, 10))
        nxt = model.predict_next(hist, img)
        self.assertEqual(nxt.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
