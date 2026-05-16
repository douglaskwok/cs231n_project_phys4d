"""Shape tests for param-ID CNN."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.param_ident.model import MultiViewParamPredictor  # noqa: E402


class TestParamIdentShapes(unittest.TestCase):
    def test_forward_bounds(self) -> None:
        model = MultiViewParamPredictor()
        x = torch.randn(2, 8, 6, 3, 128, 128)
        y = model(x)
        self.assertEqual(y.shape, (2, 3))
        self.assertTrue((y[:, 0] >= 0.3).all() and (y[:, 0] <= 0.95).all())
        self.assertTrue((y[:, 1] >= 0.1).all() and (y[:, 1] <= 2.0).all())


if __name__ == "__main__":
    unittest.main()
