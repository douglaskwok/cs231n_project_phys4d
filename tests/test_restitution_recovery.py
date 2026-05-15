import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phys4d import BounceConfig, fit_restitution, restitution_from_raw, simulate_bounce


class RestitutionRecoveryTest(unittest.TestCase):
    def test_gradient_flows_to_restitution(self) -> None:
        config = BounceConfig(steps=90)
        target_z = simulate_bounce(torch.tensor(0.75), config).detach()

        raw = torch.tensor(0.0, requires_grad=True)
        restitution = restitution_from_raw(raw)
        predicted_z = simulate_bounce(restitution, config)
        loss = torch.mean((predicted_z - target_z) ** 2)
        loss.backward()

        self.assertIsNotNone(raw.grad)
        self.assertGreater(abs(raw.grad.item()), 1e-6)

    def test_recovers_synthetic_restitution(self) -> None:
        config = BounceConfig(steps=90)
        true_restitution = 0.68
        observed_z = simulate_bounce(torch.tensor(true_restitution), config).detach()

        result = fit_restitution(
            observed_z,
            config=config,
            initial_raw=-1.0,
            lr=0.08,
            iterations=400,
        )

        self.assertLess(result.final_loss, result.initial_loss)
        self.assertLess(abs(result.final_restitution - true_restitution), 0.03)


if __name__ == "__main__":
    unittest.main()
