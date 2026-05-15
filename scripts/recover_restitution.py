#!/usr/bin/env python
"""Recover sphere-ground restitution from a synthetic bounce trajectory."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phys4d import BounceConfig, fit_restitution, simulate_bounce  # noqa: E402


def main() -> int:
    torch.manual_seed(7)

    config = BounceConfig(steps=90, z0=1.6, vz0=0.0)
    true_restitution = 0.72

    observed_z = simulate_bounce(torch.tensor(true_restitution), config).detach()
    result = fit_restitution(
        observed_z,
        config=config,
        initial_raw=-1.0,
        lr=0.08,
        iterations=400,
    )

    abs_error = abs(result.final_restitution - true_restitution)

    print("Synthetic restitution recovery")
    print(f"true restitution:    {true_restitution:.4f}")
    print(f"initial restitution: {result.initial_restitution:.4f}")
    print(f"final restitution:   {result.final_restitution:.4f}")
    print(f"initial loss:        {result.initial_loss:.8f}")
    print(f"final loss:          {result.final_loss:.8f}")
    print(f"absolute error:      {abs_error:.6f}")

    if abs_error > 0.03 or result.final_loss > 1e-5:
        print("restitution recovery failed tolerance checks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
