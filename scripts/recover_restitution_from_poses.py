#!/usr/bin/env python
"""Fit restitution from PyBullet-exported sphere heights (object_poses.csv).

The toy simulator in ``phys4d.differentiable_bounce`` is a simplified 1D bounce;
PyBullet contact is richer, so recovered ``e`` may not match ``true_restitution``
exactly. This script is the bridge: optimize our model to explain the observed
``z_m`` trajectory from simulator logs.

Prepends ``z0`` from the experiment config so the length matches
``BounceConfig(steps=num_frames)`` (``num_frames + 1`` samples).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d import BounceConfig, fit_restitution  # noqa: E402


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_z_series(poses_csv: Path) -> list[float]:
    with poses_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "z_m" not in reader.fieldnames:
            raise ValueError(f"Expected z_m column, got: {reader.fieldnames}")
        return [float(row["z_m"]) for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sphere_bounce_m2.json",
        help="Experiment JSON (for dt, num_frames, z0, ground-truth restitution)",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=None,
        help="object_poses.csv (default: config outputs path)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    poses_path = args.poses or (REPO_ROOT / cfg["outputs"]["object_poses"])
    poses_path = poses_path.resolve()

    if not poses_path.is_file():
        print(f"Missing poses file: {poses_path}", file=sys.stderr)
        print("Run: python scripts/generate_sphere_bounce_dataset.py", file=sys.stderr)
        return 1

    sim = cfg["simulation"]
    scene = cfg["scene"]
    sphere = next(o for o in scene["objects"] if o["name"] == "sphere")
    dt = float(sim["dt_s"])
    num_frames = int(sim["num_frames"])
    z0 = float(sphere["initial_position_m"][2])
    vz0 = float(sphere["initial_velocity_m_s"][2])
    true_e = float(scene["true_restitution"])

    z_after_step = _read_z_series(poses_path)
    if len(z_after_step) != num_frames:
        print(
            f"Expected {num_frames} rows in poses CSV, got {len(z_after_step)}",
            file=sys.stderr,
        )
        return 1

    # Match simulate_bounce: z[0] is pre-physics, z[t+1] after each integrator step.
    observed_z = torch.tensor([z0] + z_after_step, dtype=torch.float32)

    bounce_cfg = BounceConfig(
        steps=num_frames,
        dt=dt,
        z0=z0,
        vz0=vz0,
        gravity=float(scene["gravity_m_s2"][2]),
        radius=float(sphere["radius_m"]),
        ground_z=float(next(o for o in scene["objects"] if o["name"] == "ground")["height_m"]),
    )

    result = fit_restitution(observed_z, config=bounce_cfg, initial_raw=-1.0, lr=0.08, iterations=600)

    print("Restitution fit from PyBullet pose CSV")
    print(f"poses:           {poses_path}")
    print(f"true (config):   {true_e:.4f}")
    print(f"initial guess:   {result.initial_restitution:.4f}")
    print(f"recovered:       {result.final_restitution:.4f}")
    print(f"initial MSE:     {result.initial_loss:.6f}")
    print(f"final MSE:       {result.final_loss:.6f}")
    print(f"|error|:         {abs(result.final_restitution - true_e):.4f}")
    if abs(result.final_restitution - true_e) > 0.05:
        print(
            "\nNote: Large error is expected when the 1D analytical contact model "
            "does not match PyBullet's solver. This script validates the data path; "
            "use image-based loss or a closer differentiable sim for accurate e.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
