#!/usr/bin/env python
"""Fit restitution on train frames only; report MSE on held-out test frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d import BounceConfig  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402
from phys4d.trajectory_metrics import fit_restitution_on_train_frames  # noqa: E402


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sphere_bounce_m2.json",
    )
    parser.add_argument(
        "--train-end-frame",
        type=int,
        default=None,
        help="Last train frame index (default: last frame in config train_frames)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    sim = cfg["simulation"]
    scene = cfg["scene"]
    sphere = next(o for o in scene["objects"] if o["name"] == "sphere")
    ground = next(o for o in scene["objects"] if o["name"] == "ground")

    poses_path = REPO_ROOT / cfg["outputs"]["object_poses"]
    if not poses_path.is_file():
        print(f"Missing poses: {poses_path}", file=sys.stderr)
        return 1

    train_end = args.train_end_frame
    if train_end is None:
        train_end = int(sim["train_frames"][1])

    trajectory = load_object_poses_csv(poses_path)
    bounce_cfg = BounceConfig(
        steps=int(sim["num_frames"]),
        dt=float(sim["dt_s"]),
        z0=float(sphere["initial_position_m"][2]),
        vz0=float(sphere["initial_velocity_m_s"][2]),
        gravity=float(scene["gravity_m_s2"][2]),
        radius=float(sphere["radius_m"]),
        ground_z=float(ground["height_m"]),
    )

    metrics = fit_restitution_on_train_frames(
        trajectory,
        bounce_cfg=bounce_cfg,
        train_end_frame=train_end,
        z0=bounce_cfg.z0,
    )

    true_e = float(scene["true_restitution"])
    print("Physics trajectory split (pose-only, 1D bounce model)")
    print(f"  train frames: {metrics.train_frames[0]}..{metrics.train_frames[1]}")
    print(f"  test frames:  {metrics.test_frames[0]}..{metrics.test_frames[1]}")
    print(f"  true e:       {true_e:.4f}")
    print(f"  recovered e: {metrics.recovered_restitution:.4f}")
    print(f"  train MSE:    {metrics.train_mse:.6f}")
    print(f"  test MSE:     {metrics.test_mse:.6f}")
    print(f"  full MSE:     {metrics.full_mse:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
