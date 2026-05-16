#!/usr/bin/env python
"""Report pose / physics metrics on the **temporal** train/test split (frames 0–59 vs 60–89).

Distinct from 4DGS DyNeRF eval (held-out cameras). Use this split for extrapolation claims.
"""

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
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs/sphere_bounce_m2/extrapolation_report.json",
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

    report = {
        "split_type": "temporal",
        "train_frames": list(metrics.train_frames),
        "test_frames": list(metrics.test_frames),
        "recovered_restitution": metrics.recovered_restitution,
        "true_restitution": float(scene["true_restitution"]),
        "z_mse": {
            "physics_train": metrics.train_mse,
            "physics_test": metrics.test_mse,
        },
        "poses_csv": str(poses_path.relative_to(REPO_ROOT)),
        "note": "Image-based extrapolation (L_render) is the target metric; this is pose-only physics baseline.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
