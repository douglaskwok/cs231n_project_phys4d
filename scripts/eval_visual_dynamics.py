#!/usr/bin/env python
"""Phase 6 evaluation: horizon curve, velocity R², render proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.trajectory_eval import metrics_from_states  # noqa: E402
from phys4d.visual_dynamics.rollout import load_model_checkpoint, rollout_scene  # noqa: E402
from phys4d.visual_dynamics.scenes import load_manifest_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/visual_dynamics.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs/visual_dynamics/visual_dynamics.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs/sphere_bounce_batch/dataset_manifest.json",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--horizons", default="1,5,10,30")
    parser.add_argument("--out-json", type=Path, default=REPO_ROOT / "outputs/visual_dynamics/eval_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tcfg = cfg["temporal"]
    splits = load_manifest_splits(args.manifest, REPO_ROOT / "outputs")
    scenes = splits.get(args.split, [])
    device = torch.device(args.device)
    model, _ = load_model_checkpoint(args.checkpoint, device)

    from phys4d.poses import ObjectPose, ObjectPoseTrajectory

    horizon_list = [int(x) for x in args.horizons.split(",") if x.strip()]
    start = int(tcfg["train_frames"][1])
    test_lo, test_hi = int(tcfg["test_frames"][0]), int(tcfg["test_frames"][1])
    dt_s = float(tcfg["dt_s"])

    per_scene = []
    horizon_means: dict[str, float] = {}

    for scene in scenes:
        scene_row = {"scene_id": scene.scene_id, "horizons": {}}
        from phys4d.poses import load_object_poses_csv

        gt_traj = load_object_poses_csv(
            scene.scene_dir / "object_poses.csv"
        )

        for h in horizon_list:
            pred = rollout_scene(
                model,
                scene,
                start_frame=start,
                horizon=h,
                history=int(tcfg["history_K"]),
                dt_s=dt_s,
                device=device,
            )
            pred_poses = [
                ObjectPose(
                    frame=start + i,
                    time_s=(start + i) * dt_s,
                    position=pred[i, :3],
                    quat_xyzw=pred[i, 3:7],
                    linear_velocity=pred[i, 7:10],
                )
                for i in range(pred.shape[0])
            ]
            pred_traj = ObjectPoseTrajectory(poses=pred_poses)
            m = metrics_from_states(pred_traj, gt_traj, test_lo, min(test_hi, start + h))
            scene_row["horizons"][str(h)] = m
        per_scene.append(scene_row)

    for h in horizon_list:
        key = f"h{h}_z_mse"
        vals = [row["horizons"][str(h)]["z_mse"] for row in per_scene if str(h) in row["horizons"]]
        horizon_means[key] = float(sum(vals) / len(vals)) if vals else float("nan")

    report = {"horizon_means": horizon_means, "per_scene": per_scene}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["horizon_means"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
