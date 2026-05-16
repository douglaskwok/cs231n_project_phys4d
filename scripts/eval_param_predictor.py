#!/usr/bin/env python
"""Evaluate param predictor on val/test splits + trajectory rollout metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.param_ident.dataset import PREDICT_NAMES, ParamIdDataset, SceneSample, load_manifest  # noqa: E402
from phys4d.param_ident.inference import load_checkpoint, load_scene_clip, predict_params  # noqa: E402
from phys4d.param_ident.mlp_baseline import PoseHistoryMLP  # noqa: E402
from phys4d.param_ident.model import MultiViewParamPredictor  # noqa: E402
from phys4d.param_ident.pose_dataset import PoseHistoryDataset  # noqa: E402
from phys4d.sim_rollout import rollout_from_config, write_pose_csv  # noqa: E402
from phys4d.trajectory_eval import metrics_from_csv  # noqa: E402


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _param_mse_dict(pred: dict[str, float], gt: dict[str, float]) -> dict[str, float]:
    return {k: (pred[k] - gt[k]) ** 2 for k in PREDICT_NAMES}


@torch.no_grad()
def _eval_cnn_scene(
    model: MultiViewParamPredictor,
    scene: SceneSample,
    device: torch.device,
    max_views: int,
) -> dict[str, float]:
    clip = load_scene_clip(scene.scene_dir, max_views=max_views)
    pred = predict_params(model, clip, device=device)
    gt = {PREDICT_NAMES[i]: float(scene.param_vector[i]) for i in range(len(PREDICT_NAMES))}
    return _param_mse_dict(pred, gt)


@torch.no_grad()
def _eval_mlp_scene(
    model: PoseHistoryMLP,
    scene: SceneSample,
    device: torch.device,
) -> dict[str, float]:
    ds = PoseHistoryDataset([scene])
    x, y = ds[0]
    pred_t = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
    gt = {PREDICT_NAMES[i]: float(y[i]) for i in range(len(PREDICT_NAMES))}
    pred = {PREDICT_NAMES[i]: float(pred_t[i]) for i in range(len(PREDICT_NAMES))}
    return _param_mse_dict(pred, gt)


def _eval_rollout(
    scene: SceneSample,
    pred_params: dict[str, float],
    cfg: dict,
    out_dir: Path,
) -> dict[str, float]:
    pred_csv = out_dir / f"{scene.scene_id}_poses_pred.csv"
    write_pose_csv(rollout_from_config(cfg, pred_params), pred_csv)
    gt_csv = scene.scene_dir / "object_poses.csv"
    test = cfg["simulation"]["test_frames"]
    return metrics_from_csv(pred_csv, gt_csv, int(test[0]), int(test[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--split", choices=("val", "test", "train"), default="test")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/sphere_bounce_m2.json")
    parser.add_argument("--max-views", type=int, default=10)
    parser.add_argument("--out-json", type=Path, default=REPO_ROOT / "outputs/param_predictor/eval_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    scenes = load_manifest(args.manifest, args.split, data_root=args.data_root)
    if not scenes:
        raise SystemExit(f"No scenes in split {args.split}")

    if args.model == "cnn":
        model, _ = load_checkpoint(args.checkpoint, device=device)
    else:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = PoseHistoryMLP().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

    cfg = _load_config(args.config)
    per_scene: list[dict] = []
    for scene in scenes:
        if args.model == "cnn":
            clip = load_scene_clip(scene.scene_dir, max_views=args.max_views)
            pred = predict_params(model, clip, device=device)
            pmse = _param_mse_dict(
                pred,
                {PREDICT_NAMES[i]: float(scene.param_vector[i]) for i in range(len(PREDICT_NAMES))},
            )
        else:
            pmse = _eval_mlp_scene(model, scene, device)
            x, _ = PoseHistoryDataset([scene])[0]
            pred_t = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
            pred = {PREDICT_NAMES[i]: float(pred_t[i]) for i in range(len(PREDICT_NAMES))}

        traj = _eval_rollout(scene, pred, cfg, args.out_json.parent / "rollout_cache")
        per_scene.append(
            {
                "scene_id": scene.scene_id,
                "param_mse": pmse,
                "param_rmse": {k: v**0.5 for k, v in pmse.items()},
                "trajectory": traj,
            }
        )

    def _mean(key_path: str) -> float:
        vals = []
        for row in per_scene:
            d = row
            for part in key_path.split("."):
                d = d[part]
            vals.append(float(d))
        return float(sum(vals) / len(vals)) if vals else float("nan")

    summary = {
        "split": args.split,
        "model": args.model,
        "num_scenes": len(per_scene),
        "mean_param_mse": {n: _mean(f"param_mse.{n}") for n in PREDICT_NAMES},
        "mean_z_mse_test": _mean("trajectory.z_mse"),
        "mean_z_r2_test": _mean("trajectory.z_r2"),
        "mean_speed_r2_test": _mean("trajectory.speed_r2"),
        "mean_vel_mse_test": _mean("trajectory.vel_mse"),
    }
    report = {"summary": summary, "per_scene": per_scene}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
