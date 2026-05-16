#!/usr/bin/env python
"""Phase 4 E2E: video → CNN params → PyBullet rollout → warp 3DGS → mask coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d.camera_project import load_gs_cameras, mask_coverage_fraction  # noqa: E402
from phys4d.gaussian_ply import (  # noqa: E402
    estimate_pb_to_gs_z_scale,
    filter_sphere_gaussians_by_masks,
    filter_sphere_gaussians_by_opacity,
    load_gaussian_ply,
    save_gaussian_ply,
    warp_gaussians_to_frame,
)
from phys4d.param_ident.dataset import PREDICT_NAMES  # noqa: E402
from phys4d.param_ident.inference import predict_scene  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402
from phys4d.sim_rollout import rollout_from_config, write_pose_csv  # noqa: E402


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _param_errors(pred: dict[str, float], gt: dict[str, float] | None) -> dict[str, float] | None:
    if gt is None:
        return None
    return {k: float((pred[k] - gt[k]) ** 2) for k in PREDICT_NAMES}


def _trajectory_z_mse(pred_csv: Path, gt_csv: Path, frame_start: int, frame_end: int) -> float:
    pred = load_object_poses_csv(pred_csv)
    gt = load_object_poses_csv(gt_csv)
    errs = []
    for f in range(frame_start, frame_end + 1):
        errs.append((pred.by_frame(f).position[2] - gt.by_frame(f).position[2]) ** 2)
    return float(np.mean(errs)) if errs else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=REPO_ROOT / "outputs/sphere_bounce_m2",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/sphere_bounce_m2.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs/param_predictor/param_predictor_best.pt",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=REPO_ROOT / "gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply",
    )
    parser.add_argument(
        "--gs-cameras",
        type=Path,
        default=REPO_ROOT / "gs_sphere_bounce/cameras.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/param_id_pipeline",
    )
    parser.add_argument("--ref-frame", type=int, default=0)
    parser.add_argument("--train-end-frame", type=int, default=59)
    parser.add_argument(
        "--eval-frames",
        type=str,
        default="60,70,80",
        help="Comma-separated frames for warp + mask eval (test horizon)",
    )
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.is_file():
        print(f"Missing checkpoint: {args.checkpoint}", file=sys.stderr)
        print("Train first: python scripts/train_param_predictor.py", file=sys.stderr)
        return 1
    if not args.ply.is_file():
        print(f"Missing 3DGS PLY: {args.ply}", file=sys.stderr)
        print("Train 3DGS: modal run modal_app.py --upload && modal run modal_app.py --train", file=sys.stderr)
        return 1

    cfg = _load_config(args.config.resolve())
    infer = predict_scene(scene_dir, args.checkpoint, device=args.device)
    pred_params = infer["predicted"]
    with (out_dir / "predicted_params.json").open("w", encoding="utf-8") as f:
        json.dump(infer, f, indent=2)

    pred_poses = out_dir / "object_poses_predicted.csv"
    rows = rollout_from_config(cfg, pred_params)
    write_pose_csv(rows, pred_poses)

    gt_poses = scene_dir / "object_poses.csv"
    if not gt_poses.is_file():
        gt_poses = REPO_ROOT / cfg["outputs"]["object_poses"]

    sim = cfg["simulation"]
    test_frames = [int(sim["test_frames"][0]), int(sim["test_frames"][1])]
    traj_mse = None
    if gt_poses.is_file():
        traj_mse = _trajectory_z_mse(pred_poses, gt_poses, test_frames[0], test_frames[1])

    cloud = load_gaussian_ply(args.ply)
    masks_root = scene_dir / "masks"
    gs_cams = load_gs_cameras(args.gs_cameras) if args.gs_cameras.is_file() else []
    if masks_root.is_dir() and gs_cams:
        sphere_cloud = filter_sphere_gaussians_by_masks(
            cloud,
            masks_root,
            args.ref_frame,
            min_camera_hits=1,
            gs_cameras=gs_cams[: min(6, len(gs_cams))],
        )
    else:
        sphere_cloud = filter_sphere_gaussians_by_opacity(cloud)

    gt_traj = load_object_poses_csv(gt_poses) if gt_poses.is_file() else None
    pred_traj = load_object_poses_csv(pred_poses)
    ref_cloud = filter_sphere_gaussians_by_opacity(cloud)
    gs_z_ref = float(np.mean(ref_cloud.xyz[:, 2]))
    z_scale = estimate_pb_to_gs_z_scale(
        pred_traj,
        gs_z_ref,
        ref_frame=args.ref_frame,
        align_frame=30,
        gs_z_at_align=0.70,
    )

    eval_frame_list = [int(x.strip()) for x in args.eval_frames.split(",") if x.strip()]
    warp_dir = out_dir / "warped_plys"
    warp_dir.mkdir(parents=True, exist_ok=True)
    render_proxy: list[dict] = []

    for frame in eval_frame_list:
        warped = warp_gaussians_to_frame(
            sphere_cloud,
            pred_traj,
            ref_frame=args.ref_frame,
            target_frame=frame,
            z_scale=z_scale,
        )
        ply_path = warp_dir / f"warped_frame{frame:05d}.ply"
        save_gaussian_ply(warped, ply_path)

        cov_pred = None
        cov_gt = None
        if masks_root.is_dir() and gs_cams:
            cov_pred = mask_coverage_fraction(
                warped.xyz,
                masks_root=masks_root,
                frame=frame,
                gs_cameras=gs_cams[:6],
            )
            if gt_traj is not None:
                warped_gt = warp_gaussians_to_frame(
                    sphere_cloud,
                    gt_traj,
                    ref_frame=args.ref_frame,
                    target_frame=frame,
                    z_scale=z_scale,
                )
                cov_gt = mask_coverage_fraction(
                    warped_gt.xyz,
                    masks_root=masks_root,
                    frame=frame,
                    gs_cameras=gs_cams[:6],
                )
        render_proxy.append(
            {
                "frame": frame,
                "ply": str(ply_path.relative_to(REPO_ROOT)),
                "mask_coverage_predicted_traj": cov_pred,
                "mask_coverage_gt_traj": cov_gt,
            }
        )

    report = {
        "scene_dir": str(scene_dir.relative_to(REPO_ROOT)),
        "predicted_params": pred_params,
        "ground_truth_params": infer.get("ground_truth"),
        "param_mse": _param_errors(pred_params, infer.get("ground_truth")),
        "trajectory_z_mse_test_half": traj_mse,
        "z_scale_pb_to_gs": z_scale,
        "predicted_poses_csv": str(pred_poses.relative_to(REPO_ROOT)),
        "render_proxy": render_proxy,
        "note": "render_proxy uses mask coverage on warped 3DGS; full RGB PSNR needs graphdeco render on Modal.",
    }
    report_path = out_dir / "pipeline_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
