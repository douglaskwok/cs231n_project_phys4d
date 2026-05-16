#!/usr/bin/env python
"""Phase 5: test video → perception features → rollout → warp Gaussians → metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.camera_project import load_gs_cameras, mask_coverage_fraction  # noqa: E402
from phys4d.gaussian_ply import (  # noqa: E402
    estimate_pb_to_gs_z_scale,
    filter_sphere_gaussians_by_opacity,
    load_gaussian_ply,
    save_gaussian_ply,
    select_sphere_gaussians,
    warp_gaussians_to_frame,
)
from phys4d.object_state import ObjectState
from phys4d.perception.extract import extract_scene_perception  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402
from phys4d.trajectory_eval import metrics_from_states  # noqa: E402
from phys4d.visual_dynamics.rollout import load_model_checkpoint, rollout_scene  # noqa: E402
from phys4d.visual_dynamics.scenes import SceneRecord  # noqa: E402


def _pred_trajectory_from_rollout(rows: np.ndarray, start_frame: int, dt_s: float):
    from phys4d.poses import ObjectPoseTrajectory, ObjectPose

    poses = []
    for i, row in enumerate(rows):
        f = start_frame + i
        poses.append(
            ObjectPose(
                frame=f,
                time_s=f * dt_s,
                position=row[:3],
                quat_xyzw=row[3:7],
                linear_velocity=row[7:10],
            )
        )
    return ObjectPoseTrajectory(poses=poses)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/visual_dynamics.json")
    parser.add_argument("--scene-dir", type=Path, default=REPO_ROOT / "outputs/sphere_bounce_m2")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs/visual_dynamics/visual_dynamics.pt",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=REPO_ROOT / "gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply",
    )
    parser.add_argument("--gs-cameras", type=Path, default=REPO_ROOT / "gs_sphere_bounce/cameras.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/visual_dynamics_pipeline")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tcfg = cfg["temporal"]
    scene_dir = args.scene_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_scene_perception(
        scene_dir,
        dt_s=float(tcfg["dt_s"]),
        visual_frame=int(cfg["perception"]["visual_feature_frame"]),
        image_size=int(cfg["perception"]["image_size"]),
        max_views=int(cfg["perception"]["max_views"]),
        device=args.device,
    )

    import torch

    device = torch.device(args.device)
    model, _ = load_model_checkpoint(args.checkpoint, device)
    scene = SceneRecord(
        scene_id=scene_dir.name,
        scene_dir=scene_dir,
        restitution=0.0,
        mass_kg=1.0,
        drop_z_m=1.5,
    )
    start = int(tcfg["train_frames"][1])
    horizon = int(tcfg["T_future"])
    pred_rows = rollout_scene(
        model,
        scene,
        start_frame=start,
        horizon=horizon,
        history=int(tcfg["history_K"]),
        dt_s=float(tcfg["dt_s"]),
        device=device,
    )
    np.save(out_dir / "predicted_states.npy", pred_rows)

    gt_traj = load_object_poses_csv(scene_dir / "object_poses.csv")
    pred_traj = _pred_trajectory_from_rollout(pred_rows, start, float(tcfg["dt_s"]))
    test_start, test_end = int(tcfg["test_frames"][0]), int(tcfg["test_frames"][1])
    traj_metrics = metrics_from_states(pred_traj, gt_traj, test_start, test_end)

    render_proxy = []
    if args.ply.is_file():
        cloud = load_gaussian_ply(args.ply)
        masks_root = scene_dir / "masks"
        gs_cams = load_gs_cameras(args.gs_cameras) if args.gs_cameras.is_file() else []
        sphere_cloud = select_sphere_gaussians(
            cloud,
            masks_root=masks_root if masks_root.is_dir() else None,
            ref_frame=0,
            gs_cameras=gs_cams if gs_cams else None,
        )
        ref_cloud = filter_sphere_gaussians_by_opacity(cloud)
        z_scale = estimate_pb_to_gs_z_scale(
            gt_traj,
            float(np.mean(ref_cloud.xyz[:, 2])),
            ref_frame=0,
            align_frame=30,
            gs_z_at_align=0.70,
        )
        warp_dir = out_dir / "warped_plys"
        warp_dir.mkdir(exist_ok=True)
        for fi in [test_start, (test_start + test_end) // 2, test_end]:
            if fi < start or fi - start >= pred_rows.shape[0]:
                continue
            warped = warp_gaussians_to_frame(
                sphere_cloud,
                pred_traj,
                ref_frame=start,
                target_frame=fi,
                z_scale=z_scale,
            )
            ply_path = warp_dir / f"warped_frame{fi:05d}.ply"
            save_gaussian_ply(warped, ply_path)
            cov = None
            if masks_root.is_dir() and gs_cams:
                cov = mask_coverage_fraction(
                    warped.xyz, masks_root=masks_root, frame=fi, gs_cameras=gs_cams[:6]
                )
            render_proxy.append({"frame": fi, "mask_coverage": cov, "ply": str(ply_path.name)})

    report = {
        "scene_dir": str(scene_dir.relative_to(REPO_ROOT)),
        "checkpoint": str(args.checkpoint),
        "horizon": horizon,
        "trajectory_metrics_test": traj_metrics,
        "render_proxy": render_proxy,
    }
    with (out_dir / "pipeline_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
