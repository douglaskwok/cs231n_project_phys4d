#!/usr/bin/env python
"""Extrapolation eval: warp 3DGS with predicted poses + image proxies on test frames (60–89).

Metrics per method (physics rollout, learned dynamics, GT poses):
  - z / position MSE on test frames
  - mask_coverage: fraction of warped sphere centers inside GT mask
  - crop_mae: MAE on 64x64 masked sphere crops (L_render proxy without CUDA rasterizer)

Requires a trained static 3DGS PLY (default: gs_sphere_bounce from Modal).
"""

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

from phys4d import BounceConfig  # noqa: E402
from phys4d.camera_project import (  # noqa: E402
    load_gs_cameras,
    mask_coverage_fraction,
    project_gaussian_splatting_points,
)
from phys4d.extrapolation_baselines import predict_physics_z  # noqa: E402
from phys4d.gaussian_ply import (  # noqa: E402
    estimate_pb_to_gs_z_scale,
    filter_sphere_gaussians_by_opacity,
    load_gaussian_ply,
    warp_gaussians_to_frame,
)
from phys4d.object_state import stack_state_vectors, trajectory_to_states  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402
from phys4d.visual_dynamics.dataset import SceneRecord, discover_batch_scenes  # noqa: E402
from phys4d.visual_dynamics.rollout import (  # noqa: E402
    load_visual_dynamics_model,
    rollout_learned_states,
    states_to_trajectory,
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scene_bounce_cfg(cfg: dict) -> tuple[BounceConfig, float, int]:
    sim = cfg["simulation"]
    scene = cfg["scene"]
    sphere = next(o for o in scene["objects"] if o["name"] == "sphere")
    ground = next(o for o in scene["objects"] if o["name"] == "ground")
    z0 = float(sphere["initial_position_m"][2])
    bounce_cfg = BounceConfig(
        steps=int(sim["num_frames"]),
        dt=float(sim["dt_s"]),
        z0=z0,
        vz0=float(sphere["initial_velocity_m_s"][2]),
        gravity=float(scene["gravity_m_s2"][2]),
        radius=float(sphere["radius_m"]),
        ground_z=float(ground["height_m"]),
    )
    return bounce_cfg, z0, int(sim["train_frames"][1])


def _trajectory_from_physics_z(
    gt_traj,
    pred_z: np.ndarray,
    z0: float,
) -> "ObjectPoseTrajectory":
    from phys4d.poses import ObjectPose, ObjectPoseTrajectory

    poses = []
    z_series = gt_traj.z_series()
    for i, p in enumerate(gt_traj.poses):
        z_idx = i + 1
        z_val = float(pred_z[z_idx]) if z_idx < len(pred_z) else float(p.position[2])
        pos = p.position.copy()
        pos[2] = z_val
        poses.append(
            ObjectPose(
                frame=p.frame,
                time_s=p.time_s,
                position=pos,
                quat_xyzw=p.quat_xyzw.copy(),
                linear_velocity=p.linear_velocity.copy(),
            )
        )
    return ObjectPoseTrajectory(poses=poses)


def _warp_cloud(
    cloud,
    gt_traj,
    pred_traj,
    *,
    ref_frame: int,
    target_frame: int,
    z_scale: float,
):
    """Warp canonical cloud using pose deltas from predicted trajectory."""

    # Reuse warp_gaussians_to_frame by temporarily using pred_traj frames.
    return warp_gaussians_to_frame(
        cloud,
        pred_traj,
        ref_frame=ref_frame,
        target_frame=target_frame,
        z_scale=z_scale,
    )


def _crop_mae(gt_rgb: np.ndarray, mask: np.ndarray, pred_centroid_uv: tuple[float, float]) -> float:
    """MAE between GT masked crop and same crop shifted to predicted centroid."""

    try:
        import imageio.v2 as imageio  # noqa: F401
    except ImportError:
        return float("nan")

    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return float("nan")
    cy_gt = int(0.5 * (ys.min() + ys.max()))
    cx_gt = int(0.5 * (xs.min() + xs.max()))
    cy_p, cx_p = int(pred_centroid_uv[1]), int(pred_centroid_uv[0])
    r = max(12, int(0.55 * max(ys.max() - ys.min(), xs.max() - xs.min())))
    size = 2 * r

    def _crop_at(cy: int, cx: int) -> np.ndarray:
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        patch = gt_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
        m = mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        if patch.ndim == 3 and m.ndim == 2:
            m = m[..., None]
        import torch

        t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
        out = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return out.squeeze(0).permute(1, 2, 0).numpy()

    gt_crop = _crop_at(cy_gt, cx_gt)
    shifted = _crop_at(cy_p, cx_p)
    return float(np.mean(np.abs(gt_crop - shifted)))


def _eval_method_on_scene(
    name: str,
    pred_traj,
    scene: SceneRecord,
    cfg: dict,
    cloud,
    gs_cams: dict,
    *,
    ref_frame: int,
    train_end: int,
    z_scale: float,
    device,
) -> dict:
    gt_traj = load_object_poses_csv(scene.poses_path)
    masks_root = scene.rgb_root.parent / "masks"
    test_frames = range(train_end + 1, min(train_end + 11, len(gt_traj.poses)))  # sample 10 test frames

    z_errs, covs, maes = [], [], []
    states_gt = trajectory_to_states(gt_traj)
    states_pr = trajectory_to_states(pred_traj)

    for frame in test_frames:
        warped = _warp_cloud(cloud, gt_traj, pred_traj, ref_frame=ref_frame, target_frame=frame, z_scale=z_scale)
        pts = warped.xyz
        s_gt = states_gt[frame]
        s_pr = states_pr[frame]
        z_errs.append(float((s_pr.position[2] - s_gt.position[2]) ** 2))
        frame_maes, frame_covs = [], []
        for ci, cam in gs_cams.items():
            mask_path = masks_root / f"cam{ci:02d}" / f"frame{frame:05d}.png"
            if not mask_path.is_file():
                continue
            try:
                import imageio.v2 as imageio

                mask = imageio.imread(mask_path)
                rgb = imageio.imread(scene.rgb_root / f"cam{ci:02d}" / f"frame{frame:05d}.png")
            except Exception:
                continue
            frac = mask_coverage_fraction(pts, mask_path, gs_camera=cam)
            frame_covs.append(frac)
            uv, vis = project_gaussian_splatting_points(pts.mean(axis=0, keepdims=True), cam)
            if vis[0]:
                frame_maes.append(_crop_mae(rgb, mask, (float(uv[0, 0]), float(uv[0, 1]))))
        if frame_covs:
            covs.append(float(np.mean(frame_covs)))
        if frame_maes:
            maes.append(float(np.mean(frame_maes)))

    return {
        "method": name,
        "scene_id": scene.scene_id,
        "z_mse": float(np.mean(z_errs)) if z_errs else None,
        "mask_coverage_mean": float(np.mean(covs)) if covs else None,
        "crop_mae_mean": float(np.mean(maes)) if maes else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=REPO_ROOT / "outputs/sphere_bounce_m2/config.json",
        help="Single scene config (default: original m2)",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help="If set, evaluate first N batch scenes instead of --scene-config",
    )
    parser.add_argument("--max-scenes", type=int, default=3)
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
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs/visual_dynamics/visual_dynamics.pt",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "outputs/extrapolation_render.json",
    )
    args = parser.parse_args()

    if not args.ply.is_file():
        print(f"Missing 3DGS PLY: {args.ply}", file=sys.stderr)
        print("Download: modal volume get phys4d-gs-output gs_sphere_bounce . --force", file=sys.stderr)
        return 1
    if not args.gs_cameras.is_file():
        print(f"Missing cameras: {args.gs_cameras}", file=sys.stderr)
        return 1

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gs_cams = {int(c.get("id", i)): c for i, c in enumerate(load_gs_cameras(args.gs_cameras)[:6])}
    cloud = filter_sphere_gaussians_by_opacity(load_gaussian_ply(args.ply))

    scenes: list[SceneRecord] = []
    if args.batch_root:
        batch_root = args.batch_root
        if not batch_root.is_absolute():
            batch_root = REPO_ROOT / batch_root
        scenes = discover_batch_scenes(batch_root.resolve())[: args.max_scenes]
    else:
        cfg_path = args.scene_config.resolve()
        if not cfg_path.is_file():
            cfg_path = REPO_ROOT / "configs/sphere_bounce_m2.json"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg0 = json.load(f)
        parent = REPO_ROOT / "outputs/sphere_bounce_m2"
        sphere = next(o for o in cfg0["scene"]["objects"] if o["name"] == "sphere")
        scenes = [
            SceneRecord(
                scene_id="sphere_bounce_m2",
                config_path=cfg_path,
                poses_path=parent / "object_poses.csv",
                rgb_root=parent / "rgb",
                restitution=float(cfg0["scene"]["true_restitution"]),
                mass_kg=float(sphere["mass_kg"]),
                drop_z_m=float(sphere["initial_position_m"][2]),
            )
        ]

    model = None
    history = 8
    train_end = 59
    if args.checkpoint.is_file():
        model, history, train_end = load_visual_dynamics_model(args.checkpoint, device)

    all_rows: list[dict] = []
    for scene in scenes:
        cfg_path = scene.config_path
        if not cfg_path.is_file():
            cfg_path = REPO_ROOT / "configs/sphere_bounce_m2.json"
        cfg = _load_config(cfg_path)
        bounce_cfg, z0, train_end_cfg = _scene_bounce_cfg(cfg)
        train_end = train_end_cfg
        gt_traj = load_object_poses_csv(scene.poses_path)
        gs_z_ref = float(np.mean(cloud.xyz[:, 2]))
        z_scale = estimate_pb_to_gs_z_scale(
            gt_traj, gs_z_ref, ref_frame=0, align_frame=min(30, train_end), gs_z_at_align=0.70
        )

        methods: list[tuple[str, object]] = [("gt_poses", gt_traj)]
        pred_z = predict_physics_z(gt_traj, bounce_cfg=bounce_cfg, train_end_frame=train_end, z0=z0)
        methods.append(("physics_bounce", _trajectory_from_physics_z(gt_traj, pred_z, z0)))

        if model is not None:
            learned_states = rollout_learned_states(
                model, scene, train_end_frame=train_end, history=history, device=device
            )
            methods.append(("learned_visual", states_to_trajectory(learned_states)))

        for name, pred_traj in methods:
            row = _eval_method_on_scene(
                name,
                pred_traj,
                scene,
                cfg,
                cloud,
                gs_cams,
                ref_frame=0,
                train_end=train_end,
                z_scale=z_scale,
                device=device,
            )
            all_rows.append(row)
            print(json.dumps(row))

    summary: dict[str, dict[str, float]] = {}
    for method in {r["method"] for r in all_rows}:
        sub = [r for r in all_rows if r["method"] == method]
        summary[method] = {
            k: float(np.nanmean([r[k] for r in sub if r.get(k) is not None]))
            for k in ("z_mse", "mask_coverage_mean", "crop_mae_mean")
        }

    report = {"per_scene": all_rows, "summary": summary}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
