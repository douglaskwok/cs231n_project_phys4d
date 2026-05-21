#!/usr/bin/env python
"""Flashy demo MP4: multi-view RGB + GT vs neural rollout overlays + z(t) plot."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phys4d.camera_project import project_pybullet_points  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402
from phys4d.trajectory_eval import metrics_from_states  # noqa: E402
from phys4d.visual_dynamics.rollout import load_model_checkpoint, rollout_scene  # noqa: E402
from phys4d.visual_dynamics.scenes import SceneRecord  # noqa: E402


def _load_cameras(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cameras" in data:
        return list(data["cameras"])
    return list(data)


def _draw_ball(draw, pos, cam, *, color_rgb: tuple[int, int, int], radius: int = 9) -> None:
    from PIL import ImageDraw

    uv, vis = project_pybullet_points(np.asarray(pos, dtype=np.float64).reshape(1, 3), cam)
    if not vis[0]:
        return
    u, v = float(uv[0, 0]), float(uv[0, 1])
    draw.ellipse(
        (u - radius, v - radius, u + radius, v + radius),
        outline=color_rgb + (255,),
        width=2,
    )
    draw.ellipse(
        (u - radius // 3, v - radius // 3, u + radius // 3, v + radius // 3),
        fill=color_rgb + (255,),
    )


def _compose_frame(
    scene_dir: Path,
    cameras: list[dict],
    cam_ids: list[int],
    frame: int,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    *,
    rollout_start: int,
    z_series_gt: np.ndarray,
    z_series_pred: np.ndarray,
    t_axis: np.ndarray,
    metrics_text: str,
) -> np.ndarray:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    tile = int(cameras[0]["image_size"][0])
    grid = 2
    mosaic = Image.new("RGB", (tile * grid, tile * grid), (12, 12, 18))
    for slot, ci in enumerate(cam_ids[: grid * grid]):
        row, col = divmod(slot, grid)
        rgb_path = scene_dir / "rgb" / f"cam{ci:02d}" / f"frame{frame:05d}.png"
        if rgb_path.is_file():
            panel = Image.open(rgb_path).convert("RGB")
            panel = panel.resize((tile, tile), Image.Resampling.BILINEAR)
        else:
            panel = Image.new("RGB", (tile, tile), (30, 30, 40))
        cam = cameras[ci]
        d = ImageDraw.Draw(panel)
        _draw_ball(d, gt_pos, cam, color_rgb=(80, 220, 100))
        _draw_ball(d, pred_pos, cam, color_rgb=(255, 90, 90))
        mosaic.paste(panel, (col * tile, row * tile))

    plot_h = 140
    plot_w = mosaic.width
    plot = Image.new("RGB", (plot_w, plot_h), (32, 32, 40))
    dp = ImageDraw.Draw(plot)
    t0, t1 = float(t_axis[0]), float(t_axis[-1])
    z_min = float(min(z_series_gt.min(), z_series_pred.min()) - 0.05)
    z_max = float(max(z_series_gt.max(), z_series_pred.max()) + 0.05)
    z_span = max(z_max - z_min, 1e-3)

    def _map(t: float, z: float) -> tuple[float, float]:
        x = (t - t0) / max(t1 - t0, 1e-6) * (plot_w - 40) + 20
        y = plot_h - 20 - (z - z_min) / z_span * (plot_h - 40)
        return x, y

    def _poly(series: np.ndarray, color: tuple[int, int, int]) -> list[tuple[float, float]]:
        return [_map(float(t_axis[i]), float(series[i])) for i in range(len(t_axis))]

    dp.line(_poly(z_series_gt, (80, 220, 100)), fill=(80, 220, 100), width=2)
    dp.line(_poly(z_series_pred, (255, 90, 90)), fill=(255, 90, 90), width=2)

    t_cur = float(t_axis[min(frame, len(t_axis) - 1)])
    cx, _ = _map(t_cur, z_min)
    dp.line([(cx, 10), (cx, plot_h - 10)], fill=(180, 180, 180), width=1)
    if rollout_start < len(t_axis):
        rx, _ = _map(float(t_axis[rollout_start]), z_min)
        dp.line([(rx, 10), (rx, plot_h - 10)], fill=(140, 140, 220), width=1)
        dp.text((rx + 4, 8), "rollout", fill=(180, 180, 255))

    dp.text((10, 6), "z (m)  green=GT  red=NN", fill=(210, 210, 210))
    dp.text((10, plot_h - 18), metrics_text, fill=(190, 190, 190))

    header = Image.new("RGB", (mosaic.width, 36), (24, 24, 30))
    dh = ImageDraw.Draw(header)
    dh.text(
        (10, 8),
        f"Visual dynamics  |  frame {frame:03d}  |  {metrics_text}",
        fill=(235, 235, 235),
    )

    out = Image.new("RGB", (mosaic.width, header.height + mosaic.height + plot.height))
    out.paste(header, (0, 0))
    out.paste(mosaic, (0, header.height))
    out.paste(plot, (0, header.height + mosaic.height))
    return np.asarray(out)


def _write_mp4(frames_dir: Path, out_mp4: Path, fps: float) -> None:
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    if shutil.which("ffmpeg"):
        try:
            import sys
            from pathlib import Path

            _4d_scripts = Path(__file__).resolve().parent.parent / "4dgs" / "scripts"
            if str(_4d_scripts) not in sys.path:
                sys.path.insert(0, str(_4d_scripts))
            from render_4dgs_trajectory import _write_mp4_ffmpeg

            _write_mp4_ffmpeg(paths=[str(p) for p in paths], mp4_out=str(out_mp4), fps=fps)
            return
        except Exception:
            pass
    import imageio.v2 as imageio

    writer = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=8)
    for p in paths:
        writer.append_data(imageio.imread(p))
    writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/visual_dynamics.json")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=REPO_ROOT / "outputs/sphere_bounce_m2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "visual_dynamics/visual_dynamics.pt",
    )
    parser.add_argument("--cameras", type=Path, default=None)
    parser.add_argument("--cams", default="0,1,2,3")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "latest/visual_dynamics_demo.mp4")
    parser.add_argument("--frames-dir", type=Path, default=REPO_ROOT / "latest/demo_frames")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    import imageio.v2 as imageio

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tcfg = cfg["temporal"]
    scene_dir = args.scene_dir.resolve()
    cam_path = args.cameras or (scene_dir / "cameras.json")
    cameras = _load_cameras(cam_path)
    cam_ids = [int(x.strip()) for x in args.cams.split(",") if x.strip()]

    import torch

    device = torch.device(args.device)
    model, _ = load_model_checkpoint(args.checkpoint.resolve(), device)
    scene = SceneRecord(
        scene_id=scene_dir.name,
        scene_dir=scene_dir,
        restitution=0.0,
        mass_kg=1.0,
        drop_z_m=1.5,
    )
    gt_traj = load_object_poses_csv(scene_dir / "object_poses.csv")
    rollout_start = int(tcfg["train_frames"][1])
    horizon = int(tcfg["test_frames"][1]) - rollout_start
    pred_rows = rollout_scene(
        model,
        scene,
        start_frame=rollout_start,
        horizon=horizon,
        history=int(tcfg["history_K"]),
        dt_s=float(tcfg["dt_s"]),
        device=device,
    )

    from phys4d.poses import ObjectPose, ObjectPoseTrajectory

    pred_poses = [
        ObjectPose(
            frame=rollout_start + i,
            time_s=(rollout_start + i) * float(tcfg["dt_s"]),
            position=pred_rows[i, :3],
            quat_xyzw=pred_rows[i, 3:7],
            linear_velocity=pred_rows[i, 7:10],
        )
        for i in range(pred_rows.shape[0])
    ]
    pred_traj = ObjectPoseTrajectory(poses=pred_poses)
    test_lo, test_hi = int(tcfg["test_frames"][0]), int(tcfg["test_frames"][1])
    m = metrics_from_states(pred_traj, gt_traj, test_lo, test_hi)
    metrics_text = f"test z_mse={m['z_mse']:.3f}  speed_r2={m['speed_r2']:.2f}"

    n_frames = len(gt_traj.poses)
    t_axis = np.array([p.time_s for p in gt_traj.poses], dtype=np.float64)
    z_gt = np.array([p.position[2] for p in gt_traj.poses], dtype=np.float64)
    z_pred = z_gt.copy()
    for p in pred_poses:
        if p.frame < len(z_pred):
            z_pred[p.frame] = p.position[2]

    pred_by_frame = {p.frame: p.position for p in pred_poses}
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(n_frames):
        gt_p = gt_traj.by_frame(fi).position
        pred_p = pred_by_frame.get(fi, gt_p)
        if fi < rollout_start:
            pred_p = gt_p
        rgb = _compose_frame(
            scene_dir,
            cameras,
            cam_ids,
            fi,
            gt_p,
            pred_p,
            rollout_start=rollout_start,
            z_series_gt=z_gt,
            z_series_pred=z_pred,
            t_axis=t_axis,
            metrics_text=metrics_text,
        )
        imageio.imwrite(args.frames_dir / f"frame_{fi:05d}.png", rgb)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_mp4(args.frames_dir, args.out.resolve(), args.fps)

    h264 = args.out.with_name(args.out.stem + "_h264.mp4")
    if shutil.which("ffmpeg") and args.out.is_file():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(args.out),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(h264),
            ],
            check=False,
            capture_output=True,
        )

    meta = {
        "scene_dir": str(scene_dir),
        "checkpoint": str(args.checkpoint),
        "metrics_test": m,
        "mp4": str(args.out),
        "h264_mp4": str(h264) if h264.is_file() else None,
        "frames_dir": str(args.frames_dir),
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {args.out}")
    if h264.is_file():
        print(f"Wrote {h264}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
