#!/usr/bin/env python
"""Render fudan 4D Gaussian Splatting checkpoints to PNG frames + MP4.

**dataset** mode: every view in ``transforms_*.json`` at its ``time`` → motion + multi-cam.

**orbit** mode: horizontal orbit around the mean camera position using dataset intrinsics;
time sweeps linearly (``--orbit-time-start`` … ``--orbit-time-end``).

Modal::

    modal run modal_app.py --render-4d
    modal run modal_app.py --render-4d-orbit

Local (CUDA + fudan clone)::

    export FOURDGS_ROOT=/path/to/4d-gaussian-splatting
    python 4dgs/scripts/render_4dgs_trajectory.py --mode dataset ...
    python 4dgs/scripts/render_4dgs_trajectory.py --mode orbit ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fourdgs_common as fc


def _flip_blender_then_w2c_rt(c2w_blender):
    """Match ``readCamerasFromTransforms`` (fudan): flip Y/Z columns, then w2c → stored R,T."""
    import numpy as np

    c2w = np.array(c2w_blender, dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    r = np.transpose(w2c[:3, :3])
    t = w2c[:3, 3]
    return r, t


def _build_orbit_camera_infos(
    dataset: Path,
    orbit_frames: int,
    time_start: float,
    time_end: float,
    ref_frame_index: int,
) -> list:
    """Synthetic ``CameraInfo`` list; no PNGs — use with ``dataloader`` LP override (meta-only load)."""
    import numpy as np
    from scene.dataset_readers import CameraInfo

    tf = dataset / "transforms_train.json"
    if not tf.is_file():
        tf = dataset / "transforms_test.json"
    if not tf.is_file():
        raise FileNotFoundError(f"No transforms_train/test.json under {dataset}")

    contents = json.loads(tf.read_text(encoding="utf-8"))
    frames = contents["frames"]
    if not frames:
        raise ValueError("No frames in transforms JSON")

    origins = []
    for fr in frames:
        m = np.array(fr["transform_matrix"], dtype=np.float64)[:3, 3]
        origins.append(m)
    target = np.mean(np.stack(origins, axis=0), axis=0)

    ref_idx = min(max(0, ref_frame_index), len(frames) - 1)
    ref_b = np.array(frames[ref_idx]["transform_matrix"], dtype=np.float64)
    w = int(contents["w"])
    h = int(contents["h"])
    fl_x = float(contents["fl_x"])
    fl_y = float(contents["fl_y"])
    cx = float(contents["cx"])
    cy = float(contents["cy"])

    infos: list = []
    thetas = [2.0 * math.pi * k / max(1, orbit_frames) for k in range(orbit_frames)]
    for i, theta in enumerate(thetas):
        c, s = math.cos(theta), math.sin(theta)
        ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
        new_b = np.eye(4, dtype=np.float64)
        new_b[:3, :3] = ry @ ref_b[:3, :3]
        new_b[:3, 3] = target + ry @ (ref_b[:3, 3] - target)

        r, t = _flip_blender_then_w2c_rt(new_b)
        u = i / max(1, orbit_frames - 1)
        ts = time_start + u * (time_end - time_start)

        infos.append(
            CameraInfo(
                uid=i,
                R=r,
                T=t,
                FovY=-1.0,
                FovX=-1.0,
                image=np.empty(0),
                depth=None,
                image_path="",
                image_name=f"orbit_{i:04d}",
                width=w,
                height=h,
                timestamp=float(ts),
                fl_x=fl_x,
                fl_y=fl_y,
                cx=cx,
                cy=cy,
            )
        )
    return infos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dataset", "orbit"), default="dataset")
    parser.add_argument("--fourd-root", default=os.environ.get("FOURDGS_ROOT", "/opt/4dgs"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--mp4", type=str, default="trajectory.mp4")
    parser.add_argument("--dry-run-max", type=int, default=0)

    parser.add_argument("--orbit-frames", type=int, default=180)
    parser.add_argument("--orbit-ref-frame", type=int, default=0)
    parser.add_argument("--orbit-time-start", type=float, default=None)
    parser.add_argument("--orbit-time-end", type=float, default=None)

    cli = parser.parse_args()

    fourd = Path(cli.fourd_root).resolve()
    os.environ["FOURDGS_ROOT"] = str(fourd)
    if str(fourd) not in sys.path:
        sys.path.insert(0, str(fourd))

    import torch
    from gaussian_renderer import render
    from scene.dataset_readers import readCamerasFromTransforms
    from torchvision.utils import save_image
    from tqdm import tqdm
    from utils.camera_utils import cameraList_from_camInfos

    args_ns, lp, _, pp = fc.merge_config_yaml(Path(cli.config).resolve())
    ckpt_path = cli.checkpoint.resolve()
    dataset = cli.dataset.resolve()
    out_dir = cli.out_dir.resolve()

    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    if cli.mode == "dataset":
        all_infos: list = []
        for tf_name in ("transforms_train.json", "transforms_test.json"):
            if not (dataset / tf_name).is_file():
                continue
            infos = readCamerasFromTransforms(
                str(dataset),
                tf_name,
                lp.white_background,
                lp.extension,
                time_duration=None,
                frame_ratio=lp.frame_ratio,
                dataloader=lp.dataloader,
            )
            all_infos.extend(infos)

        all_infos.sort(
            key=lambda ci: (float(ci.timestamp), fc.cam_sort_key(ci.image_name)),
        )

        if cli.dry_run_max > 0:
            all_infos = all_infos[: cli.dry_run_max]

        if not all_infos:
            print("No camera viewpoints.", file=sys.stderr)
            return 1

        lp_cam = lp
    else:
        td = list(args_ns.time_duration)
        t0 = cli.orbit_time_start if cli.orbit_time_start is not None else td[0]
        t1 = cli.orbit_time_end if cli.orbit_time_end is not None else td[1]
        all_infos = _build_orbit_camera_infos(
            dataset,
            cli.orbit_frames,
            t0,
            t1,
            cli.orbit_ref_frame,
        )
        lp_cam = SimpleNamespace(**{**vars(lp), "dataloader": True})

    camera_stack = cameraList_from_camInfos(all_infos, resolution_scale=1.0, args=lp_cam)

    bg = torch.tensor(
        [1.0, 1.0, 1.0] if lp.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    gaussians = fc.instantiate_gaussian_model(args_ns, lp, pp)

    blob = torch.load(str(ckpt_path), map_location="cuda")
    if not isinstance(blob, (tuple, list)) or len(blob) != 2:
        print("Bad checkpoint tuple.", file=sys.stderr)
        return 1

    gaussians.restore(blob[0], None)

    png_paths: list[str] = []
    with torch.no_grad():
        for idx, viewpoint in enumerate(tqdm(camera_stack, desc="Rendering")):
            v = viewpoint.cuda()
            rp = render(v, gaussians, pp, bg)
            img = torch.clamp(rp["render"], 0.0, 1.0)
            stem = f"{idx:06d}_{v.image_name.replace('/', '_').replace('.', '_')}"
            pfp = out_dir / f"{stem}.png"
            save_image(img, str(pfp))
            png_paths.append(str(pfp))

    mp4_path = str(out_dir / cli.mp4)
    try:
        if _ffmpeg_available():
            _write_mp4_ffmpeg(paths=png_paths, mp4_out=mp4_path, fps=cli.fps)
            enc = "libx264 (QuickTime-friendly)"
        else:
            _write_mp4_opencv(png_paths, mp4_path, cli.fps)
            enc = "mp4v (OpenCV)"
        print(f"MP4: {mp4_path}  [{enc}]")
    except Exception as exc:
        print(
            f"MP4 failed ({exc}). Re-encode: ffmpeg -y -framerate {cli.fps} "
            "-i out/%06d.png -c:v libx264 -pix_fmt yuv420p -movflags +faststart "
            "out_fixed.mp4  (adapt -i pattern to your PNG names)\n"
            "Or install ffmpeg and rerun.",
            file=sys.stderr,
        )

    meta = {
        "num_frames": len(png_paths),
        "fps": cli.fps,
        "checkpoint": str(ckpt_path),
        "mode": cli.mode,
    }
    (out_dir / "render_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(png_paths)} PNGs → {out_dir}")
    return 0


def _ffmpeg_available() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None


def _write_mp4_ffmpeg(*, paths: list[str], mp4_out: str, fps: float) -> None:
    """H.264 + yuv420p + faststart — plays in QuickTime / Chrome / VLC."""
    import os
    import shlex
    import subprocess
    import tempfile

    if not paths:
        raise ValueError("no frames")
    td = 1.0 / max(float(fps), 1e-6)
    # ffconcat: per-frame duration; repeat last file (required by concat demuxer).
    lines = ["ffconcat version 1.0", ""]
    for p in paths:
        ap = shlex.quote(os.path.abspath(p))
        lines.append(f"file {ap}")
        lines.append(f"duration {td:.8f}")
        lines.append("")
    lines.append(f"file {shlex.quote(os.path.abspath(paths[-1]))}")
    lines.append("")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        script = f.name
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                script,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                mp4_out,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:] if proc.stderr else "ffmpeg failed")
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def _write_mp4_opencv(paths: list[str], mp4: str, fps: float) -> None:
    import cv2

    img0 = cv2.imread(paths[0])
    if img0 is None:
        raise RuntimeError(f"opencv could not read {paths[0]}")
    h, w = img0.shape[:2]
    writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter")
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        writer.write(im)
    writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
