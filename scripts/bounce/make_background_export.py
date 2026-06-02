#!/usr/bin/env python
"""Build a static background 3DGS training dataset (ball masked out) for a Wu scene.

Step 5 of the bounce pipeline composites the moving object Gaussians over a STATIC
background 3DGS. That background asset (``background.ply``) does not exist yet; this
script produces the multi-view, ball-removed image set + camera transforms that the
graphdeco 3DGS trainer (``modal_app.py --train-bg``) consumes to fit it.

Approach
--------
The background is static while the ball moves, so any single pixel is unoccluded in
most frames. We therefore export MANY (camera, frame) pairs with the ball region
removed (inverted ball mask); multi-view + multi-frame coverage lets 3DGS fill the
pixels that any one frame happens to occlude.

Output (Blender/NeRF-style, matching the proven graphdeco exporter):
  <out>/images/cam<ID>_<FRAME>.png   ball pixels replaced by the background color
  <out>/transforms_train.json        per-frame intrinsics + Blender-convention c2w
  <out>/transforms_test.json         one frame per camera (graphdeco wants a test set)
  <out>/export_meta.json             provenance / counts

Poses come from ``cameras.json``'s GL ``view_matrix_row_major`` inverted into a
camera-to-world matrix (OpenGL/Blender basis: x right, y up, z back) — the exact
convention graphdeco's Blender reader expects.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def view_matrix_to_c2w(view_col_major_16: list[float]) -> np.ndarray:
    """Invert a column-major GL view matrix into a camera-to-world matrix."""

    v = np.array(view_col_major_16, dtype=np.float64).reshape(4, 4, order="F")
    return np.linalg.inv(v)


def camera_angle_x_from_vertical_fov(fov_deg: float, width: int, height: int) -> float:
    """Horizontal FoV (rad) implied by a vertical FoV over a width x height frame."""

    v = math.radians(float(fov_deg))
    return float(2.0 * math.atan(math.tan(v / 2.0) * (float(width) / float(height))))


def _intrinsics(fov_deg: float, width: int, height: int) -> dict[str, float | int]:
    cam_angle_x = camera_angle_x_from_vertical_fov(fov_deg, width, height)
    return {
        "w": width,
        "h": height,
        "fl_x": float(width / (2.0 * math.tan(cam_angle_x / 2.0))),
        "fl_y": float(height / (2.0 * math.tan(math.radians(fov_deg) / 2.0))),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "camera_angle_x": cam_angle_x,
    }


def _read_rgb(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    arr = imageio.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    mask = np.asarray(imageio.imread(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def _apply_background_mask(rgb: np.ndarray, ball_mask: np.ndarray, background: str) -> np.ndarray:
    """Keep background pixels; replace the ball region with the background color."""

    bg_value = 255 if background == "white" else 0
    out = np.full_like(rgb, bg_value)
    keep = ~ball_mask
    out[keep] = rgb[keep]
    return out


def _resolve_cameras(arg: str | None, available: list[int], cfg_train: list[int]) -> list[int]:
    if arg is None:
        return [int(c) for c in (cfg_train or available)]
    if arg.strip().lower() == "all":
        return list(available)
    return [int(tok) for tok in arg.split(",") if tok.strip()]


def export_background_dynerf(
    *,
    scene_dir: Path,
    out_dir: Path,
    background: str,
    cameras_arg: str | None,
    frame_start: int,
    frame_end: int | None,
    frame_stride: int,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    cfg = json.loads((scene_dir / "config.json").read_text(encoding="utf-8"))
    cams = json.loads((scene_dir / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    cam_by_index = {int(c["index"]): c for c in cams}

    sim = cfg["simulation"]
    cam_cfg = cfg["cameras"]
    num_frames = int(sim["num_frames"])
    duration_s = float(sim.get("duration_s", 0.0))
    fps = (num_frames - 1) / duration_s if duration_s > 0 else 1.0 / float(sim.get("dt_s", 1.0 / 60.0))
    width, height = (int(v) for v in cam_cfg["image_size"])

    rgb_root = scene_dir / "rgb"
    masks_root = scene_dir / "masks"
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"Missing rgb/: {rgb_root}")
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Missing masks/: {masks_root}")

    available = sorted(cam_by_index)
    cam_indices = _resolve_cameras(cameras_arg, available, cam_cfg.get("train_cameras", []))
    missing = [c for c in cam_indices if c not in cam_by_index]
    if missing:
        raise ValueError(f"Requested cameras not in cameras.json: {missing}")

    hi = num_frames - 1 if frame_end is None else int(frame_end)
    frames = list(range(int(frame_start), hi + 1, max(1, int(frame_stride))))
    if not frames:
        raise ValueError("Empty frame selection; check --frame-start/--frame-end/--frame-stride.")

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    train_entries: list[dict] = []
    test_entries: list[dict] = []
    empty_masks = 0

    for cam_id in cam_indices:
        rec = cam_by_index[cam_id]
        c2w = view_matrix_to_c2w(rec["view_matrix_row_major"])
        intr = _intrinsics(float(rec.get("fov_deg", 60.0)), width, height)
        for fi, frame_idx in enumerate(frames):
            tag = f"{frame_idx:05d}"
            rgb_path = rgb_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
            mask_path = masks_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
            if not rgb_path.is_file():
                raise FileNotFoundError(f"Missing RGB frame: {rgb_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask frame: {mask_path}")

            rgb = _read_rgb(rgb_path)
            ball = _read_mask(mask_path)
            if not ball.any():
                empty_masks += 1
            masked = _apply_background_mask(rgb, ball, background)

            stem = f"images/cam{cam_id:02d}_{frame_idx:05d}"
            imageio.imwrite(out_dir / f"{stem}.png", masked)
            entry = {
                **intr,
                "file_path": stem,
                "transform_matrix": c2w.tolist(),
                "time": float(frame_idx) / float(fps),
            }
            train_entries.append(entry)
            # One frame per camera goes into the (optional) test split.
            if fi == 0:
                test_entries.append(entry)

    shared = _intrinsics(float(cam_by_index[cam_indices[0]].get("fov_deg", 60.0)), width, height)
    (out_dir / "transforms_train.json").write_text(
        json.dumps({**shared, "frames": train_entries}, indent=2), encoding="utf-8"
    )
    (out_dir / "transforms_test.json").write_text(
        json.dumps({**shared, "frames": test_entries}, indent=2), encoding="utf-8"
    )

    meta = {
        "format": "dynerf_background_static",
        "source_scene": str(scene_dir),
        "background": background,
        "fps": float(fps),
        "cameras": cam_indices,
        "frames": frames,
        "frame_stride": int(frame_stride),
        "num_train_views": len(train_entries),
        "num_test_views": len(test_entries),
        "empty_ball_masks": empty_masks,
        "image_size": [width, height],
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Background dataset folder to create.")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument(
        "--cameras",
        type=str,
        default="all",
        help="'all', a comma list like '0,3,10', or omit to use config train_cameras.",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None, help="Inclusive; default last frame.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Subsample frames to limit dataset size (background is static).",
    )
    args = parser.parse_args()

    meta = export_background_dynerf(
        scene_dir=args.scene_dir.resolve(),
        out_dir=args.out.resolve(),
        background=args.background,
        cameras_arg=args.cameras,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote background DyNeRF dataset -> {args.out.resolve()}")
    print(f"  views: {meta['num_train_views']} train / {meta['num_test_views']} test")
    print(f"  empty ball masks: {meta['empty_ball_masks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
