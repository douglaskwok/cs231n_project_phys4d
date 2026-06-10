#!/usr/bin/env python
"""Synthesize a minimal DyNeRF-style export for the Wu ball scene.

Phases 3 and 4 of the bounce pipeline expect a DyNeRF export folder providing the
train/test frame split (``export_meta.json``) and held-out camera intrinsics +
camera-to-world poses (``transforms_test.json``). The Wu scene only ships a
PyBullet-style ``cameras.json`` (eye / look_at / up + GL matrices) and a
``config.json`` split, so we translate those into the format the existing phase
scripts already consume. No re-training or re-export is required.

We emit camera-to-world matrices in an OpenCV-style basis (x right, y down,
z forward). Phase 4 auto-calibrates the exact sign convention against the GT ball
mask, so only a consistent, metric-correct pose is required here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _look_at_c2w(eye: np.ndarray, target: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    """Build an OpenCV camera-to-world matrix from eye / look-at / up.

    OpenCV camera basis: +z points from the camera toward the scene, +x points
    right, +y points down. Translation is the camera center in world space.
    """

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up_world)
    right = right / (np.linalg.norm(right) + 1e-12)
    down = np.cross(forward, right)
    down = down / (np.linalg.norm(down) + 1e-12)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def _focal_from_fov(fov_deg: float, length_px: int) -> float:
    """Pinhole focal length (px) for a vertical/horizontal FoV over ``length_px``."""

    return 0.5 * float(length_px) / math.tan(0.5 * math.radians(float(fov_deg)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Export folder to create.")
    parser.add_argument(
        "--train-end",
        type=int,
        default=None,
        help="Last train frame (inclusive). Default: scene config train_frames[1].",
    )
    parser.add_argument(
        "--test-start",
        type=int,
        default=None,
        help="First held-out frame. Default: train-end + 1.",
    )
    parser.add_argument(
        "--test-end",
        type=int,
        default=None,
        help="Last held-out frame (inclusive). Default: num_frames - 1.",
    )
    parser.add_argument(
        "--test-cameras",
        type=str,
        default=None,
        help=(
            "Override which cameras to render/evaluate on the held-out frames. "
            "Use 'all' for every camera in cameras.json, or a comma-separated list "
            "like '0,3,10,11'. Default: scene config cameras.test_cameras. "
            "Note: this only changes which viewpoints visualize the (camera-"
            "independent) physics prediction; it does not affect the fit."
        ),
    )
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    cfg = json.loads((scene_dir / "config.json").read_text(encoding="utf-8"))
    cams = json.loads((scene_dir / "cameras.json").read_text(encoding="utf-8"))["cameras"]

    sim = cfg["simulation"]
    num_frames = int(sim["num_frames"])
    duration_s = float(sim["duration_s"])
    fps = (num_frames - 1) / duration_s if duration_s > 0 else 120.0

    train_end = args.train_end if args.train_end is not None else int(sim["train_frames"][1])
    test_start = args.test_start if args.test_start is not None else train_end + 1
    test_end = args.test_end if args.test_end is not None else num_frames - 1

    train_frames = list(range(0, train_end + 1))
    test_frames = list(range(test_start, test_end + 1))

    cam_cfg = cfg["cameras"]
    # Resolve the held-out camera set. By default we honor the scene config split,
    # but --test-cameras lets us render the prediction from more (or all) POVs.
    available_cam_indices = sorted(int(c["index"]) for c in cams)
    if args.test_cameras is None:
        test_cam_indices = [int(c) for c in cam_cfg.get("test_cameras", [])]
    elif args.test_cameras.strip().lower() == "all":
        test_cam_indices = list(available_cam_indices)
    else:
        test_cam_indices = [int(tok) for tok in args.test_cameras.split(",") if tok.strip()]
    missing = [c for c in test_cam_indices if c not in available_cam_indices]
    if missing:
        raise ValueError(f"Requested test cameras not in cameras.json: {missing}")
    img_w, img_h = (int(v) for v in cam_cfg["image_size"])

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    export_meta = {
        "backend": "wu_4dgaussians_synth_export",
        "fps": float(fps),
        "num_frames": num_frames,
        "train_timestamps": train_frames,
        "test_timestamps": test_frames,
        "train_cameras": [int(c) for c in cam_cfg.get("train_cameras", [])],
        "test_cameras": test_cam_indices,
        "image_size": [img_w, img_h],
        "source_scene": str(scene_dir),
    }
    (out_dir / "export_meta.json").write_text(
        json.dumps(export_meta, indent=2) + "\n", encoding="utf-8"
    )

    # frame_map.json bridges PyBullet frame indices <-> kept training order and
    # carries the train/test split that step4a/step4c scripts read.
    # Times use the same frame/fps mapping the Wu run was trained on.
    def _frame_rows(frame_list: list[int]) -> list[dict]:
        return [
            {
                "kept_index": i,
                "original_frame": int(f),
                "original_time_s": float(f) / float(fps),
            }
            for i, f in enumerate(frame_list)
        ]

    frame_map = {
        "fps": float(fps),
        "num_frames": num_frames,
        "train": _frame_rows(train_frames),
        "test": _frame_rows(test_frames),
    }
    (out_dir / "frame_map.json").write_text(
        json.dumps(frame_map, indent=2) + "\n", encoding="utf-8"
    )

    cam_by_index = {int(c["index"]): c for c in cams}

    def _build_transforms(cam_indices: list[int], frames: list[int]) -> dict:
        # Cameras in this rig do NOT all share the same FoV, so we attach per-frame
        # intrinsics (fl_x/fl_y/cx/cy) derived from each camera's own FoV. The
        # top-level block keeps the first camera's intrinsics for backward
        # compatibility with consumers that only read a single shared model.
        frame_records = []
        for cam_idx in cam_indices:
            cam = cam_by_index[cam_idx]
            eye = np.asarray(cam["eye_m"], dtype=np.float64)
            target = np.asarray(cam["look_at_m"], dtype=np.float64)
            up = np.asarray(cam["up"], dtype=np.float64)
            c2w = _look_at_c2w(eye, target, up)
            # Square pixels: the GL projection matrix confirms fx == fy, so a single
            # focal derived from the vertical FoV over image height applies to both.
            fl = _focal_from_fov(cam["fov_deg"], img_h)
            for frame in frames:
                frame_records.append(
                    {
                        "file_path": f"images/cam{cam_idx:02d}_{frame:05d}",
                        "time": float(frame) / float(fps),
                        "transform_matrix": c2w.tolist(),
                        "fl_x": fl,
                        "fl_y": fl,
                        "cx": img_w / 2.0,
                        "cy": img_h / 2.0,
                    }
                )
        # Top-level shared intrinsics fall back to the first listed camera.
        ref_cam = cam_by_index[cam_indices[0]]
        fl_y = _focal_from_fov(ref_cam["fov_deg"], img_h)
        fl_x = fl_y
        return {
            "w": img_w,
            "h": img_h,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": img_w / 2.0,
            "cy": img_h / 2.0,
            "frames": frame_records,
        }

    # transforms_test.json carries per-frame intrinsics (cameras differ in FoV),
    # plus a shared top-level block for backward compatibility.
    transforms_test = _build_transforms(test_cam_indices, test_frames)
    (out_dir / "transforms_test.json").write_text(
        json.dumps(transforms_test, indent=2) + "\n", encoding="utf-8"
    )

    # transforms_train.json kept for completeness / debugging (not strictly needed).
    transforms_train = _build_transforms(
        [int(c) for c in cam_cfg.get("train_cameras", [])], train_frames
    )
    (out_dir / "transforms_train.json").write_text(
        json.dumps(transforms_train, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote export -> {out_dir}")
    print(f"  train frames: {train_frames[0]}..{train_frames[-1]} ({len(train_frames)})")
    print(f"  test frames:  {test_frames[0]}..{test_frames[-1]} ({len(test_frames)})")
    print(f"  test cameras: {test_cam_indices}")
    print(f"  fps={fps:.3f}  image={img_w}x{img_h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
