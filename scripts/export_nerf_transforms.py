#!/usr/bin/env python
"""Write a NeRF-style ``transforms_train.json`` for multi-view RGB (one timestep).

This is the usual next step before 3D Gaussian Splatting or NeRF-style training:
you get calibrated pinhole cameras + image paths. Official ``gaussian-splatting``
expects COLMAP; many forks ingest ``transforms*.json``. You can also use
``ns-process-data`` or manual COLMAP export from these matrices.

Coordinate note: we invert PyBullet's OpenGL view matrix (column-major 4x4) to
obtain camera-to-world. If a downstream tool looks mirrored, apply a fixed GL
axis flip (documented in many NeRF / GS conversion guides).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _view_list_to_c2w(view_col_major_16: list[float]) -> np.ndarray:
    """Invert PyBullet 4x4 view matrix (column-major) -> camera-to-world."""
    v = np.array(view_col_major_16, dtype=np.float64).reshape(4, 4, order="F")
    return np.linalg.inv(v)


def _camera_angle_x_from_vertical_fov(fov_deg: float, width: int, height: int) -> float:
    """NeRF ``camera_angle_x``: horizontal FOV in radians (Blender convention)."""
    v = math.radians(float(fov_deg))
    h = 2.0 * math.atan(math.tan(v / 2.0) * (float(width) / float(height)))
    return float(h)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sphere_bounce_m2.json",
    )
    parser.add_argument(
        "--cameras-json",
        type=Path,
        default=None,
        help="Default: <repo>/<config outputs>/cameras.json",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Which timestep PNG to use for all cameras (static multi-view slice)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "sphere_bounce_m2" / "transforms_train_static.json",
        help="Written JSON path (parent dirs created)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    cam_path = args.cameras_json or (REPO_ROOT / cfg["outputs"]["camera_poses"])
    cam_path = cam_path.resolve()
    if not cam_path.is_file():
        print(f"Missing cameras file: {cam_path}", file=sys.stderr)
        print("Run: python scripts/generate_sphere_bounce_dataset.py", file=sys.stderr)
        return 1

    cams_cfg = cfg["cameras"]
    w, h = int(cams_cfg["image_size"][0]), int(cams_cfg["image_size"][1])
    fov_deg = float(cams_cfg.get("fov_deg", 60.0))

    with cam_path.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    frame_tag = f"{args.frame:05d}"
    frames: list[dict] = []
    for rec in blob["cameras"]:
        ci = int(rec["index"])
        c2w = _view_list_to_c2w(rec["view_matrix_row_major"])
        rel = f"rgb/cam{ci:02d}/frame{frame_tag}.png"
        frames.append(
            {
                "file_path": rel,
                "transform_matrix": c2w.tolist(),
            }
        )

    out = {
        "camera_angle_x": _camera_angle_x_from_vertical_fov(fov_deg, w, h),
        "W": w,
        "H": h,
        "fl_x": float(w / (2.0 * math.tan(_camera_angle_x_from_vertical_fov(fov_deg, w, h) / 2.0))),
        "fl_y": float(h / (2.0 * math.tan(math.radians(fov_deg) / 2.0))),
        "cx": w / 2.0,
        "cy": h / 2.0,
        "frames": frames,
        "sphere_bounce_m2_note": "Paths are relative to the directory containing this JSON (place file under outputs/sphere_bounce_m2/).",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.output} with {len(frames)} views for frame {frame_tag}")
    print("Place this file next to the rgb/ folder (same parent as rgb/cam00/...).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
