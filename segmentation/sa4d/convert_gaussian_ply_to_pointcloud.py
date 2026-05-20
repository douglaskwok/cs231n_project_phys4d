#!/usr/bin/env python
"""Convert a Gaussian Splatting PLY into a generic XYZ/RGB point-cloud PLY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BINDING_DIR = REPO_ROOT / "segmentation" / "01_mask_guided_gaussian_binding"
if str(BINDING_DIR) not in sys.path:
    sys.path.insert(0, str(BINDING_DIR))

from select_ball_gaussians import load_gaussian_ply  # noqa: E402

SH_C0 = 0.28209479177387814


def _rgb_from_gaussian_fields(vertices: np.ndarray) -> np.ndarray:
    names = set(vertices.dtype.names or [])
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        rgb = np.stack(
            [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
            axis=1,
        )
        rgb = np.clip(rgb * SH_C0 + 0.5, 0.0, 1.0)
        return (rgb * 255.0).round().astype(np.uint8)
    if {"red", "green", "blue"}.issubset(names):
        return np.stack(
            [vertices["red"], vertices["green"], vertices["blue"]],
            axis=1,
        ).astype(np.uint8)
    return np.full((len(vertices), 3), 255, dtype=np.uint8)


def convert_gaussian_ply_to_pointcloud(src: Path, dst: Path) -> dict:
    cloud = load_gaussian_ply(src)
    xyz = cloud.xyz.astype(np.float32)
    rgb = _rgb_from_gaussian_fields(cloud.vertices)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(
                f"{float(p[0])} {float(p[1])} {float(p[2])} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

    return {
        "input": str(src),
        "output": str(dst),
        "points": int(len(xyz)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    meta = convert_gaussian_ply_to_pointcloud(args.input.resolve(), args.output.resolve())
    print(f"Wrote {meta['output']}")
    print(f"  points={meta['points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
