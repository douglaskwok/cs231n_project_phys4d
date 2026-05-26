#!/usr/bin/env python
"""Export a fudan 4DGS ``chkpnt*.pth`` to a SuperSplat-friendly PLY.

The fudan trainer saves only ``chkpnt{iter}.pth`` (not ``point_cloud/.../point_cloud.ply``).
Run on Modal after training::

  modal run modal_app.py --export-4d-ply

Or locally if you have the 4DGS CUDA env and checkpoint path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from plyfile import PlyData, PlyElement
except ModuleNotFoundError:
    PlyData = None
    PlyElement = None


def _write_ply(
    path: Path,
    *,
    xyz: np.ndarray,
    f_dc: np.ndarray,
    f_rest: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    t: np.ndarray | None = None,
    scale_t: np.ndarray | None = None,
    rotation_r: np.ndarray | None = None,
) -> None:
    """Write 3DGS-style PLY (+ optional 4D time fields)."""
    n = xyz.shape[0]
    normals = np.zeros_like(xyz, dtype=np.float32)

    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    cols: list[np.ndarray] = [xyz, normals]

    for i in range(f_dc.shape[1]):
        attrs.append(f"f_dc_{i}")
    cols.append(f_dc.astype(np.float32))

    for i in range(f_rest.shape[1]):
        attrs.append(f"f_rest_{i}")
    cols.append(f_rest.astype(np.float32))

    attrs.append("opacity")
    cols.append(opacity.astype(np.float32))

    for i in range(scale.shape[1]):
        attrs.append(f"scale_{i}")
    cols.append(scale.astype(np.float32))

    for i in range(rotation.shape[1]):
        attrs.append(f"rot_{i}")
    cols.append(rotation.astype(np.float32))

    if t is not None:
        attrs.append("t")
        cols.append(t.reshape(n, 1).astype(np.float32))
    if scale_t is not None:
        for i in range(scale_t.shape[1]):
            attrs.append(f"scale_t_{i}")
        cols.append(scale_t.astype(np.float32))
    if rotation_r is not None:
        for i in range(rotation_r.shape[1]):
            attrs.append(f"rot_r_{i}")
        cols.append(rotation_r.astype(np.float32))

    dtype = [(a, "f4") for a in attrs]
    stacked = np.concatenate(cols, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if PlyData is not None and PlyElement is not None:
        elements = np.empty(n, dtype=dtype)
        elements[:] = list(map(tuple, stacked))
        PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))
        return

    elements = np.empty(n, dtype=dtype)
    elements[:] = list(map(tuple, stacked))
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for attr in attrs:
        header.append(f"property float {attr}")
    header.append("end_header")
    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        elements.tofile(f)


def export_checkpoint_to_ply(checkpoint: Path, output: Path) -> dict:
    """Load ``chkpnt*.pth`` tensors and write PLY."""
    checkpoint = checkpoint.resolve()
    blob = torch.load(checkpoint, map_location="cpu")
    if not isinstance(blob, (tuple, list)) or len(blob) != 2:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint}")

    model_params, iteration = blob
    if len(model_params) == 14:
        gaussian_dim = 3
    elif len(model_params) >= 19:
        gaussian_dim = 4
    else:
        raise ValueError(f"Unknown capture length {len(model_params)} in {checkpoint}")

    xyz_t = model_params[1]
    f_dc_t = model_params[2]
    f_rest_t = model_params[3]
    scaling_t = model_params[4]
    rotation_t = model_params[5]
    opacity_t = model_params[6]

    xyz = xyz_t.detach().cpu().numpy().astype(np.float32)
    f_dc = f_dc_t.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy().astype(np.float32)
    f_rest = f_rest_t.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy().astype(np.float32)
    opacity = opacity_t.detach().cpu().numpy().astype(np.float32)
    scale = scaling_t.detach().cpu().numpy().astype(np.float32)
    rotation = rotation_t.detach().cpu().numpy().astype(np.float32)

    t_arr = scale_t_arr = rot_r_arr = None
    if gaussian_dim == 4:
        _t, _scaling_t, _rotation_r = model_params[13], model_params[14], model_params[15]
        t_arr = _t.detach().cpu().numpy().astype(np.float32)
        scale_t_arr = _scaling_t.detach().cpu().numpy().astype(np.float32)
        rot_r_arr = _rotation_r.detach().cpu().numpy().astype(np.float32)

    _write_ply(
        output,
        xyz=xyz,
        f_dc=f_dc,
        f_rest=f_rest,
        opacity=opacity,
        scale=scale,
        rotation=rotation,
        t=t_arr,
        scale_t=scale_t_arr,
        rotation_r=rot_r_arr,
    )

    return {
        "iteration": int(iteration),
        "gaussian_dim": gaussian_dim,
        "num_points": int(xyz.shape[0]),
        "output": str(output.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("4dgs_sphere_bounce/chkpnt_best.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("4dgs_sphere_bounce/point_cloud/exported/point_cloud.ply"),
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        print(f"Missing checkpoint: {args.checkpoint}", file=sys.stderr)
        return 1
    meta = export_checkpoint_to_ply(args.checkpoint, args.output)
    print(f"Wrote {meta['output']}")
    print(f"  iteration={meta['iteration']}  points={meta['num_points']}  dim={meta['gaussian_dim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
