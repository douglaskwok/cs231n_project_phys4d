#!/usr/bin/env python
"""Neutralize non-finite Gaussians in a Wu/3DGS point_cloud.ply in place.

Some Wu fine checkpoints contain a tiny number of degenerate Gaussians with a
NaN/inf coordinate. Dropping them would misalign the per-Gaussian deformation
tensors (indexed by row), so instead we *neutralize* each offending Gaussian:
position -> origin, scales/opacity -> effectively zero (invisible), rotation ->
identity, all SH features -> 0. Index order is preserved, so deformation.pth and
deformation_table.pth remain valid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", type=Path, required=True, help="point_cloud.ply to sanitize in place")
    args = ap.parse_args()

    ply = PlyData.read(str(args.ply))
    v = ply["vertex"]
    names = [pr.name for pr in v.properties]
    data = v.data.copy()

    arr = np.stack([np.asarray(data[n], np.float64) for n in names], axis=1)
    bad = ~np.isfinite(arr).all(axis=1)
    n_bad = int(bad.sum())
    if n_bad == 0:
        print(f"OK: no non-finite Gaussians in {args.ply} (N={len(data)})")
        return 0

    idx = np.where(bad)[0]
    for n in names:
        col = np.asarray(data[n])
        if n in ("x", "y", "z"):
            col[idx] = 0.0
        elif n == "opacity" or n.startswith("scale"):
            col[idx] = -30.0  # sigmoid/exp -> ~0  (invisible, zero size)
        elif n == "rot_0":
            col[idx] = 1.0
        elif n.startswith("rot"):
            col[idx] = 0.0
        else:  # normals, f_dc_*, f_rest_*
            col[idx] = 0.0
        data[n] = col

    out = PlyElement.describe(data, "vertex")
    PlyData([out], text=ply.text, byte_order=ply.byte_order).write(str(args.ply))
    print(f"Sanitized {n_bad} Gaussian(s) at rows {idx.tolist()} in {args.ply} (N={len(data)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
