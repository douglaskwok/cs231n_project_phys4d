#!/usr/bin/env python
"""Diagnose Phase-1 x/y centroid drift by comparing centroid estimators.

The opacity-weighted MEAN of a deforming, non-rigid Gaussian cloud (plus any
low-opacity floaters) can wander in x/y even when the true sphere center does
not. We evaluate the deformed means once, then compare estimators against the
(near-constant) GT x/y to see which removes the spurious wiggle.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_phase1_wu import (  # noqa: E402
    _hidden_args_from_cfg,
    _import_wu_deform_network,
    _read_ply_means_opacity,
)

REPO = _HERE.parents[1]
SCENE = REPO / "wu_debug_ball_huge_120fps_0p5s" / "scene_0000_e0p90_a0p0"
MODEL = (
    SCENE
    / "4dgs_wu/wu_debug_ball_huge_120fps_0p5s_e0p90_a0p0_wu_object"
    / "wu4dgs_wu_debug_ball_huge_120fps_0p5s_e0p90_a0p0_wu_object"
)


def gt_xy():
    rows = list(csv.DictReader(open(SCENE / "object_poses.csv")))
    x = np.array([float(r["x_m"]) for r in rows])
    y = np.array([float(r["y_m"]) for r in rows])
    return x, y


def main() -> int:
    import torch

    iter_dir = MODEL / "point_cloud" / "iteration_5000"
    xyz, opa_logit = _read_ply_means_opacity(iter_dir / "point_cloud.ply")
    w = 1.0 / (1.0 + np.exp(-opa_logit))
    n = xyz.shape[0]

    args = _hidden_args_from_cfg(MODEL / "cfg_args")
    deform_network = _import_wu_deform_network(REPO / "third_party" / "4DGaussians")
    net = deform_network(args)
    net.load_state_dict(torch.load(str(iter_dir / "deformation.pth"), map_location="cpu"), strict=False)
    net.eval()

    xyz_t = torch.from_numpy(xyz.astype(np.float32))
    sc = torch.zeros((n, 3), dtype=torch.float32)
    ro = torch.zeros((n, 4), dtype=torch.float32)
    op = torch.zeros((n, 1), dtype=torch.float32)

    num_frames = 61
    means = np.zeros((num_frames, n, 3), dtype=np.float64)
    with torch.no_grad():
        for f in range(num_frames):
            t = torch.full((n, 1), f / (num_frames - 1), dtype=torch.float32)
            m, _, _, _, _ = net(xyz_t, sc, ro, op, None, t)
            means[f] = m.cpu().numpy()

    gx, gy = gt_xy()

    def estimate(method: str) -> np.ndarray:
        out = np.zeros((num_frames, 3))
        for f in range(num_frames):
            p = means[f]
            wf = np.clip(w, 0, None)
            if method == "mean":
                c = (p * wf[:, None]).sum(0) / wf.sum()
            elif method == "median":
                c = np.median(p, axis=0)
            elif method == "opacity_top50":
                thr = np.quantile(wf, 0.5)
                k = wf >= thr
                c = (p[k] * wf[k, None]).sum(0) / wf[k].sum()
            elif method == "trim_iqr":
                # robust center, then drop Gaussians far from it, recompute w-mean
                c0 = np.median(p, axis=0)
                d = np.linalg.norm(p - c0, axis=1)
                k = d <= np.quantile(d, 0.7)
                c = (p[k] * wf[k, None]).sum(0) / wf[k].sum()
            elif method == "opacity_top50_trim":
                thr = np.quantile(wf, 0.5)
                k0 = wf >= thr
                c0 = np.median(p[k0], axis=0)
                d = np.linalg.norm(p - c0, axis=1)
                k = k0 & (d <= np.quantile(d[k0], 0.7))
                c = (p[k] * wf[k, None]).sum(0) / wf[k].sum()
            else:
                raise ValueError(method)
            out[f] = c
        return out

    print(f"N gaussians = {n}")
    print(f"GT x range {gx.min():.4f}..{gx.max():.4f} (span {gx.max()-gx.min():.4f}), "
          f"GT y range {gy.min():.4f}..{gy.max():.4f} (span {gy.max()-gy.min():.4f})")
    print(f"{'method':22s} {'x_span':>8s} {'y_span':>8s} {'x_std':>8s} {'y_std':>8s}")
    for method in ("mean", "opacity_top50", "median", "trim_iqr", "opacity_top50_trim"):
        c = estimate(method)
        xs, ys = c[:, 0], c[:, 1]
        print(f"{method:22s} {xs.max()-xs.min():8.4f} {ys.max()-ys.min():8.4f} "
              f"{xs.std():8.4f} {ys.std():8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
