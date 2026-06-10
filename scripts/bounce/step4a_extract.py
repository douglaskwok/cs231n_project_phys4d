#!/usr/bin/env python
"""Step 4a — extract opacity-weighted Wu 4DGS ball trajectory on the train window.

Example (from repo root)::

  python scripts/bounce/step4a_extract.py \\
    --canonical dataset/.../4dgs_wu/ball_drop_e0p90_a0p0_object/point_cloud/iteration_30000/point_cloud.ply \\
    --deform    dataset/.../4dgs_wu/ball_drop_e0p90_a0p0_object/point_cloud/iteration_30000/deformation.pth \\
    --cfg-args  dataset/.../4dgs_wu/ball_drop_e0p90_a0p0_object/cfg_args \\
    --dynerf-export 4dgs/experiments/object_only/runs/ball_drop_e0p90_a0p0_object \\
    --gt-poses dataset/.../scene_0004/object_poses.csv \\
    --out outputs/bounce_pipeline/ball_drop_e0p90_a0p0_object/step4a
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.extract import run_step4a  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        required=True,
        help="point_cloud/iteration_30000/point_cloud.ply",
    )
    parser.add_argument(
        "--deform",
        type=Path,
        required=True,
        help="deformation.pth or its parent iteration_* directory",
    )
    parser.add_argument(
        "--cfg-args",
        type=Path,
        required=True,
        help="Wu training cfg_args next to the checkpoint",
    )
    parser.add_argument(
        "--dynerf-export",
        type=Path,
        required=True,
        help="Object-only DyNeRF folder (frame_map.json, transforms_train.json)",
    )
    parser.add_argument(
        "--gt-poses",
        type=Path,
        default=None,
        help="PyBullet object_poses.csv for overlay / MSE (optional)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output step4a/ directory")
    parser.add_argument(
        "--wu-root",
        type=Path,
        default=Path(os.environ.get("WU_4DGS_ROOT", _REPO_ROOT / "third_party" / "4DGaussians")),
        help="hustvl/4DGaussians clone",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device for deformation forward passes",
    )
    parser.add_argument("--savgol-window", type=int, default=9)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    parser.add_argument(
        "--no-robust-centroid",
        action="store_true",
        help="Disable iterative spatial-outlier rejection (use plain opacity-weighted mean)",
    )
    parser.add_argument(
        "--robust-k-mad",
        type=float,
        default=3.0,
        help="Inlier cutoff = median + k*MAD of distance to running centroid",
    )
    parser.add_argument("--robust-iter", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print(
                "CUDA not available; use --device cpu or run on a GPU machine.",
                file=sys.stderr,
            )
            return 1

    result = run_step4a(
        canonical_ply=args.canonical,
        deform_path=args.deform,
        cfg_args_path=args.cfg_args,
        dynerf_export=args.dynerf_export,
        out_dir=args.out,
        gt_poses=args.gt_poses,
        wu_root=args.wu_root,
        device=args.device,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        robust_centroid=not args.no_robust_centroid,
        robust_k_mad=args.robust_k_mad,
        robust_iter=args.robust_iter,
    )

    print(f"Wrote {result.num_frames} frames → {result.out_dir}")
    print(f"  raw:      {result.raw_csv}")
    print(f"  smoothed: {result.smoothed_csv}")
    print(f"  plot:     {result.plot_png}")
    if result.train_mse_m2 is not None:
        print(f"  train MSE vs GT: {result.train_mse_m2:.6f} m²")
    if result.train_mse_aligned_m2 is not None:
        print(f"  train MSE vs GT (Procrustes): {result.train_mse_aligned_m2:.6f} m²")
        if result.train_mse_aligned_m2 > 0.05:
            print(
                "  WARNING: aligned MSE > 0.05 m² (pipeline acceptance); "
                "check time normalization and opacity getter.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
