#!/usr/bin/env python
"""Step 4a (collision) — extract a Wu 4DGS trajectory per object.

Mirrors ``scripts/bounce/step4a_extract.py`` but runs the single-object extractor
once per object and writes per-object outputs under ``<out>/obj{k}/``. Pass one
value per object (comma-separated) for the per-object flags; scalar flags apply
to every object.

Example (two objects)::

  python scripts/collision/step4a_extract.py \\
    --canonical objA/point_cloud.ply,objB/point_cloud.ply \\
    --deform    objA/deformation.pth,objB/deformation.pth \\
    --cfg-args  objA/cfg_args,objB/cfg_args \\
    --dynerf-export runs/objA,runs/objB \\
    --gt-poses scene/object_poses_obj0.csv,scene/object_poses_obj1.csv \\
    --out outputs/collision_pipeline/<RUN>/step4a
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


def _split_paths(value: str) -> list[Path]:
    return [Path(v.strip()) for v in value.split(",") if v.strip()]


def _split_opt_paths(value: str | None, n: int) -> list[Path | None]:
    if value is None:
        return [None] * n
    parts = [v.strip() for v in value.split(",")]
    if len(parts) == 1:
        return [Path(parts[0]) if parts[0] else None] * n
    return [Path(p) if p else None for p in parts]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, help="Comma-separated point_cloud.ply per object")
    parser.add_argument("--deform", required=True, help="Comma-separated deformation.pth per object")
    parser.add_argument("--cfg-args", required=True, help="Comma-separated cfg_args per object")
    parser.add_argument("--dynerf-export", required=True, help="Comma-separated DyNeRF export dir per object")
    parser.add_argument("--gt-poses", default=None, help="Comma-separated object_poses.csv per object (optional)")
    parser.add_argument("--out", type=Path, required=True, help="Output step4a/ base directory")
    parser.add_argument(
        "--wu-root",
        type=Path,
        default=Path(os.environ.get("WU_4DGS_ROOT", _REPO_ROOT / "third_party" / "4DGaussians")),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--savgol-window", type=int, default=9)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    parser.add_argument("--no-robust-centroid", action="store_true")
    parser.add_argument("--robust-k-mad", type=float, default=3.0)
    parser.add_argument("--robust-iter", type=int, default=8)
    args = parser.parse_args()

    canonical = _split_paths(args.canonical)
    deform = _split_paths(args.deform)
    cfg_args = _split_paths(args.cfg_args)
    dynerf = _split_paths(args.dynerf_export)
    n = len(canonical)
    if not (len(deform) == len(cfg_args) == len(dynerf) == n):
        print("ERROR: --canonical/--deform/--cfg-args/--dynerf-export must have equal counts", file=sys.stderr)
        return 2
    gt_poses = _split_opt_paths(args.gt_poses, n)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available; use --device cpu or run on a GPU machine.", file=sys.stderr)
            return 1

    out_base = args.out.resolve()
    for k in range(n):
        out_dir = out_base / f"obj{k}"
        print(f"=== object {k} -> {out_dir} ===", flush=True)
        result = run_step4a(
            canonical_ply=canonical[k],
            deform_path=deform[k],
            cfg_args_path=cfg_args[k],
            dynerf_export=dynerf[k],
            out_dir=out_dir,
            gt_poses=gt_poses[k],
            wu_root=args.wu_root,
            device=args.device,
            savgol_window=args.savgol_window,
            savgol_polyorder=args.savgol_polyorder,
            robust_centroid=not args.no_robust_centroid,
            robust_k_mad=args.robust_k_mad,
            robust_iter=args.robust_iter,
        )
        print(f"  obj{k}: {result.num_frames} frames → {result.out_dir}")
        print(f"    smoothed: {result.smoothed_csv}")
        if result.train_mse_aligned_m2 is not None:
            print(f"    train MSE vs GT (Procrustes): {result.train_mse_aligned_m2:.6f} m²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
