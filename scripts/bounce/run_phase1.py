#!/usr/bin/env python
"""Phase 1 — extract opacity-weighted 4DGS ball trajectory on the train window.

Example (from repo root)::

  export FOURDGS_ROOT=/path/to/4d-gaussian-splatting

  python scripts/bounce/run_phase1.py \\
    --checkpoint dataset/.../4dgs/ball_drop_e0p90_a0p0_object/chkpnt30000.pth \\
    --config 4dgs/configs/room_physics_4dgs_5p0s.yaml \\
    --dynerf-export 4dgs/experiments/object_only/runs/ball_drop_e0p90_a0p0_object_vis6_px50_all12 \\
    --gt-poses dataset/.../scene_0004_e0p90_a0p0/object_poses.csv \\
    --out outputs/bounce_pipeline/ball_drop_e0p90_a0p0_object/phase1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_FOURDGS_SCRIPTS = _REPO_ROOT / "4dgs" / "scripts"
for _p in (_SRC, _FOURDGS_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phys4d.bounce.extract import run_phase1  # noqa: E402


def _default_config_for_export(export_dir: Path) -> Path | None:
    """Pick room_physics YAML from export duration when --config is omitted."""

    meta_path = export_dir / "export_meta.json"
    if not meta_path.is_file():
        return None
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    t0, t1 = meta.get("time_duration_suggested") or meta.get("effective_time_range_s") or [0.0, 5.0]
    duration = float(t1) - float(t0)
    cfg_dir = _REPO_ROOT / "4dgs" / "configs"
    if duration <= 2.7:
        name = "room_physics_4dgs_2p6s.yaml"
    elif duration <= 4.1:
        name = "room_physics_4dgs_4p0s.yaml"
    else:
        name = "room_physics_4dgs_5p0s.yaml"
    candidate = cfg_dir / name
    return candidate if candidate.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="chkpnt30000.pth (or quick 1000)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="4DGS YAML used at train time (default: infer from export_meta duration)",
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
    parser.add_argument("--out", type=Path, required=True, help="Output phase1/ directory")
    parser.add_argument(
        "--fourd-root",
        type=Path,
        default=Path(os.environ.get("FOURDGS_ROOT", "/opt/4dgs")),
        help="fudan-zvg/4d-gaussian-splatting clone",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device for Gaussian restore (CUDA strongly recommended)",
    )
    parser.add_argument("--savgol-window", type=int, default=5)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    parser.add_argument(
        "--marginal-threshold",
        type=float,
        default=0.05,
        help="Drop Gaussians with marginal_t below this at each timestamp",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also extract test timestamps from frame_map (for debugging plots)",
    )
    args = parser.parse_args()

    config = args.config
    if config is None:
        config = _default_config_for_export(args.dynerf_export.resolve())
        if config is None:
            print(
                "Pass --config (4DGS YAML from training); could not infer from export_meta.",
                file=sys.stderr,
            )
            return 2
        print(f"Using inferred config: {config}")

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print(
                "CUDA not available; use --device cpu or run on a GPU machine / Modal.",
                file=sys.stderr,
            )
            return 1

    result = run_phase1(
        checkpoint=args.checkpoint,
        config=config,
        dynerf_export=args.dynerf_export,
        out_dir=args.out,
        gt_poses=args.gt_poses,
        fourd_root=args.fourd_root,
        device=args.device,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        marginal_threshold=args.marginal_threshold,
        include_test_timestamps=args.include_test,
    )

    print(f"Wrote {result.num_frames} frames → {result.out_dir}")
    print(f"  raw:      {result.raw_csv}")
    print(f"  smoothed: {result.smoothed_csv}")
    print(f"  plot:     {result.plot_png}")
    if result.train_mse_m2 is not None:
        print(f"  train MSE vs GT: {result.train_mse_m2:.6f} m²")
    if result.train_mse_aligned_m2 is not None:
        print(f"  train MSE vs GT (Procrustes-aligned): {result.train_mse_aligned_m2:.6f} m²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
