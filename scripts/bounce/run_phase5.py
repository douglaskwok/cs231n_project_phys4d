#!/usr/bin/env python
"""Phase 5 — evaluate held-out trajectory and render outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.metrics import run_phase5  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predicted",
        type=Path,
        required=True,
        help="Phase-3 trajectory_predicted.csv path.",
    )
    parser.add_argument(
        "--gt-poses",
        type=Path,
        required=True,
        help="Scene object_poses.csv path.",
    )
    parser.add_argument(
        "--rendered",
        type=Path,
        required=True,
        help="Phase-4 renders directory containing *.png outputs.",
    )
    parser.add_argument(
        "--gt-rgb-root",
        type=Path,
        required=True,
        help="Scene rgb root with camXX/frameXXXXX.png.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output phase5 directory.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()

    result = run_phase5(
        predicted_csv=args.predicted,
        gt_poses_csv=args.gt_poses,
        rendered_dir=args.rendered,
        gt_rgb_root=args.gt_rgb_root,
        out_dir=args.out,
        fps=args.fps,
    )
    print(f"Phase 5 complete -> {result.out_dir}")
    print(f"  metrics: {result.metrics_json}")
    print(f"  plot:    {result.plot_png}")
    print(f"  pos_rmse_m:     {result.pos_rmse_m:.6f}")
    print(f"  vel_r2:         {result.vel_r2:.6f}")
    print(f"  acc_r2:         {result.acc_r2:.6f}")
    print(f"  render_psnr_db: {result.render_psnr_db:.3f}")
    print(f"  render_mae:     {result.render_mae:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
