#!/usr/bin/env python
"""Step 6 — trajectory + rendering metrics on held-out test predictions.

Example::

  python scripts/bounce/step6_eval.py \\
    --predicted outputs/bounce_pipeline/<RUN>/step4c/trajectory_predicted.csv \\
    --gt-poses dataset/.../object_poses.csv \\
    --rendered outputs/bounce_pipeline/<RUN>/step5/renders \\
    --gt-rgb-root dataset/.../rgb \\
    --out outputs/bounce_pipeline/<RUN>/step6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.metrics import run_step6  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predicted",
        type=Path,
        required=True,
        help="step4c/trajectory_predicted.csv (held-out frames only)",
    )
    parser.add_argument(
        "--gt-poses",
        type=Path,
        required=True,
        help="PyBullet object_poses.csv for trajectory GT",
    )
    parser.add_argument(
        "--rendered",
        type=Path,
        required=True,
        help="step5/renders directory (PNG per test view)",
    )
    parser.add_argument(
        "--gt-rgb-root",
        type=Path,
        required=True,
        help="Scene rgb/ root (camXX/frameXXXXX.png)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output step6/ directory")
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="FPS for finite-difference velocity/acceleration R²",
    )
    parser.add_argument(
        "--trajectory-only",
        action="store_true",
        help="Skip PSNR/SSIM render metrics (trajectory JSON + plot only)",
    )
    parser.add_argument(
        "--extracted-traj",
        type=Path,
        default=None,
        help="step4a/trajectory_smoothed.csv for fused GT+extracted+predicted plot",
    )
    parser.add_argument(
        "--refit-metric",
        type=Path,
        default=None,
        help="step4b_metric/refit_metric.json (Umeyama scale for extracted overlay)",
    )
    args = parser.parse_args()

    split_blob = None
    if args.refit_metric is not None and args.refit_metric.is_file():
        split_blob = json.loads(args.refit_metric.read_text(encoding="utf-8")).get("split")

    result = run_step6(
        predicted_csv=args.predicted,
        gt_poses_csv=args.gt_poses,
        rendered_dir=args.rendered,
        gt_rgb_root=args.gt_rgb_root,
        out_dir=args.out,
        fps=args.fps,
        skip_render_metrics=args.trajectory_only,
        extracted_traj_csv=args.extracted_traj,
        refit_metric_json=args.refit_metric,
        split=split_blob,
    )

    print(f"Step 6 complete → {result.out_dir}")
    print(f"  metrics: {result.metrics_json}")
    print(f"  plot:    {result.plot_png}")
    if result.fused_plot_png is not None:
        print(f"  fused:   {result.fused_plot_png}")
    print(f"  pos_rmse_m: {result.pos_rmse_m:.6f}")
    print(f"  vel_r2:     {result.vel_r2:.4f}")
    print(f"  acc_r2:     {result.acc_r2:.4f}")
    if result.render_psnr_db is not None:
        print(f"  render_psnr_db: {result.render_psnr_db:.3f}")
    if result.render_ssim is not None:
        print(f"  render_ssim:    {result.render_ssim:.4f}")
    if result.render_mae is not None:
        print(f"  render_mae:     {result.render_mae:.6f}")

    blob = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    passed = blob.get("milestone_pass") or {}
    for key, ok in passed.items():
        tag = "PASS" if ok else "MISS"
        print(f"  milestone {key}: {tag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
