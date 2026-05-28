#!/usr/bin/env python
"""Phase 2 — fit simple bounce physics parameters from phase-1 trajectory CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.physics import run_phase2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj",
        type=Path,
        required=True,
        help="Phase-1 smoothed trajectory CSV (frame,t_sec,x,y,z).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output phase2 directory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS for metadata; defaults to infer from t_sec spacing.",
    )
    parser.add_argument(
        "--fix-gravity",
        action="store_true",
        help="Fix gravity to -9.81 m/s^2 instead of fitting it.",
    )
    parser.add_argument(
        "--gt-poses",
        type=Path,
        default=None,
        help="Optional GT object_poses.csv for translation alignment before fitting.",
    )
    parser.add_argument(
        "--align-translation-to-gt",
        action="store_true",
        help="Align trajectory to GT by a single xyz translation before fitting.",
    )
    parser.add_argument(
        "--robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
        help="Robust least-squares loss to stabilize noisy trajectory fits.",
    )
    parser.add_argument(
        "--robust-f-scale",
        type=float,
        default=0.05,
        help="Scale parameter for robust loss.",
    )
    args = parser.parse_args()

    result = run_phase2(
        trajectory_csv=args.traj,
        out_dir=args.out,
        fps=args.fps,
        fix_gravity=args.fix_gravity,
        gt_poses_csv=args.gt_poses,
        align_translation_to_gt=args.align_translation_to_gt,
        robust_loss=args.robust_loss,
        robust_f_scale=args.robust_f_scale,
    )
    print(f"Phase 2 complete -> {result.out_dir}")
    print(f"  params: {result.params_json}")
    print(f"  plot:   {result.fit_plot_png}")
    print(f"  fit_mse_m2: {result.fit_mse_m2:.8f}")
    print(f"  bounce_count: {result.bounce_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
