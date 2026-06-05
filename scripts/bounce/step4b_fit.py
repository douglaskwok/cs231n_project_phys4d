#!/usr/bin/env python
"""Step 4b — fit bounce physics parameters from Step 4a smoothed trajectory CSV.

Example::

  python scripts/bounce/step4b_fit.py \\
    --traj outputs/bounce_pipeline/ball_drop_e0p90_a0p0_object/step4a/trajectory_smoothed.csv \\
    --out  outputs/bounce_pipeline/ball_drop_e0p90_a0p0_object/step4b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.physics import run_step4b  # noqa: E402


def _check_acceptance(result, *, fix_gravity: bool) -> int:
    """Print milestone3 acceptance warnings; return 0 (non-fatal)."""

    import json

    blob = json.loads(result.params_json.read_text(encoding="utf-8"))
    p = blob["params"]
    e = float(p["restitution"])
    g = float(p["gravity_z_m_s2"])
    mse = float(blob["fit_mse_m2"])
    code = 0

    if not (0.3 <= e <= 0.95):
        print(f"WARNING: restitution e={e:.3f} outside [0.3, 0.95]", file=sys.stderr)
        code = 1
    if not fix_gravity and abs(g - (-9.81)) > 0.15 * 9.81:
        print(f"WARNING: gravity g={g:.3f} outside ±15% of -9.81", file=sys.stderr)
        code = 1
    if mse >= 0.01:
        print(f"WARNING: fit_mse={mse:.6f} m² >= 0.01 m² target", file=sys.stderr)
        code = 1
    if result.bounce_count < 1:
        print("WARNING: no bounces detected in observed trajectory", file=sys.stderr)
        code = 1
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj",
        type=Path,
        required=True,
        help="Step 4a trajectory_smoothed.csv (frame,t_sec,x,y,z)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output step4b/ directory")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS metadata; inferred from t_sec spacing if omitted",
    )
    parser.add_argument(
        "--free-gravity",
        action="store_true",
        help="Fit gravity (default: fixed at -9.81 m/s²)",
    )
    parser.add_argument(
        "--gt-poses",
        type=Path,
        default=None,
        help="Optional object_poses.csv for translation alignment before fitting",
    )
    parser.add_argument(
        "--align-translation-to-gt",
        action="store_true",
        help="Subtract mean (GT - traj) translation before fitting (world-frame offset)",
    )
    parser.add_argument(
        "--robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
    )
    parser.add_argument("--robust-f-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run_step4b(
        trajectory_csv=args.traj,
        out_dir=args.out,
        fps=args.fps,
        fix_gravity=not args.free_gravity,
        gt_poses_csv=args.gt_poses,
        align_translation_to_gt=args.align_translation_to_gt,
        robust_loss=args.robust_loss,
        robust_f_scale=args.robust_f_scale,
    )

    print(f"Step 4b complete → {result.out_dir}")
    print(f"  params: {result.params_json}")
    print(f"  plot:   {result.fit_plot_png}")
    print(f"  fit_mse_m2: {result.fit_mse_m2:.8f}")
    print(f"  bounce_count: {result.bounce_count}")

    warn_code = _check_acceptance(result, fix_gravity=not args.free_gravity)
    return warn_code


if __name__ == "__main__":
    raise SystemExit(main())
