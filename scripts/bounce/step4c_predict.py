#!/usr/bin/env python
"""Step 4c — extrapolate fitted physics onto held-out test frames.

Example::

  python scripts/bounce/step4c_predict.py \\
    --params outputs/bounce_pipeline/<RUN>/step4b/physics_params.json \\
    --frame-map 4dgs/experiments/object_only/runs/<RUN>/frame_map.json \\
    --scene-config dataset/.../scene_<id>/config.json \\
    --traj outputs/bounce_pipeline/<RUN>/step4a/trajectory_smoothed.csv \\
    --gt-poses dataset/.../object_poses.csv \\
    --out outputs/bounce_pipeline/<RUN>/step4c
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.extrapolate import run_step4c  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True, help="step4b/physics_params.json")
    parser.add_argument(
        "--frame-map",
        type=Path,
        required=True,
        help="DyNeRF export frame_map.json (train/test split)",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=None,
        help="PyBullet scene config.json (fps + frame-count validation)",
    )
    parser.add_argument(
        "--traj",
        type=Path,
        default=None,
        help="Step 4a trajectory_smoothed.csv (recommended: plot + boundary check)",
    )
    parser.add_argument(
        "--gt-poses",
        type=Path,
        default=None,
        help="Optional object_poses.csv for GT overlay on full plot",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output step4c/ directory")
    parser.add_argument("--fps", type=float, default=60.0, help="Fallback FPS if not in metadata")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise if boundary gap >= 1 cm or floor penetration (milestone acceptance)",
    )
    args = parser.parse_args()

    if args.traj is None:
        print(
            "WARNING: --traj omitted; boundary continuity vs last train obs will not be checked.",
            file=sys.stderr,
        )

    result = run_step4c(
        params_json=args.params,
        frame_map_path=args.frame_map,
        out_dir=args.out,
        trajectory_csv=args.traj,
        scene_config=args.scene_config,
        gt_poses=args.gt_poses,
        fps=args.fps,
        strict=args.strict,
    )

    print(f"Step 4c complete → {result.out_dir}")
    print(f"  predicted: {result.predicted_csv} ({result.num_test_frames} rows)")
    print(f"  plot:      {result.plot_png}")
    if np.isfinite(result.boundary_gap_m):
        print(f"  boundary_gap_m: {result.boundary_gap_m:.6f}")
        if result.boundary_gap_m >= 0.01:
            print("  WARNING: boundary gap >= 1 cm (milestone3 target)", file=sys.stderr)
    print(f"  min_clearance_m: {result.min_clearance_m:.6f}")
    if result.min_clearance_m < -1e-4:
        print("  WARNING: floor penetration detected", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
