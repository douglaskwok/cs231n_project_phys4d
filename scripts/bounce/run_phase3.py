#!/usr/bin/env python
"""Phase 3 — extrapolate fitted bounce physics onto held-out split frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.extrapolate import run_phase3  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        type=Path,
        required=True,
        help="Phase-2 physics_params.json path.",
    )
    parser.add_argument(
        "--traj",
        type=Path,
        required=True,
        help="Phase-1 smoothed trajectory CSV path.",
    )
    parser.add_argument(
        "--dynerf-export",
        type=Path,
        required=True,
        help="Object-only export folder with frame_map.json or export_meta.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output phase3 directory.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()

    result = run_phase3(
        params_json=args.params,
        trajectory_csv=args.traj,
        dynerf_export=args.dynerf_export,
        out_dir=args.out,
        fps=args.fps,
    )
    print(f"Phase 3 complete -> {result.out_dir}")
    print(f"  predicted: {result.predicted_csv}")
    print(f"  plot:      {result.plot_png}")
    print(f"  num_test_frames: {result.num_test_frames}")
    print(f"  boundary_gap_m:  {result.boundary_gap_m:.6f}")
    print(f"  min_clearance_m: {result.min_clearance_m:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
