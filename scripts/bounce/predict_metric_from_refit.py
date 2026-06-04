#!/usr/bin/env python
"""Forward-predict the held-out window from a metric-frame refit (Step 4c, metric path).

Consumes ``refit_metric.json`` (from ``refit_metric_gt.py``): the fitted metric-frame
state ``(p0, v0, gravity, restitution, ground_z)`` valid from the train-start anchor.
Continuously integrates the discrete bounce simulator from the anchor frame through the
requested window and writes the held-out positions as ``frame,t_sec,x,y,z``.

Example::

  python scripts/bounce/predict_metric_from_refit.py \\
    --refit outputs/bounce_pipeline/<RUN>/step4b_metric/refit_metric.json \\
    --out outputs/bounce_pipeline/<RUN>/step4c/trajectory_predicted.csv \\
    --frame-start 241 --frame-end 360 --anchor-frame 0 --fps 120 --zero-horizontal \\
    --gt-poses dataset/.../object_poses.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.physics import simulate_trajectory  # noqa: E402


def _load_gt(path: Path) -> dict[int, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out: dict[int, np.ndarray] = {}
    for r in rows:
        out[int(float(r["frame"]))] = np.array(
            [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])], dtype=np.float64
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refit", type=Path, required=True, help="step4b_metric/refit_metric.json")
    ap.add_argument("--out", type=Path, required=True, help="Output trajectory_predicted.csv")
    ap.add_argument("--frame-start", type=int, default=None, help="First held-out frame (default: refit split test[0])")
    ap.add_argument("--frame-end", type=int, default=None, help="Last held-out frame (default: refit split test[1])")
    ap.add_argument(
        "--anchor-frame",
        type=int,
        default=None,
        help="Frame at which (p0,v0) are valid (default: refit split train[0])",
    )
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument(
        "--zero-horizontal",
        action="store_true",
        help="Zero the horizontal (x,y) initial velocity (pure vertical bounce)",
    )
    ap.add_argument("--gt-poses", type=Path, default=None, help="object_poses.csv for held-out RMSE")
    args = ap.parse_args()

    blob = json.loads(args.refit.read_text(encoding="utf-8"))
    split = blob.get("split") or {}
    train_rng = split.get("train") or [0, 0]
    test_rng = split.get("test") or [1, 1]
    frame_start = int(args.frame_start if args.frame_start is not None else test_rng[0])
    frame_end = int(args.frame_end if args.frame_end is not None else test_rng[1])
    anchor_frame = int(args.anchor_frame if args.anchor_frame is not None else train_rng[0])
    p = blob["params"]
    p0 = np.array(p["p0_m"], dtype=np.float64)
    v0 = np.array(p["v0_m_s"], dtype=np.float64)
    if args.zero_horizontal:
        v0 = v0.copy()
        v0[0] = 0.0
        v0[1] = 0.0
    g = float(p["gravity_z_m_s2"])
    e = float(p["restitution"])
    ground = float(p["ground_z_m"])

    # Integrate continuously from the anchor frame so bounce phase accumulates correctly,
    # then keep only the requested held-out window.
    frames = np.arange(anchor_frame, frame_end + 1, dtype=np.int64)
    times = (frames - anchor_frame).astype(np.float64) / args.fps
    pred = simulate_trajectory(times, p0=p0, v0=v0, gravity_z=g, restitution=e, ground_z=ground)

    keep = frames >= frame_start
    out_frames = frames[keep]
    out_times = frames[keep].astype(np.float64) / args.fps
    out_pos = pred[keep]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_sec", "x", "y", "z"])
        for fr, t_s, xyz in zip(out_frames.tolist(), out_times.tolist(), out_pos):
            w.writerow([int(fr), f"{t_s:.6f}", f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}"])

    rmse = float("nan")
    if args.gt_poses is not None:
        gt = _load_gt(args.gt_poses)
        common = [int(fr) for fr in out_frames.tolist() if int(fr) in gt]
        if common:
            idx = {int(fr): i for i, fr in enumerate(out_frames.tolist())}
            err = np.array([np.linalg.norm(out_pos[idx[fr]] - gt[fr]) for fr in common])
            rmse = float(np.sqrt(np.mean(err ** 2)))

    meta = {
        "refit": str(args.refit.resolve()),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "anchor_frame": anchor_frame,
        "fps": args.fps,
        "zero_horizontal": bool(args.zero_horizontal),
        "n_predicted": int(out_frames.shape[0]),
        "heldout_test_rmse_m": rmse,
        "params_used": {"p0_m": p0.tolist(), "v0_m_s": v0.tolist(),
                        "gravity_z_m_s2": g, "restitution": e, "ground_z_m": ground},
    }
    (args.out.parent / "predict_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_frames.shape[0]} predicted frames [{frame_start},{frame_end}] -> {args.out}")
    if np.isfinite(rmse):
        print(f"  held-out position RMSE vs GT: {rmse:.4f} m  ({rmse*100:.2f} cm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
