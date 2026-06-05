#!/usr/bin/env python
"""Forward-predict held-out frames from a multi-body collision refit (collision Step 4c).

Joint analogue of ``scripts/bounce/predict_metric_from_refit.py``. Consumes
``refit_metric.json`` from ``scripts/collision/refit_metric_gt.py`` (the fitted
metric-frame bodies + shared gravity / ground / pair restitution) and integrates
ALL bodies together from the anchor frame through the requested window. Writes one
``trajectory_predicted_obj{k}.csv`` per object (``frame,t_sec,x,y,z``).

Example::

  python scripts/collision/predict_metric_from_refit.py \\
    --refit outputs/collision_pipeline/<RUN>/step4b_metric/refit_metric.json \\
    --out-dir outputs/collision_pipeline/<RUN>/step4c \\
    --frame-start 40 --frame-end 60 --anchor-frame 0 --fps 120 \\
    --gt-poses scene/object_poses_obj0.csv,scene/object_poses_obj1.csv
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

from phys4d.collision.physics import Body, simulate_scene  # noqa: E402


def _split(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


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
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dir for per-object CSVs")
    ap.add_argument("--frame-start", type=int, default=None, help="First held-out frame (default: refit test[0])")
    ap.add_argument("--frame-end", type=int, default=None, help="Last held-out frame (default: refit test[1])")
    ap.add_argument("--anchor-frame", type=int, default=None, help="Frame where bodies (p0,v0) are valid")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--substeps", type=int, default=None, help="Override integrator sub-steps")
    ap.add_argument("--gt-poses", default=None, help="Comma-separated per-object object_poses.csv for RMSE")
    args = ap.parse_args()

    blob = json.loads(args.refit.read_text(encoding="utf-8"))
    split = blob.get("split") or {}
    train_rng = split.get("train") or [0, 0]
    test_rng = split.get("test") or [1, 1]
    frame_start = int(args.frame_start if args.frame_start is not None else test_rng[0])
    frame_end = int(args.frame_end if args.frame_end is not None else test_rng[1])
    anchor_frame = int(args.anchor_frame if args.anchor_frame is not None else train_rng[0])

    body_blobs = blob["bodies"]
    n_obj = len(body_blobs)
    bodies = [
        Body(
            p0=np.array(b["p0_m"], dtype=np.float64),
            v0=np.array(b["v0_m_s"], dtype=np.float64),
            radius=float(b["radius_m"]),
            mass=float(b["mass_kg"]),
            restitution_floor=float(b["restitution_floor"]),
        )
        for b in body_blobs
    ]
    g = float(blob["gravity_z_m_s2"])
    ground = float(blob["ground_z_m"])
    e_pair = float(blob["restitution_pair"])
    e_wall = float(blob.get("restitution_wall", 1.0))
    walls = blob.get("walls") or {}
    wall_x = tuple(walls["x"]) if walls.get("x") else None
    wall_y = tuple(walls["y"]) if walls.get("y") else None
    substeps = int(args.substeps if args.substeps is not None else blob.get("substeps", 8))

    # Integrate continuously from the anchor so collision phase accumulates correctly.
    frames = np.arange(anchor_frame, frame_end + 1, dtype=np.int64)
    times = (frames - anchor_frame).astype(np.float64) / args.fps
    pred = simulate_scene(
        times, bodies=bodies, gravity_z=g, ground_z=ground,
        restitution_pair=e_pair, substeps=substeps,
        wall_x=wall_x, wall_y=wall_y, restitution_wall=e_wall,
    )  # (T, nb, 3)

    keep = frames >= frame_start
    out_frames = frames[keep]
    out_times = out_frames.astype(np.float64) / args.fps
    out_pos = pred[keep]  # (K, nb, 3)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_paths = [Path(p) for p in _split(args.gt_poses)]
    rmses: list[float] = []

    for k in range(n_obj):
        csv_path = out_dir / f"trajectory_predicted_obj{k}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t_sec", "x", "y", "z"])
            for fr, t_s, xyz in zip(out_frames.tolist(), out_times.tolist(), out_pos[:, k, :]):
                w.writerow([int(fr), f"{t_s:.6f}", f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}"])

        rmse = float("nan")
        if k < len(gt_paths):
            gt = _load_gt(gt_paths[k])
            common = [int(fr) for fr in out_frames.tolist() if int(fr) in gt]
            if common:
                idx = {int(fr): i for i, fr in enumerate(out_frames.tolist())}
                err = np.array([np.linalg.norm(out_pos[idx[fr], k, :] - gt[fr]) for fr in common])
                rmse = float(np.sqrt(np.mean(err ** 2)))
        rmses.append(rmse)
        msg = f"obj{k}: {out_frames.shape[0]} frames -> {csv_path}"
        if np.isfinite(rmse):
            msg += f"  RMSE {rmse*100:.2f} cm"
        print(msg)

    meta = {
        "refit": str(args.refit.resolve()),
        "n_objects": n_obj,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "anchor_frame": anchor_frame,
        "fps": args.fps,
        "substeps": substeps,
        "n_predicted": int(out_frames.shape[0]),
        "heldout_test_rmse_per_object_m": rmses,
        "heldout_test_rmse_m": float(np.nanmean(rmses)) if rmses else float("nan"),
    }
    (out_dir / "predict_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
