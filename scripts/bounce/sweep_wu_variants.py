#!/usr/bin/env python
"""Rank Wu 4DGaussians model variants for a scene by Phase-1 agreement with GT.

Runs the (fast, CPU) Phase-1 extraction on each trained variant under the scene's
``4dgs_wu/`` folder and reports train RMSE vs GT plus the x/y centroid drift span.
Lowest RMSE / smallest x-y span = best reconstruction for the bounce pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]


def _find_model_dirs(scene_dir: Path) -> list[Path]:
    """Inner ``wu4dgs_*`` dirs (those holding point_cloud/ + cfg_args)."""

    out = []
    for cfg in sorted((scene_dir / "4dgs_wu").rglob("cfg_args")):
        if (cfg.parent / "point_cloud").is_dir():
            out.append(cfg.parent)
    return out


def _xy_span(traj_csv: Path) -> tuple[float, float]:
    rows = list(csv.DictReader(traj_csv.open()))
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    return float(x.max() - x.min()), float(y.max() - y.min())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-end", type=int, default=40)
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    gt = scene / "object_poses.csv"
    models = _find_model_dirs(scene)
    if not models:
        print(f"No model variants under {scene/'4dgs_wu'}", file=sys.stderr)
        return 1

    results = []
    for model in models:
        label = model.parent.name  # the RUN_NAME folder
        out = args.out_root.resolve() / "sweep" / label
        cmd = [
            sys.executable,
            str(_HERE / "run_phase1_wu.py"),
            "--model-dir", str(model),
            "--scene-config", str(scene / "config.json"),
            "--gt-poses", str(gt),
            "--frame-start", str(args.frame_start),
            "--frame-end", str(args.frame_end),
            "--align-to-gt",
            "--out", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        meta_path = out / "phase1_meta.json"
        if proc.returncode != 0 or not meta_path.is_file():
            print(f"[FAIL] {label}\n{proc.stderr[-400:]}", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text())
        rmse = float(np.sqrt(meta["train_mse_m2"]))
        xs, ys = _xy_span(out / "trajectory_raw.csv")
        results.append((label, rmse, xs, ys, meta["num_gaussians"]))

    results.sort(key=lambda r: r[1])
    print(f"\n{'rank':4s} {'rmse_m':>9s} {'x_span':>8s} {'y_span':>8s} {'gauss':>7s}  variant")
    for i, (label, rmse, xs, ys, ng) in enumerate(results):
        short = label.replace("wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_", "")
        print(f"{i:4d} {rmse:9.4f} {xs:8.4f} {ys:8.4f} {ng:7d}  {short}")
    if results:
        print(f"\nBEST: {results[0][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
