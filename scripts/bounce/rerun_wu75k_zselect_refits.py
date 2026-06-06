#!/usr/bin/env python3
"""Rerun wu75k bounce refit/predict/eval with train-z-RMSE model selection."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import csv
from pathlib import Path

import numpy as np


ROOT = Path("outputs/bounce_pipeline")
RUN_RE = re.compile(r"(?:wu75k_)?scene_\d+_e\d+p\d+_a(?P<a>m?\d+p\d+)(?:_pred)?$")
PY = Path(".venv_pipeline/bin/python")
PY_CMD = ["arch", "-arm64", str(PY)]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def is_zero_angle(run_name: str) -> bool:
    match = RUN_RE.match(run_name)
    return bool(match and match.group("a") == "0p0")


def load_xyz(path: Path, cols: tuple[str, str, str]) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    xyz: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(float(row["frame"])))
            xyz.append([float(row[c]) for c in cols])
    order = np.argsort(frames)
    return np.asarray(frames, dtype=np.int32)[order], np.asarray(xyz, dtype=np.float64)[order]


def safe_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0)


def r2_pearson(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.reshape(-1).astype(np.float64)
    g = gt.reshape(-1).astype(np.float64)
    p = p - float(np.mean(p))
    g = g - float(np.mean(g))
    denom = float(np.linalg.norm(p) * np.linalg.norm(g))
    if denom <= 1e-12:
        return 0.0
    corr = float(np.dot(p, g) / denom)
    return float(np.clip(corr * corr, 0.0, 1.0))


def update_step6_trajectory_metrics(run_dir: Path, gt_poses: Path, fps: float) -> None:
    metrics_path = run_dir / "step6" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pred_frames, pred = load_xyz(run_dir / "step4c" / "trajectory_predicted.csv", ("x", "y", "z"))
    gt_frames, gt_all = load_xyz(gt_poses, ("x_m", "y_m", "z_m"))
    gt_by_frame = {int(f): p for f, p in zip(gt_frames.tolist(), gt_all)}
    gt = np.stack([gt_by_frame[int(f)] for f in pred_frames.tolist()], axis=0)
    dt = 1.0 / max(float(fps), 1e-9)
    pos_rmse = float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))
    vel_r2 = r2_pearson(safe_gradient(pred, dt), safe_gradient(gt, dt))
    acc_r2 = r2_pearson(safe_gradient(safe_gradient(pred, dt), dt), safe_gradient(safe_gradient(gt, dt), dt))

    metrics["num_pred_frames"] = int(pred_frames.shape[0])
    metrics["frame_min"] = int(pred_frames[0])
    metrics["frame_max"] = int(pred_frames[-1])
    metrics["trajectory"] = {
        "pos_rmse_m": pos_rmse,
        "vel_r2": vel_r2,
        "acc_r2": acc_r2,
    }
    render = metrics.get("render") or {}
    metrics["milestone_pass"] = {
        "pos_rmse_lt_0p05m": pos_rmse < 0.05,
        "vel_r2_gt_0p9": vel_r2 > 0.9,
        "psnr_gt_25db": render.get("psnr_db_mean") is not None and render.get("psnr_db_mean") > 25.0,
        "ssim_gt_0p85": render.get("ssim_mean") is not None and render.get("ssim_mean") > 0.85,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if not PY.is_file():
        raise SystemExit(f"Missing pipeline Python: {PY}")

    run_dirs = []
    for run_dir in sorted(ROOT.iterdir()):
        if not run_dir.is_dir() or RUN_RE.match(run_dir.name) is None:
            continue
        if (run_dir / "step6" / "metrics.json").is_file() and (run_dir / "step4a" / "trajectory_smoothed.csv").is_file():
            run_dirs.append(run_dir)

    print(f"Found {len(run_dirs)} completed runs")
    for i, run_dir in enumerate(run_dirs, start=1):
        print(f"\n=== [{i}/{len(run_dirs)}] {run_dir.name} ===", flush=True)
        metrics_path = run_dir / "step6" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        split = metrics.get("split") or {}
        train = split.get("train") or split.get("canonical", {}).get("train")
        test = split.get("test") or split.get("canonical", {}).get("test")
        if not train or not test:
            raise RuntimeError(f"Missing split in {metrics_path}")
        sources = metrics["sources"]
        gt_poses = Path(sources["gt_poses_csv"])
        rendered = Path(sources["rendered_dir"])
        gt_rgb_root = Path(sources["gt_rgb_root"])
        fps = str(metrics.get("fps", 120.0))

        run(
            [
                *PY_CMD,
                "scripts/bounce/refit_metric_gt.py",
                "--traj",
                str(run_dir / "step4a" / "trajectory_smoothed.csv"),
                "--gt-poses",
                str(gt_poses),
                "--out",
                str(run_dir / "step4b_metric"),
                "--train-start",
                str(train[0]),
                "--train-end",
                str(train[1]),
                "--test-start",
                str(test[0]),
                "--test-end",
                str(test[1]),
                "--selection-metric",
                "train_z_rmse",
            ]
        )

        pred_cmd = [
            *PY_CMD,
            "scripts/bounce/predict_metric_from_refit.py",
            "--refit",
            str(run_dir / "step4b_metric" / "refit_metric.json"),
            "--out",
            str(run_dir / "step4c" / "trajectory_predicted.csv"),
            "--frame-start",
            str(test[0]),
            "--frame-end",
            str(test[1]),
            "--anchor-frame",
            str(train[0]),
            "--fps",
            fps,
            "--gt-poses",
            str(gt_poses),
        ]
        if is_zero_angle(run_dir.name):
            pred_cmd.append("--zero-horizontal")
        run(pred_cmd)

        update_step6_trajectory_metrics(run_dir, gt_poses, float(fps))
        print(f"Updated trajectory metrics in {run_dir / 'step6' / 'metrics.json'}", flush=True)

    run([sys.executable, "scripts/bounce/export_wu75k_z_summary.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
