#!/usr/bin/env python3
"""Sweep bounce refit objectives/selection metrics and summarize velocity R2."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("outputs/bounce_pipeline")
IN_SUMMARY = ROOT / "wu75k_z_summary" / "step6_metrics_summary.csv"
OUT = ROOT / "wu75k_velocity_refit_summary"
PY = Path(".venv_pipeline/bin/python")
FPS = 120.0
RUN_DIR_OVERRIDES = {
    "wu75k_scene_0003_e0p93_am5p0_pred": "scene_0003_e0p93_am5p0",
}

STRATEGIES = [
    ("pos_train_xyz", "pos", "train_xyz_rmse"),
    ("pos_train_gt_zvel", "pos", "train_gt_zvel_r2"),
    ("pos_zvel_ext_train_ext_zvel", "pos_zvel_ext", "train_ext_zvel_r2"),
    ("pos_zvel_gt_train_gt_zvel", "pos_zvel_gt", "train_gt_zvel_r2"),
]


def read_xyz(path: Path, cols: tuple[str, str, str] = ("x", "y", "z")) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    xyz: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(float(row["frame"])))
            xyz.append([float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])])
    order = np.argsort(frames)
    return np.asarray(frames, dtype=np.int32)[order], np.asarray(xyz, dtype=np.float64)[order]


def apply_similarity(sim: dict[str, object], pts: np.ndarray) -> np.ndarray:
    scale = float(sim["scale"])
    rotation = np.asarray(sim["R"], dtype=np.float64)
    translation = np.asarray(sim["t"], dtype=np.float64)
    return (scale * (rotation @ pts.T)).T + translation[None, :]


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


def trajectory_metrics(pred_csv: Path, gt_csv: Path) -> dict[str, float]:
    pred_frames, pred = read_xyz(pred_csv)
    gt_frames, gt_all = read_xyz(gt_csv, ("x_m", "y_m", "z_m"))
    gt_by_frame = {int(f): xyz for f, xyz in zip(gt_frames.tolist(), gt_all)}
    gt = np.stack([gt_by_frame[int(f)] for f in pred_frames.tolist()], axis=0)
    dt = 1.0 / FPS
    return {
        "pos_rmse_m": float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1)))),
        "z_rmse_m": float(np.sqrt(np.mean((pred[:, 2] - gt[:, 2]) ** 2))),
        "vel_r2": r2_pearson(safe_gradient(pred, dt), safe_gradient(gt, dt)),
        "zvel_r2": r2_pearson(safe_gradient(pred[:, 2:3], dt), safe_gradient(gt[:, 2:3], dt)),
        "acc_r2": r2_pearson(safe_gradient(safe_gradient(pred, dt), dt), safe_gradient(safe_gradient(gt, dt), dt)),
    }


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, fill="#111111", font=fnt)


def draw_plot(run_dir: Path, gt_csv: Path, pred_csv: Path, refit_json: Path, out_png: Path, title: str) -> None:
    gt_frames, gt_xyz = read_xyz(gt_csv, ("x_m", "y_m", "z_m"))
    pred_frames, pred_xyz = read_xyz(pred_csv)
    refit = json.loads(refit_json.read_text(encoding="utf-8"))
    ext_frames, ext_xyz = read_xyz(run_dir / "step4a" / "trajectory_smoothed.csv")
    test_start = int(refit["split"]["test"][0])
    keep = ext_frames < test_start
    ext_frames = ext_frames[keep]
    ext_xyz_m = apply_similarity(refit["similarity"], ext_xyz[keep])

    gt_t = gt_frames / FPS
    pred_t = pred_frames / FPS
    ext_t = ext_frames / FPS
    z_all = np.concatenate([gt_xyz[:, 2], pred_xyz[:, 2], ext_xyz_m[:, 2]])
    y_pad = max(float(np.ptp(z_all)) * 0.08, 0.05)
    x_min, x_max = 0.0, float(np.max(gt_t))
    y_min, y_max = float(np.min(z_all) - y_pad), float(np.max(z_all) + y_pad)

    width, height = 1500, 760
    left, right, top, bottom = 120, 52, 86, 106
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(t: np.ndarray | float) -> np.ndarray:
        return left + (np.asarray(t) - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(z: np.ndarray | float) -> np.ndarray:
        return top + (y_max - np.asarray(z)) / max(y_max - y_min, 1e-9) * plot_h

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    tick_font, label_font, title_font, legend_font = font(17), font(21), font(26), font(17)
    draw.rectangle((left, top, left + plot_w, top + plot_h), fill="white", outline="#262626", width=2)
    for t in np.linspace(x_min, x_max, 6):
        x = float(sx(t))
        draw.line((x, top, x, top + plot_h), fill="#d9d9d9", width=1)
        draw.line((x, top + plot_h, x, top + plot_h + 7), fill="#262626", width=2)
        text_center(draw, (x, top + plot_h + 28), f"{t:.1f}", tick_font)
    for z in np.linspace(y_min, y_max, 6):
        y = float(sy(z))
        draw.line((left, y, left + plot_w, y), fill="#d9d9d9", width=1)
        draw.line((left - 7, y, left, y), fill="#262626", width=2)
        text_center(draw, (left - 46, y), f"{z:.2f}", tick_font)

    draw.line(list(zip(sx(gt_t).tolist(), sy(gt_xyz[:, 2]).tolist())), fill="#2ca02c", width=3)
    draw.line(list(zip(sx(pred_t).tolist(), sy(pred_xyz[:, 2]).tolist())), fill="#d62728", width=4)
    for x, y in zip(sx(ext_t).tolist(), sy(ext_xyz_m[:, 2]).tolist()):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#111111")
    x_split = float(sx(test_start / FPS))
    y = top
    while y < top + plot_h:
        draw.line((x_split, y, x_split, min(y + 13, top + plot_h)), fill="#1f77b4", width=3)
        y += 22

    text_center(draw, (left + plot_w / 2, 39), title, title_font)
    text_center(draw, (left + plot_w / 2, height - 44), "time (seconds)", label_font)
    text_center(draw, (38, top + plot_h / 2), "z (m)", label_font)
    legend_x, legend_y = left + plot_w - 250, top + 18
    draw.rectangle((legend_x - 14, legend_y - 12, legend_x + 226, legend_y + 105), fill="white", outline="#bcbcbc")
    for idx, (color, text) in enumerate([
        ("#2ca02c", "GT PyBullet"),
        ("#d62728", "predicted (test)"),
        ("#111111", "extracted 4DGS"),
        ("#1f77b4", "test start"),
    ]):
        yy = legend_y + idx * 28
        if text == "extracted 4DGS":
            for xx in (legend_x + 14, legend_x + 24, legend_x + 34):
                draw.ellipse((xx - 2, yy + 6, xx + 2, yy + 10), fill=color)
        else:
            draw.line((legend_x, yy + 8, legend_x + 42, yy + 8), fill=color, width=4 if "predicted" in text else 3)
        draw.text((legend_x + 52, yy), text, fill="#111111", font=legend_font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def resolve_run_dir(run_name: str) -> Path:
    return ROOT / RUN_DIR_OVERRIDES.get(run_name, run_name)


def run_command(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = list(csv.DictReader(IN_SUMMARY.open(newline="", encoding="utf-8")))
    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for idx, base in enumerate(summary_rows, 1):
        run = base["run"]
        run_dir = resolve_run_dir(run)
        metrics = json.loads((run_dir / "step6" / "metrics.json").read_text(encoding="utf-8"))
        render = metrics.get("render") or {}
        gt_csv = Path(metrics["sources"]["gt_poses_csv"])
        split = metrics["split"]
        train_start, train_end = split["train"]
        test_start, test_end = split["test"]
        angle = float(base["angle_deg"]) if base.get("angle_deg") else None
        zero_horizontal = angle == 0.0
        print(f"[{idx}/{len(summary_rows)}] {run}", flush=True)

        run_candidates: list[dict[str, object]] = []
        for strategy_name, objective, selection in STRATEGIES:
            strategy_dir = OUT / "candidates" / run / strategy_name
            refit_dir = strategy_dir / "step4b_metric"
            pred_csv = strategy_dir / "trajectory_predicted.csv"
            run_command([
                str(PY),
                "scripts/bounce/refit_metric_gt.py",
                "--traj",
                str(run_dir / "step4a" / "trajectory_smoothed.csv"),
                "--gt-poses",
                str(gt_csv),
                "--out",
                str(refit_dir),
                "--train-start",
                str(train_start),
                "--train-end",
                str(train_end),
                "--test-start",
                str(test_start),
                "--test-end",
                str(test_end),
                "--fit-objective",
                objective,
                "--selection-metric",
                selection,
            ])
            predict_cmd = [
                str(PY),
                "scripts/bounce/predict_metric_from_refit.py",
                "--refit",
                str(refit_dir / "refit_metric.json"),
                "--out",
                str(pred_csv),
                "--frame-start",
                str(test_start),
                "--frame-end",
                str(test_end),
                "--anchor-frame",
                str(train_start),
                "--fps",
                str(FPS),
                "--gt-poses",
                str(gt_csv),
            ]
            if zero_horizontal:
                predict_cmd.append("--zero-horizontal")
            run_command(predict_cmd)
            refit = json.loads((refit_dir / "refit_metric.json").read_text(encoding="utf-8"))
            tm = trajectory_metrics(pred_csv, gt_csv)
            row = {
                "run": run,
                "restitution": base.get("restitution"),
                "angle_deg": base.get("angle_deg"),
                "strategy": strategy_name,
                "fit_objective": objective,
                "selection_metric": selection,
                "predicted_restitution": refit["params"]["restitution"],
                "selected_initial_restitution": refit["fit_selection"]["selected_initial_restitution"],
                "train_z_rmse_m": refit.get("train_z_rmse_m"),
                "train_ext_zvel_r2": max(
                    c.get("train_ext_zvel_r2", 0.0)
                    for c in refit["fit_selection"].get("candidates", [])
                    if c["initial_restitution"] == refit["fit_selection"]["selected_initial_restitution"]
                ),
                "train_gt_zvel_r2": max(
                    c.get("train_gt_zvel_r2", 0.0)
                    for c in refit["fit_selection"].get("candidates", [])
                    if c["initial_restitution"] == refit["fit_selection"]["selected_initial_restitution"]
                ),
                **tm,
                "psnr_db_mean": render.get("psnr_db_mean"),
                "mae_mean": render.get("mae_mean"),
                "ssim_mean": render.get("ssim_mean"),
                "render_metrics_source": "existing_step5_renders",
                "predicted_csv": str(pred_csv),
                "refit_json": str(refit_dir / "refit_metric.json"),
            }
            all_rows.append(row)
            run_candidates.append(row)

        best = max(run_candidates, key=lambda r: (float(r["vel_r2"]), -float(r["pos_rmse_m"])))
        best_rows.append(best)
        draw_plot(
            run_dir,
            gt_csv,
            Path(best["predicted_csv"]),
            Path(best["refit_json"]),
            OUT / "best_charts" / f"{run}_best_vel_z_only.png",
            f"Velocity refit z - {run} ({best['strategy']})",
        )
        print(
            f"  best {best['strategy']} pos={float(best['pos_rmse_m']):.4f} vel_r2={float(best['vel_r2']):.3f}",
            flush=True,
        )

    for path, rows in [(OUT / "all_strategy_metrics.csv", all_rows), (OUT / "best_velocity_metrics.csv", best_rows)]:
        fieldnames = list(rows[0].keys()) if rows else []
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: scalar(v) for k, v in row.items()})

    (OUT / "best_velocity_metrics.json").write_text(json.dumps(best_rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'best_velocity_metrics.csv'}")
    print(f"Wrote {OUT / 'all_strategy_metrics.csv'}")
    print(f"Wrote charts under {OUT / 'best_charts'}")


if __name__ == "__main__":
    main()
