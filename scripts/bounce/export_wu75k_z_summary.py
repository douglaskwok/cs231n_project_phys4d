#!/usr/bin/env python3
"""Export wu75k bounce step6 metrics and z-only fused trajectory plots."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("outputs/bounce_pipeline")
OUT = Path("outputs/bounce_pipeline/wu75k_z_summary")
RUN_RE = re.compile(r"(?:wu75k_)?scene_\d+_e(?P<e>\d+p\d+)_a(?P<a>m?\d+p\d+)(?:_pred)?$")
DISPLAY_NAME_OVERRIDES = {
    "scene_0003_e0p93_am5p0": "wu75k_scene_0003_e0p93_am5p0_pred",
}


def display_run_name(run_name: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(run_name, run_name)


def load_xyz_csv(path: Path, cols: tuple[str, str, str] = ("x", "y", "z")) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    xyz: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(float(row["frame"])))
            xyz.append([float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])])
    order = np.argsort(frames)
    return np.asarray(frames, dtype=np.int32)[order], np.asarray(xyz, dtype=np.float64)[order]


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_xyz_csv(path, ("x_m", "y_m", "z_m"))


def apply_similarity(sim: dict[str, object], pts: np.ndarray) -> np.ndarray:
    scale = float(sim["scale"])
    rotation = np.asarray(sim["R"], dtype=np.float64)
    translation = np.asarray(sim["t"], dtype=np.float64)
    return (scale * (rotation @ pts.T)).T + translation[None, :]


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def parse_run_params(run_name: str) -> tuple[float | None, float | None]:
    match = RUN_RE.match(run_name)
    if match is None:
        return None, None
    restitution = float(match.group("e").replace("p", "."))
    angle_token = match.group("a")
    sign = -1.0 if angle_token.startswith("m") else 1.0
    angle_text = angle_token[1:] if angle_token.startswith("m") else angle_token
    angle_deg = sign * float(angle_text.replace("p", "."))
    return restitution, angle_deg


def draw_series(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int = 2) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_dots(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, radius: int = 2) -> None:
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, fill=fill, font=fnt)


def plot_z_only(run_dir: Path, metrics: dict[str, object], out_path: Path, title_run: str | None = None) -> None:
    fps = float(metrics.get("fps", 120.0))
    split = metrics.get("split") if isinstance(metrics.get("split"), dict) else {}
    test = split.get("test") if isinstance(split, dict) else None
    test_start = int(test[0]) if test else None

    sources = metrics["sources"]
    gt_frames, gt_xyz = load_gt(Path(sources["gt_poses_csv"]))
    pred_frames, pred_xyz = load_xyz_csv(Path(sources["predicted_csv"]))

    refit_json = run_dir / "step4b_metric" / "refit_metric.json"
    extracted_csv = run_dir / "step4a" / "trajectory_smoothed.csv"
    ext_frames: np.ndarray | None = None
    ext_xyz_metric: np.ndarray | None = None
    if refit_json.is_file() and extracted_csv.is_file():
        refit = json.loads(refit_json.read_text(encoding="utf-8"))
        if refit.get("similarity") is not None:
            ext_frames, ext_xyz = load_xyz_csv(extracted_csv)
            if test_start is not None:
                keep = ext_frames < test_start
                ext_frames = ext_frames[keep]
                ext_xyz = ext_xyz[keep]
            ext_xyz_metric = apply_similarity(refit["similarity"], ext_xyz)

    gt_t = gt_frames / fps
    pred_t = pred_frames / fps
    ext_t = ext_frames / fps if ext_frames is not None else np.asarray([], dtype=np.float64)
    z_chunks = [gt_xyz[:, 2], pred_xyz[:, 2]]
    if ext_xyz_metric is not None and ext_xyz_metric.size:
        z_chunks.append(ext_xyz_metric[:, 2])
    z_all = np.concatenate(z_chunks)
    z_pad = max(float(np.ptp(z_all)) * 0.08, 0.05)
    x_min = 0.0
    x_max = float(max(np.max(gt_t), np.max(pred_t), np.max(ext_t) if ext_t.size else 0.0))
    y_min = float(np.min(z_all) - z_pad)
    y_max = float(np.max(z_all) + z_pad)

    width, height = 1500, 760
    left, right, top, bottom = 120, 52, 86, 106
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(t: np.ndarray | float) -> np.ndarray | float:
        return left + (np.asarray(t) - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(z: np.ndarray | float) -> np.ndarray | float:
        return top + (y_max - np.asarray(z)) / max(y_max - y_min, 1e-9) * plot_h

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    tick_font = font(17)
    title_font = font(26)
    label_font = font(21)
    legend_font = font(17)

    # Matplotlib-like axes: white face, light grid, dark spines, ticks outside.
    draw.rectangle((left, top, left + plot_w, top + plot_h), fill="white", outline="#262626", width=2)
    x_ticks = np.linspace(x_min, x_max, 6)
    y_ticks = np.linspace(y_min, y_max, 6)
    for t in x_ticks:
        x = float(sx(t))
        draw.line((x, top, x, top + plot_h), fill="#d9d9d9", width=1)
        draw.line((x, top + plot_h, x, top + plot_h + 7), fill="#262626", width=2)
        text_center(draw, (x, top + plot_h + 28), f"{t:.1f}", tick_font, "#111111")
    for z in y_ticks:
        y = float(sy(z))
        draw.line((left, y, left + plot_w, y), fill="#d9d9d9", width=1)
        draw.line((left - 7, y, left, y), fill="#262626", width=2)
        text_center(draw, (left - 46, y), f"{z:.2f}", tick_font, "#111111")

    gt_points = list(zip(sx(gt_t).tolist(), sy(gt_xyz[:, 2]).tolist()))
    pred_points = list(zip(sx(pred_t).tolist(), sy(pred_xyz[:, 2]).tolist()))
    draw_series(draw, gt_points, "#2ca02c", width=3)
    draw_series(draw, pred_points, "#d62728", width=4)
    if ext_t.size and ext_xyz_metric is not None:
        ext_points = list(zip(sx(ext_t).tolist(), sy(ext_xyz_metric[:, 2]).tolist()))
        draw_dots(draw, ext_points, "#111111", radius=2)
    if test_start is not None:
        x_split = float(sx(test_start / fps))
        y = top
        while y < top + plot_h:
            draw.line((x_split, y, x_split, min(y + 13, top + plot_h)), fill="#1f77b4", width=3)
            y += 22

    text_center(draw, (left + plot_w / 2, 39), f"Trajectory z — {title_run or run_dir.name}", title_font, "#111111")
    text_center(draw, (left + plot_w / 2, height - 44), "time (seconds)", label_font, "#111111")
    text_center(draw, (38, top + plot_h / 2), "z (m)", label_font, "#111111")

    legend_x, legend_y = left + plot_w - 250, top + 18
    legend = [
        ("#2ca02c", "GT PyBullet", "line"),
        ("#d62728", "predicted (test)", "line"),
        ("#111111", "extracted 4DGS", "dot"),
        ("#1f77b4", "test start", "dash"),
    ]
    draw.rectangle((legend_x - 14, legend_y - 12, legend_x + 226, legend_y + 105), fill="white", outline="#bcbcbc", width=1)
    for idx, (color, text, kind) in enumerate(legend):
        y = legend_y + idx * 28
        if kind == "dot":
            draw_dots(draw, [(legend_x + 14, y + 8), (legend_x + 24, y + 8), (legend_x + 34, y + 8)], color, radius=2)
        elif kind == "dash":
            draw.line((legend_x, y + 8, legend_x + 42, y + 8), fill=color, width=3)
        else:
            draw.line((legend_x, y + 8, legend_x + 42, y + 8), fill=color, width=3 if text == "GT PyBullet" else 4)
        draw.text((legend_x + 52, y), text, fill="#111111", font=legend_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def load_predicted_restitution(run_dir: Path) -> float | None:
    refit_json = run_dir / "step4b_metric" / "refit_metric.json"
    if not refit_json.is_file():
        return None
    refit = json.loads(refit_json.read_text(encoding="utf-8"))
    params = refit.get("params") if isinstance(refit.get("params"), dict) else {}
    value = params.get("restitution")
    return float(value) if value is not None else None


def load_refit_selection(run_dir: Path) -> tuple[float | None, float | None]:
    refit_json = run_dir / "step4b_metric" / "refit_metric.json"
    if not refit_json.is_file():
        return None, None
    refit = json.loads(refit_json.read_text(encoding="utf-8"))
    selection = refit.get("fit_selection") if isinstance(refit.get("fit_selection"), dict) else {}
    selected_initial = selection.get("selected_initial_restitution")
    train_z_rmse = refit.get("train_z_rmse_m")
    return (
        float(selected_initial) if selected_initial is not None else None,
        float(train_z_rmse) if train_z_rmse is not None else None,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for run_dir in sorted(ROOT.iterdir()):
        if not run_dir.is_dir() or RUN_RE.match(run_dir.name) is None:
            continue
        metrics_path = run_dir / "step6" / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        trajectory = metrics.get("trajectory") or {}
        render = metrics.get("render") or {}
        split = metrics.get("split") or {}
        restitution, angle_deg = parse_run_params(run_dir.name)
        predicted_restitution = load_predicted_restitution(run_dir)
        selected_initial_restitution, train_z_rmse_m = load_refit_selection(run_dir)
        run_name = display_run_name(run_dir.name)
        row = {
            "run": run_name,
            "restitution": restitution,
            "predicted_restitution": predicted_restitution,
            "selected_initial_restitution": selected_initial_restitution,
            "angle_deg": angle_deg,
            "fps": metrics.get("fps"),
            "train": split.get("train"),
            "test": split.get("test"),
            "num_pred_frames": metrics.get("num_pred_frames"),
            "frame_min": metrics.get("frame_min"),
            "frame_max": metrics.get("frame_max"),
            "train_z_rmse_m": train_z_rmse_m,
            "pos_rmse_m": trajectory.get("pos_rmse_m"),
            "vel_r2": trajectory.get("vel_r2"),
            "acc_r2": trajectory.get("acc_r2"),
            "psnr_db_mean": render.get("psnr_db_mean"),
            "mae_mean": render.get("mae_mean"),
            "ssim_mean": render.get("ssim_mean"),
        }
        rows.append(row)
        plot_z_only(run_dir, metrics, OUT / f"{run_name}_z_only.png", run_name)

    csv_path = OUT / "step6_metrics_summary.csv"
    json_path = OUT / "step6_metrics_summary.json"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: scalar(v) for k, v in row.items()})
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} runs to {OUT}")
    print(f"Metrics: {csv_path}")
    print(f"Plots: {OUT}/*_z_only.png")


if __name__ == "__main__":
    main()
