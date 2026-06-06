#!/usr/bin/env python3
"""Export collision x-trajectory charts for object A/B and a master grid."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("outputs/collision_pipeline")
SUMMARY = ROOT / "summary" / "step6_metrics_summary_compact.csv"
OUT = ROOT / "summary" / "x_charts"
FPS_DEFAULT = 60.0


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


def text_center(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], text: str, fnt: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), text, fill=fill, font=fnt)


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


def apply_similarity(sim: dict[str, object], pts: np.ndarray) -> np.ndarray:
    scale = float(sim["scale"])
    rotation = np.asarray(sim["R"], dtype=np.float64)
    translation = np.asarray(sim["t"], dtype=np.float64)
    return (scale * (rotation @ pts.T)).T + translation[None, :]


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_dots(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, radius: int = 2) -> None:
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    fps: float,
    split: tuple[int, int],
    gt_csv: Path,
    pred_csv: Path,
    extracted_csv: Path,
    similarity: dict[str, object],
    start_frame: int,
    y_limits: tuple[float, float],
) -> None:
    left, top, right, bottom = box
    label_font = font(20)
    tick_font = font(18)
    small_font = font(18)
    title_font = font(42, bold=True)
    plot_left, plot_right = left + 86, right - 24
    plot_top, plot_bottom = top + 54, bottom - 54
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    test_start, test_end = split
    gt_frames, gt_xyz = load_xyz_csv(gt_csv, ("x_m", "y_m", "z_m"))
    pred_frames, pred_xyz = load_xyz_csv(pred_csv)
    ext_frames, ext_xyz = load_xyz_csv(extracted_csv)
    ext_metric = apply_similarity(similarity, ext_xyz)

    gt_keep = (gt_frames >= start_frame) & (gt_frames <= test_end)
    pred_keep = (pred_frames >= start_frame) & (pred_frames <= test_end)
    ext_keep = (ext_frames >= start_frame) & (ext_frames < test_start)
    gt_frames, gt_xyz = gt_frames[gt_keep], gt_xyz[gt_keep]
    pred_frames, pred_xyz = pred_frames[pred_keep], pred_xyz[pred_keep]
    ext_frames, ext_metric = ext_frames[ext_keep], ext_metric[ext_keep]

    x_min = start_frame / fps
    x_max = test_end / fps
    y_min, y_max = y_limits

    def sx(t: np.ndarray | float) -> np.ndarray:
        return plot_left + (np.asarray(t) - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(v: np.ndarray | float) -> np.ndarray:
        return plot_top + (y_max - np.asarray(v)) / max(y_max - y_min, 1e-9) * plot_h

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), fill="white", outline="#262626", width=2)
    for t in np.linspace(x_min, x_max, 5):
        x = float(sx(t))
        draw.line((x, plot_top, x, plot_bottom), fill="#dedede", width=1)
        draw.line((x, plot_bottom, x, plot_bottom + 6), fill="#262626", width=2)
        text_center(draw, (x - 32, plot_bottom + 10, x + 32, plot_bottom + 30), f"{t:.1f}", tick_font, "#111111")
    for v in np.linspace(y_min, y_max, 5):
        y = float(sy(v))
        draw.line((plot_left, y, plot_right, y), fill="#dedede", width=1)
        draw.line((plot_left - 6, y, plot_left, y), fill="#262626", width=2)
        text_center(draw, (left + 6, y - 10, plot_left - 10, y + 10), f"{v:.2f}", tick_font, "#111111")

    draw_polyline(draw, list(zip(sx(gt_frames / fps).tolist(), sy(gt_xyz[:, 0]).tolist())), "#2ca02c", 3)
    if len(pred_frames):
        draw_polyline(draw, list(zip(sx(pred_frames / fps).tolist(), sy(pred_xyz[:, 0]).tolist())), "#d62728", 4)
    if len(ext_frames):
        draw_dots(draw, list(zip(sx(ext_frames / fps).tolist(), sy(ext_metric[:, 0]).tolist())), "#111111", 2)

    x_split = float(sx(test_start / fps))
    y = plot_top
    while y < plot_bottom:
        draw.line((x_split, y, x_split, min(y + 11, plot_bottom)), fill="#1f77b4", width=3)
        y += 19

    text_center(draw, (left, top + 2, right, top + 48), title, title_font, "#111111")
    text_center(draw, (plot_left, bottom - 33, plot_right, bottom - 10), "time (seconds)", label_font, "#111111")
    text_center(draw, (left + 8, plot_top, left + 48, plot_bottom), "x (m)", label_font, "#111111")

    legend_x, legend_y = plot_right - 210, plot_top + 12
    draw.rectangle((legend_x - 10, legend_y - 8, legend_x + 196, legend_y + 82), fill="white", outline="#c8c8c8")
    for idx, (color, text, kind) in enumerate([
        ("#2ca02c", "GT PyBullet", "line"),
        ("#d62728", "predicted (test)", "line"),
        ("#111111", "extracted 4DGS", "dot"),
    ]):
        yy = legend_y + idx * 27
        if kind == "dot":
            draw_dots(draw, [(legend_x + 12, yy + 8), (legend_x + 22, yy + 8), (legend_x + 32, yy + 8)], color, 2)
        else:
            draw.line((legend_x, yy + 8, legend_x + 42, yy + 8), fill=color, width=4 if "predicted" in text else 3)
        draw.text((legend_x + 52, yy), text, fill="#111111", font=small_font)


def scene_chart(run: str, y_limits: tuple[float, float]) -> Path:
    run_dir = ROOT / run
    refit = json.loads((run_dir / "step4b_metric" / "refit_metric.json").read_text(encoding="utf-8"))
    split = tuple(int(v) for v in refit["split"]["test"])
    test_start, test_end = split
    heldout_len = test_end - test_start + 1
    start_frame = max(0, test_start - heldout_len)
    metrics = json.loads((run_dir / "step6" / "obj0" / "metrics.json").read_text(encoding="utf-8"))
    fps = float(metrics.get("fps", FPS_DEFAULT))

    width, height = 2600, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(34, bold=True)
    text_center(draw, (0, 10, width, 52), f"{run}  |  x trajectory balanced zoom", title_font, "#111111")

    panel_w = (width - 58) // 2
    for obj_idx, name in [(0, "Object A"), (1, "Object B")]:
        left = 20 + obj_idx * (panel_w + 18)
        draw_panel(
            draw,
            (left, 62, left + panel_w, height - 20),
            name,
            fps,
            split,
            run_dir / "_poses" / f"object_poses_obj{obj_idx}.csv",
            run_dir / "step4c" / f"trajectory_predicted_obj{obj_idx}.csv",
            run_dir / "step4a" / f"obj{obj_idx}" / "trajectory_smoothed.csv",
            refit["similarity"][obj_idx],
            start_frame,
            y_limits,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{run}_x_objA_objB_zoom_side_by_side.png"
    img.save(out_path)
    return out_path


def collect_x_values(run: str) -> np.ndarray:
    run_dir = ROOT / run
    refit = json.loads((run_dir / "step4b_metric" / "refit_metric.json").read_text(encoding="utf-8"))
    test_start, test_end = [int(v) for v in refit["split"]["test"]]
    heldout_len = test_end - test_start + 1
    start_frame = max(0, test_start - heldout_len)
    chunks: list[np.ndarray] = []
    for obj_idx in (0, 1):
        gt_frames, gt_xyz = load_xyz_csv(run_dir / "_poses" / f"object_poses_obj{obj_idx}.csv", ("x_m", "y_m", "z_m"))
        pred_frames, pred_xyz = load_xyz_csv(run_dir / "step4c" / f"trajectory_predicted_obj{obj_idx}.csv")
        ext_frames, ext_xyz = load_xyz_csv(run_dir / "step4a" / f"obj{obj_idx}" / "trajectory_smoothed.csv")
        ext_metric = apply_similarity(refit["similarity"][obj_idx], ext_xyz)
        chunks.append(gt_xyz[(gt_frames >= start_frame) & (gt_frames <= test_end), 0])
        chunks.append(pred_xyz[(pred_frames >= start_frame) & (pred_frames <= test_end), 0])
        chunks.append(ext_metric[(ext_frames >= start_frame) & (ext_frames < test_start), 0])
    return np.concatenate([chunk for chunk in chunks if len(chunk)])


def shared_y_limits(runs: list[str]) -> tuple[float, float]:
    values = np.concatenate([collect_x_values(run) for run in runs])
    pad = max(float(np.ptp(values)) * 0.06, 0.03)
    return float(np.min(values) - pad), float(np.max(values) + pad)


def make_grid(chart_paths: list[Path], out_path: Path, *, cols: int, cell_w: int, cell_h: int) -> Path:
    rows = int(np.ceil(len(chart_paths) / cols))
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
    for idx, path in enumerate(chart_paths):
        im = Image.open(path).convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        grid.paste(im, ((idx % cols) * cell_w, (idx // cols) * cell_h))
    grid.save(out_path)
    return out_path


def make_master(chart_paths: list[Path]) -> Path:
    cell_w, cell_h = 1200, 392
    cols = 2
    out_path = OUT / "master_collision_x_objA_objB_2col_balanced_zoom_side_by_side_shared_y.png"
    return make_grid(chart_paths, out_path, cols=cols, cell_w=cell_w, cell_h=cell_h)


def make_paperwide_master(chart_paths: list[Path]) -> Path:
    out_path = OUT / "master_collision_x_objA_objB_4col_paperwide_balanced_zoom_shared_y.png"
    return make_grid(chart_paths, out_path, cols=4, cell_w=1200, cell_h=260)


def make_split_masters(chart_paths: list[Path]) -> list[Path]:
    groups = [
        ("0000_0003", chart_paths[:4]),
        ("0004_0007", chart_paths[4:8]),
    ]
    paths: list[Path] = []
    for label, group in groups:
        if not group:
            continue
        paths.append(
            make_grid(
                group,
                OUT / f"master_collision_x_objA_objB_{label}_1col_wide_rows_balanced_zoom_shared_y.png",
                cols=1,
                cell_w=2600,
                cell_h=620,
            )
        )
    return paths


def main() -> None:
    rows = list(csv.DictReader(SUMMARY.open(newline="", encoding="utf-8")))
    runs = [row["run"] for row in rows]
    y_limits = shared_y_limits(runs)
    chart_paths = [scene_chart(run, y_limits) for run in runs]
    master = make_master(chart_paths)
    paperwide_master = make_paperwide_master(chart_paths)
    split_masters = make_split_masters(chart_paths)
    print(f"Wrote {len(chart_paths)} scene charts to {OUT}")
    print(f"Shared y-limits: {y_limits[0]:.4f}, {y_limits[1]:.4f}")
    print(f"Master: {master}")
    print(f"Paperwide master: {paperwide_master}")
    for split_master in split_masters:
        print(f"Split master: {split_master}")


if __name__ == "__main__":
    main()
