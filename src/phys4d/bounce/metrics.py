"""Phase 5 — evaluate held-out trajectory and render outputs."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phys4d.poses import load_object_poses_csv


_RENDER_NAME_RE = re.compile(r"^\d+_cam(\d+)_(\d+)\.png$")


@dataclass(frozen=True)
class Phase5Result:
    """Artifacts produced by phase 5 evaluation."""

    out_dir: Path
    metrics_json: Path
    plot_png: Path
    pos_rmse_m: float
    vel_r2: float
    acc_r2: float
    render_psnr_db: float
    render_mae: float


def _read_predicted_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read prediction CSV into sorted frame ids and xyz coordinates."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing predicted CSV: {path}")
    rows: list[tuple[int, np.ndarray]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"frame", "x", "y", "z"}
        if reader.fieldnames is None or not needed.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns {sorted(needed)}")
        for row in reader:
            frame = int(float(row["frame"]))
            xyz = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=np.float64,
            )
            rows.append((frame, xyz))
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    rows.sort(key=lambda t: t[0])
    frames = np.asarray([r[0] for r in rows], dtype=np.int32)
    xyz = np.stack([r[1] for r in rows], axis=0)
    return frames, xyz


def _positions_for_frames(gt_poses_csv: Path, frames: np.ndarray) -> np.ndarray:
    """Select ground-truth xyz positions for the specified frame ids."""

    traj = load_object_poses_csv(gt_poses_csv.resolve())
    return np.stack([traj.by_frame(int(f)).position for f in frames.tolist()], axis=0)


def _r2_from_series(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute squared Pearson correlation across flattened series."""

    p = pred.reshape(-1)
    g = gt.reshape(-1)
    p = p - float(np.mean(p))
    g = g - float(np.mean(g))
    denom = float(np.linalg.norm(p) * np.linalg.norm(g))
    if denom <= 1e-12:
        return 0.0
    corr = float(np.dot(p, g) / denom)
    return float(np.clip(corr * corr, 0.0, 1.0))


def _safe_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    """Numerical derivative with stable spacing for short sequences."""

    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0)


def _read_rgb(path: Path) -> np.ndarray:
    """Read RGB image with fallback when imageio is unavailable."""

    try:
        import imageio.v2 as imageio

        arr = imageio.imread(path)
    except Exception:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.float64) / 255.0


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _ssim_if_available(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Compute SSIM only when skimage is available in the runtime."""

    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None
    vals: list[float] = []
    for c in range(pred.shape[2]):
        vals.append(float(structural_similarity(pred[..., c], gt[..., c], data_range=1.0)))
    return float(np.mean(vals))


def _render_metrics(rendered_dir: Path, gt_rgb_root: Path) -> dict[str, float | int | None]:
    """Evaluate rendered PNGs against GT RGB using filename frame/camera ids."""

    rendered_dir = rendered_dir.resolve()
    gt_rgb_root = gt_rgb_root.resolve()
    if not rendered_dir.is_dir():
        raise FileNotFoundError(f"Missing rendered directory: {rendered_dir}")
    if not gt_rgb_root.is_dir():
        raise FileNotFoundError(f"Missing GT RGB root: {gt_rgb_root}")

    psnrs: list[float] = []
    maes: list[float] = []
    ssims: list[float] = []
    matched = 0
    missing_gt = 0
    for png in sorted(rendered_dir.glob("*.png")):
        m = _RENDER_NAME_RE.match(png.name)
        if not m:
            continue
        cam_idx = int(m.group(1))
        frame_idx = int(m.group(2))
        gt_path = gt_rgb_root / f"cam{cam_idx:02d}" / f"frame{frame_idx:05d}.png"
        if not gt_path.is_file():
            missing_gt += 1
            continue
        pred = _read_rgb(png)
        gt = _read_rgb(gt_path)
        if pred.shape != gt.shape:
            h, w = gt.shape[0], gt.shape[1]
            pred = pred[:h, :w, :3]
            gt = gt[: pred.shape[0], : pred.shape[1], :3]
        psnrs.append(_psnr(pred, gt))
        maes.append(float(np.mean(np.abs(pred - gt))))
        ssim_val = _ssim_if_available(pred, gt)
        if ssim_val is not None:
            ssims.append(ssim_val)
        matched += 1

    if matched == 0:
        raise ValueError(f"No render/GT matches found in {rendered_dir} vs {gt_rgb_root}")
    return {
        "num_matched": matched,
        "num_missing_gt": missing_gt,
        "psnr_db_mean": float(np.mean(psnrs)),
        "mae_mean": float(np.mean(maes)),
        "ssim_mean": float(np.mean(ssims)) if ssims else None,
    }


def _plot_metrics(path: Path, frames: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    """Save a compact trajectory comparison plot for held-out frames."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, name in enumerate(labels):
        axes[i].plot(frames, gt[:, i], "--", lw=1.5, label="GT")
        axes[i].plot(frames, pred[:, i], "-", lw=2.0, label="Pred")
        axes[i].set_ylabel(f"{name} (m)")
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle("Phase 5 — held-out trajectory comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_phase5(
    *,
    predicted_csv: Path,
    gt_poses_csv: Path,
    rendered_dir: Path,
    gt_rgb_root: Path,
    out_dir: Path,
    fps: float = 60.0,
) -> Phase5Result:
    """Compute trajectory and render metrics for held-out predictions."""

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, pred = _read_predicted_csv(predicted_csv)
    gt = _positions_for_frames(gt_poses_csv, frames)
    pos_rmse = float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))

    dt = 1.0 / float(fps)
    pred_vel = _safe_gradient(pred, dt)
    gt_vel = _safe_gradient(gt, dt)
    pred_acc = _safe_gradient(pred_vel, dt)
    gt_acc = _safe_gradient(gt_vel, dt)
    vel_r2 = _r2_from_series(pred_vel, gt_vel)
    acc_r2 = _r2_from_series(pred_acc, gt_acc)

    render_metrics = _render_metrics(rendered_dir, gt_rgb_root)
    metrics = {
        "fps": float(fps),
        "num_pred_frames": int(frames.shape[0]),
        "frame_min": int(frames[0]),
        "frame_max": int(frames[-1]),
        "trajectory": {
            "pos_rmse_m": pos_rmse,
            "vel_r2": vel_r2,
            "acc_r2": acc_r2,
        },
        "render": render_metrics,
        "sources": {
            "predicted_csv": str(predicted_csv.resolve()),
            "gt_poses_csv": str(gt_poses_csv.resolve()),
            "rendered_dir": str(rendered_dir.resolve()),
            "gt_rgb_root": str(gt_rgb_root.resolve()),
        },
    }

    metrics_json = out_dir / "metrics.json"
    plot_png = out_dir / "metrics_plot.png"
    metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _plot_metrics(plot_png, frames, pred, gt)

    return Phase5Result(
        out_dir=out_dir,
        metrics_json=metrics_json,
        plot_png=plot_png,
        pos_rmse_m=pos_rmse,
        vel_r2=vel_r2,
        acc_r2=acc_r2,
        render_psnr_db=float(render_metrics["psnr_db_mean"]),
        render_mae=float(render_metrics["mae_mean"]),
    )
