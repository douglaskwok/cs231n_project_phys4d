"""Step 6 — trajectory and rendering metrics on held-out test frames."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phys4d.poses import load_object_poses_csv

# Step 5 writes: renders/0060_cam0.png
_RENDER_NAME_RE = re.compile(r"^(\d+)_cam(\d+)\.png$", re.IGNORECASE)

# Milestone3 acceptance targets (reference; not hard failures by default)
MILESTONE_POS_RMSE_M = 0.05
MILESTONE_VEL_R2 = 0.9
MILESTONE_PSNR_DB = 25.0
MILESTONE_SSIM = 0.85


@dataclass
class Step6Result:
    """Artifacts written under ``step6/``."""

    out_dir: Path
    metrics_json: Path
    plot_png: Path
    fused_plot_png: Path | None
    pos_rmse_m: float
    vel_r2: float
    acc_r2: float
    render_psnr_db: float | None
    render_ssim: float | None
    render_mae: float | None


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
    traj = load_object_poses_csv(gt_poses_csv.resolve())
    return np.stack([traj.by_frame(int(f)).position for f in frames.tolist()], axis=0)


def _r2_pearson(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pearson R² between flattened series (milestone3 velocity / acceleration)."""

    p = pred.reshape(-1).astype(np.float64)
    g = gt.reshape(-1).astype(np.float64)
    p = p - float(np.mean(p))
    g = g - float(np.mean(g))
    denom = float(np.linalg.norm(p) * np.linalg.norm(g))
    if denom <= 1e-12:
        return 0.0
    corr = float(np.dot(p, g) / denom)
    return float(np.clip(corr * corr, 0.0, 1.0))


def _safe_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0)


def _read_rgb(path: Path) -> np.ndarray:
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


def _ssim_mean(pred: np.ndarray, gt: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    vals = [
        float(structural_similarity(pred[..., c], gt[..., c], data_range=1.0))
        for c in range(pred.shape[2])
    ]
    return float(np.mean(vals))


def _render_metrics(rendered_dir: Path, gt_rgb_root: Path) -> dict[str, float | int | None]:
    """Match Step 5 PNG names to PyBullet GT rgb tree."""

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
    unparsed = 0

    for png in sorted(rendered_dir.glob("*.png")):
        m = _RENDER_NAME_RE.match(png.name)
        if not m:
            unparsed += 1
            continue
        frame_idx = int(m.group(1))
        cam_idx = int(m.group(2))
        gt_path = gt_rgb_root / f"cam{cam_idx:02d}" / f"frame{frame_idx:05d}.png"
        if not gt_path.is_file():
            missing_gt += 1
            continue
        pred = _read_rgb(png)
        gt = _read_rgb(gt_path)
        h, w = min(pred.shape[0], gt.shape[0]), min(pred.shape[1], gt.shape[1])
        pred = pred[:h, :w, :3]
        gt = gt[:h, :w, :3]
        psnrs.append(_psnr(pred, gt))
        maes.append(float(np.mean(np.abs(pred - gt))))
        ssim_val = _ssim_mean(pred, gt)
        if ssim_val is not None:
            ssims.append(ssim_val)
        matched += 1

    if matched == 0:
        raise ValueError(
            f"No render/GT pairs under {rendered_dir} (unparsed names: {unparsed}, "
            f"missing GT: {missing_gt}). Expected names like 0060_cam0.png"
        )
    return {
        "num_matched": matched,
        "num_missing_gt": missing_gt,
        "num_unparsed_filenames": unparsed,
        "psnr_db_mean": float(np.mean(psnrs)),
        "mae_mean": float(np.mean(maes)),
        "ssim_mean": float(np.mean(ssims)) if ssims else None,
    }


def _load_traj_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``frame,t_sec,x,y,z`` CSV as frame ids and xyz (metric or 4DGS frame)."""

    frames: list[int] = []
    xyz: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(float(row["frame"])))
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    if not frames:
        raise ValueError(f"No rows in {path}")
    order = np.argsort(frames)
    fr = np.asarray(frames, dtype=np.int32)[order]
    pos = np.asarray(xyz, dtype=np.float64)[order]
    return fr, pos


def _apply_similarity(sim: dict[str, object], pts: np.ndarray) -> np.ndarray:
    s = float(sim["scale"])
    R = np.asarray(sim["R"], dtype=np.float64)
    t = np.asarray(sim["t"], dtype=np.float64)
    return (s * (R @ pts.T)).T + t[None, :]


def _plot_fused_trajectory(
    path: Path,
    *,
    gt_poses_csv: Path,
    predicted_frames: np.ndarray,
    predicted_xyz: np.ndarray,
    extracted_csv: Path | None,
    refit_metric_json: Path | None,
    split: dict[str, object] | None,
) -> None:
    """GT + metric extracted (train) + metric predicted (test) on one timeline."""

    import matplotlib.pyplot as plt

    from phys4d.poses import load_object_poses_csv

    gt_traj = load_object_poses_csv(gt_poses_csv.resolve())
    gt_frames = np.array([p.frame for p in gt_traj.poses], dtype=np.int32)
    gt_xyz = np.stack([p.position for p in gt_traj.poses], axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    split_train = (split or {}).get("train") if split else None
    split_test = (split or {}).get("test") if split else None
    if split_train and split_test:
        title = f"Trajectory — train {split_train[0]}..{split_train[1]} / test {split_test[0]}..{split_test[1]}"
    else:
        title = "Trajectory — extracted (train) + predicted (test) vs GT"

    extracted_plotted = False
    if extracted_csv is not None and extracted_csv.is_file() and refit_metric_json is not None:
        blob = json.loads(refit_metric_json.read_text(encoding="utf-8"))
        sim = blob.get("similarity")
        if sim is not None:
            ext_fr, ext_xyz = _load_traj_xyz(extracted_csv)
            ext_m = _apply_similarity(sim, ext_xyz)
            for ax, axis, name in zip(axes, range(3), ("x", "y", "z")):
                ax.plot(ext_fr, ext_m[:, axis], "k.", ms=3, alpha=0.5, label="extracted (4DGS→metric)")
                extracted_plotted = True

    for ax, axis, name in zip(axes, range(3), ("x", "y", "z")):
        ax.plot(gt_frames, gt_xyz[:, axis], "g-", lw=1.8, label="GT PyBullet")
        if extracted_plotted and ax.get_legend_handles_labels()[0]:
            pass  # extracted label already added
        ax.plot(predicted_frames, predicted_xyz[:, axis], "r-", lw=2.0, label="predicted (test)")
        if split_test:
            ax.axvline(int(split_test[0]), color="b", ls="--", alpha=0.45, label="test start")
        ax.set_ylabel(f"{name} (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("simulation frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_metrics(path: Path, frames: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, axis, name in zip(axes, range(3), ("x", "y", "z")):
        ax.plot(frames, gt[:, axis], "--", lw=1.5, label="GT PyBullet")
        ax.plot(frames, pred[:, axis], "-", lw=2.0, label="predicted")
        ax.set_ylabel(f"{name} (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("simulation frame")
    fig.suptitle("Step 6 — held-out trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _milestone_checks(
    *,
    pos_rmse_m: float,
    vel_r2: float,
    render_psnr: float | None,
    render_ssim: float | None,
) -> dict[str, bool]:
    return {
        "pos_rmse_lt_0p05m": pos_rmse_m < MILESTONE_POS_RMSE_M,
        "vel_r2_gt_0p9": vel_r2 > MILESTONE_VEL_R2,
        "psnr_gt_25db": render_psnr is not None and render_psnr > MILESTONE_PSNR_DB,
        "ssim_gt_0p85": render_ssim is not None and render_ssim > MILESTONE_SSIM,
    }


def run_step6(
    *,
    predicted_csv: Path,
    gt_poses_csv: Path,
    rendered_dir: Path,
    gt_rgb_root: Path,
    out_dir: Path,
    fps: float = 60.0,
    skip_render_metrics: bool = False,
    extracted_traj_csv: Path | None = None,
    refit_metric_json: Path | None = None,
    split: dict[str, object] | None = None,
) -> Step6Result:
    """Compute trajectory + rendering metrics for held-out predictions."""

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, pred = _read_predicted_csv(predicted_csv)
    gt = _positions_for_frames(gt_poses_csv, frames)
    pos_rmse = float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))

    dt = 1.0 / max(float(fps), 1e-9)
    vel_r2 = _r2_pearson(_safe_gradient(pred, dt), _safe_gradient(gt, dt))
    acc_r2 = _r2_pearson(_safe_gradient(_safe_gradient(pred, dt), dt), _safe_gradient(_safe_gradient(gt, dt), dt))

    render_block: dict[str, float | int | None] | None = None
    if not skip_render_metrics:
        render_block = _render_metrics(rendered_dir, gt_rgb_root)

    metrics = {
        "fps": float(fps),
        "split": split,
        "num_pred_frames": int(frames.shape[0]),
        "frame_min": int(frames[0]),
        "frame_max": int(frames[-1]),
        "trajectory": {
            "pos_rmse_m": pos_rmse,
            "vel_r2": vel_r2,
            "acc_r2": acc_r2,
        },
        "render": render_block,
        "milestone_targets": {
            "pos_rmse_m": MILESTONE_POS_RMSE_M,
            "vel_r2": MILESTONE_VEL_R2,
            "psnr_db": MILESTONE_PSNR_DB,
            "ssim": MILESTONE_SSIM,
        },
        "milestone_pass": _milestone_checks(
            pos_rmse_m=pos_rmse,
            vel_r2=vel_r2,
            render_psnr=render_block["psnr_db_mean"] if render_block else None,
            render_ssim=render_block["ssim_mean"] if render_block else None,
        ),
        "sources": {
            "predicted_csv": str(predicted_csv.resolve()),
            "gt_poses_csv": str(gt_poses_csv.resolve()),
            "rendered_dir": str(rendered_dir.resolve()),
            "gt_rgb_root": str(gt_rgb_root.resolve()),
        },
    }

    metrics_json = out_dir / "metrics.json"
    plot_png = out_dir / "metrics_plot.png"
    fused_plot_png = out_dir / "trajectory_fused_plot.png"
    metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _plot_metrics(plot_png, frames, pred, gt)
    if extracted_traj_csv is not None or refit_metric_json is not None:
        _plot_fused_trajectory(
            fused_plot_png,
            gt_poses_csv=gt_poses_csv,
            predicted_frames=frames,
            predicted_xyz=pred,
            extracted_csv=extracted_traj_csv,
            refit_metric_json=refit_metric_json,
            split=split,
        )
    else:
        fused_plot_png = None

    return Step6Result(
        out_dir=out_dir,
        metrics_json=metrics_json,
        plot_png=plot_png,
        fused_plot_png=fused_plot_png,
        pos_rmse_m=pos_rmse,
        vel_r2=vel_r2,
        acc_r2=acc_r2,
        render_psnr_db=float(render_block["psnr_db_mean"]) if render_block else None,
        render_ssim=float(render_block["ssim_mean"]) if render_block and render_block["ssim_mean"] is not None else None,
        render_mae=float(render_block["mae_mean"]) if render_block else None,
    )
