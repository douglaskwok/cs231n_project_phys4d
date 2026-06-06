#!/usr/bin/env python3
"""Plot Spearman correlations for Gaussian-train vs prediction metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BOUNCE_CSV = Path("outputs/diagnostics/ball_bounce_z_gaussian_train_vs_prediction.csv")
COLLISION_CSV = Path("outputs/diagnostics/collision_x_gaussian_train_vs_prediction.csv")
OUT_DIR = Path("outputs/diagnostics")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dense_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = math.sqrt(float(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    rho = pearson(dense_ranks(x), dense_ranks(y))
    p_value = float("nan")
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        p_value = float(result.pvalue)
    except Exception:
        pass
    return rho, p_value


def metric_arrays(rows: list[dict[str, str]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row[f"gaussian_train_{metric}"]) for row in rows], dtype=np.float64),
        np.asarray([float(row[f"prediction_{metric}"]) for row in rows], dtype=np.float64),
    )


def short_label(row: dict[str, str]) -> str:
    scene = row["run"].split("scene_")[-1][:4]
    if row["object"]:
        return f"{scene}{'A' if row['object'] == 'Object A' else 'B'}"
    return scene


def fmt_num(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100 or (abs(value) < 0.01 and value != 0):
        return f"{value:.2e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def fmt(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def pad_limits(values: np.ndarray, frac: float = 0.08) -> tuple[float, float]:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if abs(vmax - vmin) < 1e-12:
        return vmin - 1.0, vmax + 1.0
    pad = (vmax - vmin) * frac
    return vmin - pad, vmax + pad


def add_panel(
    ax: plt.Axes,
    *,
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> dict[str, object]:
    x, y = metric_arrays(rows, metric)
    rho, p_value = spearman(x, y)

    ax.scatter(
        x,
        y,
        s=42,
        color="#1f77b4",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        label="scene/object",
    )
    for row, xi, yi in zip(rows, x, y):
        ax.annotate(
            short_label(row),
            (float(xi), float(yi)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
        )

    if len(x) >= 2 and np.ptp(x) > 1e-12:
        slope, intercept = np.polyfit(x, y, deg=1)
        xfit = np.linspace(float(np.min(x)), float(np.max(x)), 120)
        ax.plot(
            xfit,
            slope * xfit + intercept,
            color="#d62728",
            linewidth=1.8,
            alpha=0.9,
            label="linear trend",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*pad_limits(x))
    ax.set_ylim(*pad_limits(y, frac=0.12))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.text(
        0.04,
        0.96,
        f"Spearman rho = {rho:.3f}\np = {fmt_num(p_value)}\nn = {len(x)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#c7c7c7",
            "alpha": 0.92,
        },
    )
    return {
        "comparison": title,
        "metric": metric,
        "n": int(len(x)),
        "spearman_rho": rho,
        "spearman_p_value": p_value,
    }


def write_correlations(rows: list[dict[str, object]]) -> None:
    csv_path = OUT_DIR / "axis_metric_spearman_correlations.csv"
    json_path = OUT_DIR / "axis_metric_spearman_correlations.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bounce = read_rows(BOUNCE_CSV)
    collision = read_rows(COLLISION_CSV)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), constrained_layout=True)
    panels = [
        (axes[0, 0], bounce, "rmse_m", "Ball Bounce RMSE", "Gaussian train RMSE (m)", "Prediction RMSE (m)"),
        (axes[0, 1], bounce, "r2", "Ball Bounce Position R²", "Gaussian train R²", "Prediction R²"),
        (axes[1, 0], collision, "rmse_m", "Collision RMSE", "Gaussian train RMSE (m)", "Prediction RMSE (m)"),
        (
            axes[1, 1],
            collision,
            "r2",
            "Collision Velocity R²",
            "Gaussian train velocity R²",
            "Prediction step6 velocity R²",
        ),
    ]
    summaries = [
        add_panel(ax, rows=rows, metric=metric, title=title, xlabel=xlabel, ylabel=ylabel)
        for ax, rows, metric, title, xlabel, ylabel in panels
    ]
    fig.suptitle("Gaussian Train Fit vs Prediction Pipeline Metrics", fontsize=15, fontweight="bold")
    svg_path = OUT_DIR / "axis_metric_spearman_scatter.svg"
    png_path = OUT_DIR / "axis_metric_spearman_scatter.png"
    fig.savefig(svg_path)
    fig.savefig(png_path)
    plt.close(fig)
    write_correlations(summaries)
    print(f"Wrote {svg_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {OUT_DIR / 'axis_metric_spearman_correlations.csv'}")


if __name__ == "__main__":
    main()
