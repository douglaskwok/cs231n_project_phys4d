"""Canonical time-based train/test split for bounce-pipeline datasets.

All scenes use the same fraction of GT timeline frames for train vs held-out test,
computed from ``object_poses.csv`` (or any contiguous frame range).
"""

from __future__ import annotations

import csv
from pathlib import Path

TRAIN_FRAC = 0.65
TEST_FRAC = 0.35


def frame_range_from_gt_poses(gt_poses_csv: Path) -> tuple[int, int]:
    """Return inclusive ``(frame_min, frame_max)`` from PyBullet ``object_poses.csv``."""

    gt_poses_csv = gt_poses_csv.resolve()
    frames: list[int] = []
    with gt_poses_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frames.append(int(float(row["frame"])))
    if not frames:
        raise ValueError(f"No frames in {gt_poses_csv}")
    return min(frames), max(frames)


def compute_time_split(
    frame_min: int,
    frame_max: int,
    *,
    train_frac: float = TRAIN_FRAC,
) -> dict[str, object]:
    """Split a contiguous frame timeline into train / test (inclusive indices).

    Example: 361 frames (0..360) at ``train_frac=0.65`` → train 0..234, test 235..360.
    """

    if frame_max < frame_min:
        raise ValueError(f"frame_max ({frame_max}) < frame_min ({frame_min})")
    n = int(frame_max - frame_min + 1)
    if n < 2:
        raise ValueError(f"Need at least 2 frames for a train/test split, got {n}")

    n_train = int(round(n * float(train_frac)))
    n_train = max(1, min(n - 1, n_train))
    train_start = int(frame_min)
    train_end = int(frame_min + n_train - 1)
    test_start = int(train_end + 1)
    test_end = int(frame_max)
    n_test = int(test_end - test_start + 1)

    return {
        "train_frac": float(train_frac),
        "test_frac": float(1.0 - train_frac),
        "frame_min": int(frame_min),
        "frame_max": int(frame_max),
        "num_frames": n,
        "train": [train_start, train_end],
        "test": [test_start, test_end],
        "n_train": n_train,
        "n_test": n_test,
    }


def split_from_gt_poses(
    gt_poses_csv: Path,
    *,
    train_frac: float = TRAIN_FRAC,
) -> dict[str, object]:
    """Convenience: read GT pose CSV and return the canonical split."""

    frame_min, frame_max = frame_range_from_gt_poses(gt_poses_csv)
    out = compute_time_split(frame_min, frame_max, train_frac=train_frac)
    out["gt_poses_csv"] = str(gt_poses_csv.resolve())
    return out
