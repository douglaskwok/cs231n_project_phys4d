#!/usr/bin/env python3
"""Validate A/B collision episodes from PyBullet object pose CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_rows(path: Path) -> dict[int, dict[str, dict[str, float]]]:
    frames: dict[int, dict[str, dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            obj = row["object_name"]
            frames.setdefault(frame, {})[obj] = {
                "time_s": float(row["time_s"]),
                "x_m": float(row["x_m"]),
                "y_m": float(row["y_m"]),
                "z_m": float(row["z_m"]),
                "vx_m_s": float(row["vx_m_s"]),
                "vy_m_s": float(row["vy_m_s"]),
            }
    return frames


def _episodes(flags: list[bool], *, min_gap: int, min_len: int) -> list[tuple[int, int]]:
    raw: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                raw.append((start, idx - 1))
            start = None
    if start is not None and len(flags) - start >= min_len:
        raw.append((start, len(flags) - 1))

    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def validate_scene(
    scene_dir: Path,
    *,
    contact_threshold_m: float,
    y_threshold_m: float,
    min_gap_frames: int,
    min_contact_frames: int,
) -> dict:
    pose_path = scene_dir / "object_poses.csv"
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    frames = _load_rows(pose_path)
    frame_ids = sorted(frames)
    contact_flags: list[bool] = []
    min_abs_dx = float("inf")
    min_abs_dy = float("inf")
    for frame in frame_ids:
        row = frames[frame]
        a = row.get("object_a")
        b = row.get("object_b")
        if a is None or b is None:
            contact_flags.append(False)
            continue
        dx = abs(a["x_m"] - b["x_m"])
        dy = abs(a["y_m"] - b["y_m"])
        min_abs_dx = min(min_abs_dx, dx)
        min_abs_dy = min(min_abs_dy, dy)
        contact_flags.append(dx <= contact_threshold_m and dy <= y_threshold_m)

    eps = _episodes(contact_flags, min_gap=min_gap_frames, min_len=min_contact_frames)
    times = []
    for start, end in eps:
        start_frame = frame_ids[start]
        end_frame = frame_ids[end]
        times.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_s": frames[start_frame]["object_a"]["time_s"],
                "end_s": frames[end_frame]["object_a"]["time_s"],
            }
        )

    return {
        "scene": scene_dir.name,
        "ok": len(eps) >= 2,
        "num_contact_episodes": len(eps),
        "contact_episodes": times,
        "min_abs_dx_m": min_abs_dx,
        "min_abs_dy_m": min_abs_dy,
        "contact_threshold_m": contact_threshold_m,
        "y_threshold_m": y_threshold_m,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Collision variant root containing scene_* folders.")
    parser.add_argument("--contact-threshold-m", type=float, default=0.43)
    parser.add_argument("--y-threshold-m", type=float, default=0.43)
    parser.add_argument("--min-gap-frames", type=int, default=8)
    parser.add_argument(
        "--min-contact-frames",
        type=int,
        default=1,
        help="Minimum sampled frames for a contact episode. Use 1 for sharp 60fps rigid impacts.",
    )
    parser.add_argument("--write-json", type=Path, default=None)
    args = parser.parse_args()

    scene_dirs = sorted(path for path in args.root.glob("scene_*") if path.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No scene_* directories under {args.root}")
    report = [
        validate_scene(
            scene_dir,
            contact_threshold_m=args.contact_threshold_m,
            y_threshold_m=args.y_threshold_m,
            min_gap_frames=args.min_gap_frames,
            min_contact_frames=args.min_contact_frames,
        )
        for scene_dir in scene_dirs
    ]
    for row in report:
        status = "OK" if row["ok"] else "FAIL"
        windows = ", ".join(
            f"{ep['start_s']:.2f}-{ep['end_s']:.2f}s" for ep in row["contact_episodes"]
        )
        print(
            f"{status} {row['scene']}: {row['num_contact_episodes']} episodes "
            f"[{windows}] min_dx={row['min_abs_dx_m']:.3f}m"
        )
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["ok"] for row in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
