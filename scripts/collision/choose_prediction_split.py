#!/usr/bin/env python3
"""Choose a collision prediction split around the second object-object impact.

The prediction workflow accepts explicit train/test frame ranges. For the walled
collision scenes we usually want an event-aligned split: hold out a short window
starting just before the second impact, regardless of the absolute timestamp.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _episodes(flags: list[bool], frames: list[int], *, min_gap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    prev_idx: int | None = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        if start is not None and prev_idx is not None and idx - prev_idx > min_gap + 1:
            out.append((frames[start], frames[prev_idx]))
            start = idx if flag else None
        if flag:
            prev_idx = idx
        elif start is not None and prev_idx == idx - 1:
            out.append((frames[start], frames[prev_idx]))
            start = None
            prev_idx = None
    if start is not None and prev_idx is not None:
        out.append((frames[start], frames[prev_idx]))
    return out


def _load_poses(path: Path) -> dict[int, dict[int, dict[str, float]]]:
    by_frame: dict[int, dict[int, dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(float(row["frame"]))
            obj = int(float(row["object_index"]))
            by_frame.setdefault(frame, {})[obj] = {
                "time_s": float(row["time_s"]),
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
                "z": float(row["z_m"]),
                "vx": float(row["vx_m_s"]),
                "vy": float(row["vy_m_s"]),
            }
    return by_frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--pre-collision-frames", type=int, default=5)
    ap.add_argument("--post-collision-frames", type=int, default=14)
    ap.add_argument("--contact-slack-m", type=float, default=0.015)
    ap.add_argument("--min-gap-frames", type=int, default=8)
    ap.add_argument(
        "--require-wall-before-second",
        action="store_true",
        help="Fail if no wall touch/extremum is detected before the second impact.",
    )
    ap.add_argument("--json", action="store_true", help="Print JSON instead of shell variables.")
    args = ap.parse_args()

    scene_dir = args.scene_dir.resolve()
    cfg = json.loads((scene_dir / "config.json").read_text(encoding="utf-8"))
    poses = _load_poses(scene_dir / "object_poses.csv")
    frames = sorted(poses)
    if not frames:
        raise ValueError(f"No pose frames under {scene_dir}")

    params = cfg.get("scene", {}).get("scenario_params", {})
    half = params.get("object_half_extents_m") or [0.21, 0.21, 0.13125]
    half_x = float(half[0])
    half_y = float(half[1])
    pair_x = 2.0 * half_x + float(args.contact_slack_m)
    pair_y = 2.0 * half_y + float(args.contact_slack_m)
    wall_x = params.get("wall_half_x_m")
    wall_y = params.get("wall_half_y_m")
    wall_center_x = float(wall_x) - half_x if wall_x is not None else None
    wall_center_y = float(wall_y) - half_y if wall_y is not None else None

    pair_flags: list[bool] = []
    for fr in frames:
        row = poses[fr]
        a = row.get(0)
        b = row.get(1)
        pair_flags.append(
            bool(
                a
                and b
                and abs(a["x"] - b["x"]) <= pair_x
                and abs(a["y"] - b["y"]) <= pair_y
            )
        )
    pair_eps = _episodes(pair_flags, frames, min_gap=args.min_gap_frames)
    if len(pair_eps) < 2:
        raise RuntimeError(f"Need at least two pair contacts; found {pair_eps}")

    first_pair = pair_eps[0]
    second_pair = pair_eps[1]
    second_frame = second_pair[0]
    search_start = first_pair[1] + 1
    search_end = second_pair[0] - 1
    wall_candidates: list[tuple[float, int, int, str]] = []
    for fr in frames:
        if fr < search_start or fr > search_end:
            continue
        for obj, row in poses[fr].items():
            if wall_center_x is not None:
                wall_candidates.append((abs(row["x"]) - wall_center_x, fr, obj, "x"))
            if wall_center_y is not None:
                wall_candidates.append((abs(row["y"]) - wall_center_y, fr, obj, "y"))
    wall_frame: int | None = None
    wall_object: int | None = None
    wall_axis: str | None = None
    if wall_candidates:
        _, wall_frame, wall_object, wall_axis = max(wall_candidates, key=lambda t: t[0])
    elif args.require_wall_before_second:
        raise RuntimeError("No wall candidates found between first and second contact.")

    window_frames = int(args.pre_collision_frames) + int(args.post_collision_frames) + 1
    test_start = max(frames[0] + 1, second_frame - int(args.pre_collision_frames))
    train_start = frames[0]
    test_end = min(frames[-1], test_start + window_frames - 1)
    if test_end - test_start + 1 < window_frames:
        test_start = max(frames[0] + 1, test_end - window_frames + 1)
    train_end = test_start - 1

    out = {
        "scene": scene_dir.name,
        "frame_min": frames[0],
        "frame_max": frames[-1],
        "first_pair_contact": list(first_pair),
        "second_pair_contact": list(second_pair),
        "wall_bounce_before_second": (
            {"frame": wall_frame, "object_index": wall_object, "axis": wall_axis}
            if wall_frame is not None
            else None
        ),
        "train": [train_start, train_end],
        "test": [test_start, test_end],
        "model_last": frames[-1],
        "pre_collision_frames": int(args.pre_collision_frames),
        "post_collision_frames": int(args.post_collision_frames),
        "window_frames": window_frames,
        "train_after_wall_bounce": bool(train_end > wall_frame) if wall_frame is not None else None,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"SCENE_NAME={out['scene']}")
        print(f"TRAIN_START={train_start}")
        print(f"TRAIN_END={train_end}")
        print(f"TEST_START={test_start}")
        print(f"TEST_END={test_end}")
        print(f"MODEL_LAST={frames[-1]}")
        print(f"FIRST_PAIR={first_pair[0]}-{first_pair[1]}")
        print(f"SECOND_PAIR={second_pair[0]}-{second_pair[1]}")
        if wall_frame is not None:
            print(f"WALL_BOUNCE_FRAME={wall_frame}")
            print(f"WALL_BOUNCE_OBJECT={wall_object}")
            print(f"WALL_BOUNCE_AXIS={wall_axis}")
            print(f"TRAIN_AFTER_WALL_BOUNCE={str(train_end > wall_frame).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
