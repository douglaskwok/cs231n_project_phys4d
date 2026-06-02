"""Index 4DGS render PNGs by time (frame) and camera."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

RENDER_PNG_RE = re.compile(r"^(\d+)_(?:images_)?cam(\d+)_(\d+)\.png$")
ORBIT_PNG_RE = re.compile(r"^(\d+)_orbit_(\d+)\.png$")
SEQUENTIAL_PNG_RE = re.compile(r"^(\d+)\.png$")


def _load_frame_times(dataset: Path) -> dict[tuple[int, int], float]:
    """Map (cam_id, frame_idx) -> time in seconds from DyNeRF transforms."""
    out: dict[tuple[int, int], float] = {}
    for name in ("transforms_train.json", "transforms_test.json"):
        path = dataset / name
        if not path.is_file():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        for fr in blob.get("frames", []):
            fp = fr.get("file_path", "")
            m = re.search(r"cam(\d+)_(\d+)", fp)
            if not m:
                continue
            cam_id, frame_idx = int(m.group(1)), int(m.group(2))
            out[(cam_id, frame_idx)] = float(fr.get("time", frame_idx))
    return out


def _load_transform_order(dataset: Path) -> list[tuple[int, int, float]]:
    """Return train/test frames in transform order as (cam_id, frame_idx, time)."""
    out: list[tuple[int, int, float]] = []
    for name in ("transforms_train.json", "transforms_test.json"):
        path = dataset / name
        if not path.is_file():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        for fr in blob.get("frames", []):
            fp = fr.get("file_path", "")
            m = re.search(r"cam(\d+)_(\d+)", fp)
            if not m:
                continue
            cam_id, frame_idx = int(m.group(1)), int(m.group(2))
            out.append((cam_id, frame_idx, float(fr.get("time", frame_idx))))
    return out


def index_render_dir(
    render_dir: Path,
    *,
    dataset: Path | None = None,
    rel_prefix: str = "",
) -> dict:
    """Build a manifest for the time viewer.

    Returns a JSON-serializable dict with frames sorted by (time_s, frame_idx).
    """
    render_dir = render_dir.resolve()
    frame_times = _load_frame_times(dataset.resolve()) if dataset else {}
    transform_order = _load_transform_order(dataset.resolve()) if dataset else []

    by_frame: dict[int, dict[int, str]] = defaultdict(dict)
    orbit_rows: list[tuple[int, int, str]] = []
    sequential_rows: list[tuple[int, str]] = []
    mode = "multiview"

    for path in sorted(render_dir.glob("*.png")):
        m = RENDER_PNG_RE.match(path.name)
        if m:
            _seq, cam_id, frame_idx = map(int, m.groups())
            rel = f"{rel_prefix}{path.name}" if rel_prefix else path.name
            by_frame[frame_idx][cam_id] = rel
            continue
        om = ORBIT_PNG_RE.match(path.name)
        if om:
            seq_idx, orbit_idx = map(int, om.groups())
            rel = f"{rel_prefix}{path.name}" if rel_prefix else path.name
            orbit_rows.append((orbit_idx, seq_idx, rel))
            continue
        sm = SEQUENTIAL_PNG_RE.match(path.name)
        if sm:
            seq_idx = int(sm.group(1))
            rel = f"{rel_prefix}{path.name}" if rel_prefix else path.name
            sequential_rows.append((seq_idx, rel))

    frames_out: list[dict] = []
    if by_frame:
        camera_ids = sorted({cam for cams in by_frame.values() for cam in cams})
        for frame_idx in sorted(by_frame):
            cams = by_frame[frame_idx]
            times = [
                frame_times.get((cam, frame_idx))
                for cam in cams
                if (cam, frame_idx) in frame_times
            ]
            time_s = float(times[0]) if times else float(frame_idx)
            frames_out.append(
                {
                    "index": frame_idx,
                    "time_s": time_s,
                    "cams": {str(k): v for k, v in sorted(cams.items())},
                }
            )
        frames_out.sort(key=lambda row: (row["time_s"], row["index"]))
        mode = "multiview"
    elif sequential_rows and transform_order:
        sequential_rows.sort(key=lambda row: row[0])
        camera_ids = sorted({cam_id for cam_id, _frame_idx, _time_s in transform_order})
        for (_seq_idx, rel), (cam_id, frame_idx, time_s) in zip(
            sequential_rows, transform_order
        ):
            by_frame[frame_idx][cam_id] = rel
            frame_times[(cam_id, frame_idx)] = time_s
        for frame_idx in sorted(by_frame):
            cams = by_frame[frame_idx]
            times = [
                frame_times.get((cam, frame_idx))
                for cam in cams
                if (cam, frame_idx) in frame_times
            ]
            time_s = float(times[0]) if times else float(frame_idx)
            frames_out.append(
                {
                    "index": frame_idx,
                    "time_s": time_s,
                    "cams": {str(k): v for k, v in sorted(cams.items())},
                }
            )
        frames_out.sort(key=lambda row: (row["time_s"], row["index"]))
        mode = "wu_multiview"
    elif sequential_rows:
        sequential_rows.sort(key=lambda row: row[0])
        camera_ids = [0]
        for seq_idx, rel in sequential_rows:
            frames_out.append(
                {
                    "index": seq_idx,
                    "time_s": float(seq_idx),
                    "cams": {"0": rel},
                }
            )
        mode = "sequential"
    elif orbit_rows:
        orbit_rows.sort(key=lambda row: (row[0], row[1]))
        camera_ids = [0]
        for orbit_idx, _seq, rel in orbit_rows:
            frames_out.append(
                {
                    "index": orbit_idx,
                    "time_s": float(orbit_idx),
                    "cams": {"0": rel},
                }
            )
        mode = "orbit"

    if not frames_out:
        raise FileNotFoundError(
            f"No 4DGS render PNGs under {render_dir}. "
            "Expected names like 000000_cam00_00000.png or 000000_images_cam00_00000.png"
        )

    gt_frames: list[dict] | None = None
    if dataset and mode in {"multiview", "wu_multiview"}:
        images_dir = dataset.resolve() / "images"
        if images_dir.is_dir():
            gt_by_frame: dict[int, dict[int, str]] = defaultdict(dict)
            for img in images_dir.glob("cam*_*.png"):
                m = re.match(r"cam(\d+)_(\d+)\.png$", img.name)
                if not m:
                    continue
                cam_id, frame_idx = int(m.group(1)), int(m.group(2))
                rel = str(img.relative_to(dataset.resolve()))
                gt_by_frame[frame_idx][cam_id] = rel
            if gt_by_frame:
                gt_frames = []
                for frame_idx in sorted(gt_by_frame):
                    cams = gt_by_frame[frame_idx]
                    times = [
                        frame_times.get((cam, frame_idx))
                        for cam in cams
                        if (cam, frame_idx) in frame_times
                    ]
                    time_s = float(times[0]) if times else float(frame_idx)
                    gt_frames.append(
                        {
                            "index": frame_idx,
                            "time_s": time_s,
                            "cams": {str(k): v for k, v in sorted(cams.items())},
                        }
                    )
                gt_frames.sort(key=lambda row: (row["time_s"], row["index"]))

    return {
        "mode": mode,
        "render_dir": str(render_dir),
        "dataset": str(dataset.resolve()) if dataset else None,
        "camera_ids": camera_ids,
        "frames": frames_out,
        "gt_frames": gt_frames,
    }
