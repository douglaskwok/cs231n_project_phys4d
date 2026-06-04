#!/usr/bin/env python3
"""Merge Janhavi's split collision exports into the phys4d_final layout.

The two source folders contain overlapping scene names, but different folders
are more complete for different scenes/subdirectories. This script chooses the
source with the most PNG frames for each scene/subdir/camera folder and merges
videos without touching the source exports.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DATA_SUBDIRS = (
    "rgb",
    "masks",
    "masks_object_a",
    "masks_object_b",
    "masks_object_table",
    "masks_table",
)

TOP_LEVEL_FILES = (
    "cameras.json",
    "config.json",
    "metadata.json",
    "object_poses.csv",
    "physics_params.json",
    "scenario_manifest.json",
)


def png_count(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png")) if path.is_dir() else 0


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))


def copy_file_best(scene_sources: list[Path], scene_out: Path, report: dict) -> None:
    for name in TOP_LEVEL_FILES:
        candidates = [src / name for src in scene_sources if (src / name).is_file()]
        if not candidates:
            continue
        # Prefer the larger metadata file when both exist; usually they are identical.
        best = max(candidates, key=lambda p: p.stat().st_size)
        shutil.copy2(best, scene_out / name)
        report["top_level_files"][name] = str(best)


def merge_data_subdirs(scene_sources: list[Path], scene_out: Path, report: dict) -> None:
    for subdir in DATA_SUBDIRS:
        chosen_for_subdir: dict[str, dict] = {}
        cameras = sorted(
            {
                cam.name
                for src in scene_sources
                for cam in (src / subdir).glob("cam*")
                if cam.is_dir()
            }
        )
        if not cameras:
            continue
        out_subdir = scene_out / subdir
        out_subdir.mkdir(parents=True, exist_ok=True)
        for cam in cameras:
            candidates = [(src / subdir / cam, src) for src in scene_sources if (src / subdir / cam).is_dir()]
            if not candidates:
                continue
            best_path, best_src = max(candidates, key=lambda item: png_count(item[0]))
            count = png_count(best_path)
            if count == 0:
                continue
            copy_tree(best_path, out_subdir / cam)
            chosen_for_subdir[cam] = {
                "source": str(best_src),
                "frame_count": count,
            }
        if chosen_for_subdir:
            report["data_subdirs"][subdir] = chosen_for_subdir


def merge_videos(scene_sources: list[Path], scene_out: Path, report: dict) -> None:
    video_out = scene_out / "videos"
    chosen: dict[str, dict] = {}
    for src in scene_sources:
        videos = src / "videos"
        if not videos.is_dir():
            continue
        for mp4 in videos.rglob("*.mp4"):
            rel = mp4.relative_to(videos)
            dst = video_out / rel
            if not dst.exists() or mp4.stat().st_size > dst.stat().st_size:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp4, dst)
                chosen[str(rel)] = {"source": str(src), "bytes": mp4.stat().st_size}
    if chosen:
        report["videos"] = chosen


def merge_dataset(sources: list[Path], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    scene_names = sorted(
        {
            scene.name
            for src in sources
            for scene in src.glob("scene_*")
            if scene.is_dir()
        }
    )
    report = {
        "sources": [str(src) for src in sources],
        "out": str(out),
        "scenes": {},
    }
    for scene_name in scene_names:
        scene_sources = [src / scene_name for src in sources if (src / scene_name).is_dir()]
        scene_out = out / scene_name
        scene_out.mkdir(parents=True, exist_ok=True)
        scene_report = {
            "source_dirs": [str(path) for path in scene_sources],
            "top_level_files": {},
            "data_subdirs": {},
            "videos": {},
        }
        copy_file_best(scene_sources, scene_out, scene_report)
        merge_data_subdirs(scene_sources, scene_out, scene_report)
        merge_videos(scene_sources, scene_out, scene_report)
        report["scenes"][scene_name] = scene_report
    (out / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", action="append", required=True, help="Source collision export folder. Repeatable.")
    parser.add_argument("--out", required=True, help="Merged output folder.")
    args = parser.parse_args()

    sources = [Path(src).expanduser().resolve() for src in args.src]
    out = Path(args.out).expanduser().resolve()
    for src in sources:
        if not src.is_dir():
            raise FileNotFoundError(src)

    report = merge_dataset(sources, out)
    print(f"Merged {len(report['scenes'])} scenes -> {out}")
    print(f"Wrote {out / 'merge_report.json'}")
    for scene_name, scene in report["scenes"].items():
        completeish = {
            subdir: len(cams)
            for subdir, cams in scene["data_subdirs"].items()
        }
        print(f"{scene_name}: {completeish}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
