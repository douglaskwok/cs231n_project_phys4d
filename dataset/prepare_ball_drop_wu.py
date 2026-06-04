#!/usr/bin/env python
"""Prepare Wu-labeled ball-drop exports under dataset/outputs for 4DGaussians.

This script does not retrain or render directly. It prepares object-only DyNeRF
exports and a manifest in a clearly labeled sibling folder:

    dataset/outputs/phys4d_final/ball_drop_3x3_60fps_wu/

The heavy steps (training and rasterization) are expected to run on Modal.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OBJECT_ONLY_EXPORT_DIR = REPO_ROOT / "4dgs" / "experiments" / "object_only"
if str(OBJECT_ONLY_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(OBJECT_ONLY_EXPORT_DIR))

from export_object_only_dynerf import export_object_only_dynerf  # type: ignore  # noqa: E402


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _iter_ball_drop_scenes(source_root: Path, scene_glob: str) -> list[Path]:
    scenes = sorted(
        scene
        for scene in source_root.glob(scene_glob)
        if scene.is_dir() and (scene / "config.json").is_file()
    )
    if not scenes:
        raise FileNotFoundError(
            f"No {scene_glob} folders with config.json found under {source_root}"
        )
    return scenes


def _copy_scene_metadata(source_scene: Path, target_scene: Path) -> None:
    # Keep a lightweight copy of scene metadata for reproducibility in the Wu folder.
    for name in ("config.json", "metadata.json", "scenario_manifest.json"):
        src = source_scene / name
        if src.is_file():
            shutil.copy2(src, target_scene / name)


def prepare_ball_drop_wu(
    *,
    source_root: Path,
    target_root: Path,
    mode: str,
    background: str,
    min_mask_pixels: int,
    min_visible_cameras: int,
    drop_invisible_frames: bool,
    trim_empty_time_ends: bool,
    scene_glob: str,
    dry_run: bool,
) -> dict[str, Any]:
    scenes = _iter_ball_drop_scenes(source_root, scene_glob)
    scene_rows: list[dict[str, Any]] = []
    target_root.mkdir(parents=True, exist_ok=True)

    for scene_path in scenes:
        scene_name = scene_path.name
        target_scene = target_root / scene_name
        export_dir = target_scene / "dynerf_object_wu_train_only"
        masks_root = scene_path / "masks"
        config_path = scene_path / "config.json"

        row: dict[str, Any] = {
            "scene": scene_name,
            "source_scene": _repo_rel(scene_path),
            "target_scene": _repo_rel(target_scene),
            "wu_export": _repo_rel(export_dir),
            "config": _repo_rel(config_path),
            "masks_root": _repo_rel(masks_root),
        }

        if dry_run:
            scene_rows.append(row)
            continue

        target_scene.mkdir(parents=True, exist_ok=True)
        _copy_scene_metadata(scene_path, target_scene)

        meta = export_object_only_dynerf(
            config=config_path.resolve(),
            masks_root=masks_root.resolve(),
            output=export_dir.resolve(),
            background=background,
            mode=mode,
            min_mask_pixels=min_mask_pixels,
            min_visible_cameras=min_visible_cameras,
            trim_empty_time_ends=trim_empty_time_ends,
            drop_invisible_frames=drop_invisible_frames,
            drop_invisible_views=False,
            frame_list=None,
            all_train=False,
        )
        row["export_meta"] = meta
        scene_rows.append(row)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_label": "ball_drop_3x3_60fps_wu",
        "backend_target": "hustvl/4DGaussians",
        "source_dataset": _repo_rel(source_root),
        "target_dataset": _repo_rel(target_root),
        "notes": [
            "DyNeRF export format is backend-agnostic and reused here for Wu 4DGaussians.",
            "Exports are train/test split aware and keep all-train disabled for held-out evaluation.",
            "Run heavy training and rasterization on Modal using the per-scene export folders.",
        ],
        "export_params": {
            "mode": mode,
            "background": background,
            "min_mask_pixels": min_mask_pixels,
            "min_visible_cameras": min_visible_cameras,
            "drop_invisible_frames": drop_invisible_frames,
            "trim_empty_time_ends": trim_empty_time_ends,
            "all_train": False,
        },
        "scene_glob": scene_glob,
        "modal_commands": {
            "train_template": (
                "modal run modal_app.py --train-4d "
                "--train-4d-config room_physics_4dgs_4p0s.yaml "
                "--train-4d-model 4dgs_<scene>_wu"
            ),
            "render_template": (
                "modal run modal_app.py --render-4d "
                "--render-4d-model 4dgs_<scene>_wu "
                "--render-4d-checkpoint chkpnt30000.pth"
            ),
        },
        "num_scenes": len(scene_rows),
        "scenes": scene_rows,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "outputs" / "phys4d_final" / "ball_drop_3x3_60fps",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "outputs" / "phys4d_final" / "ball_drop_3x3_60fps_wu",
    )
    parser.add_argument("--mode", choices=("object", "background", "full"), default="object")
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument("--min-mask-pixels", type=int, default=50)
    parser.add_argument("--min-visible-cameras", type=int, default=6)
    parser.add_argument("--drop-invisible-frames", action="store_true")
    parser.add_argument("--trim-empty-time-ends", action="store_true")
    parser.add_argument(
        "--scene-glob",
        default="scene_*",
        help="Glob pattern under source-root used to select scenes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = prepare_ball_drop_wu(
        source_root=args.source_root.resolve(),
        target_root=args.target_root.resolve(),
        mode=args.mode,
        background=args.background,
        min_mask_pixels=args.min_mask_pixels,
        min_visible_cameras=args.min_visible_cameras,
        drop_invisible_frames=args.drop_invisible_frames,
        trim_empty_time_ends=args.trim_empty_time_ends,
        scene_glob=args.scene_glob,
        dry_run=args.dry_run,
    )

    print(json.dumps(manifest, indent=2))
    if not args.dry_run:
        manifest_path = args.target_root.resolve() / "wu_dataset_manifest.json"
        _write_json(manifest_path, manifest)
        print(f"Wrote Wu ball-drop dataset manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
