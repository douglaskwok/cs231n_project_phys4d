#!/usr/bin/env python
"""Generate the final Phys4D PyBullet dataset layout.

This script is intentionally a thin orchestrator around the existing 12-view
exporters. It creates the clean final dataset tree:

    dataset/outputs/phys4d_final/
      ball_drop_3x3_60fps/
      collision_base_60fps/
      stacking_base_60fps/
      deformable_base_60fps/

The source truth remains PyBullet RGB/mask/camera data. 4DGS outputs should live
under ``4dgs/experiments`` and be treated as derived artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.export_ping_pong_12view import (  # noqa: E402
    BALL_MASS,
    BALL_RADIUS,
    BALL_SIDE_START_X,
    BALL_START_HEIGHT_ABOVE_SURFACE,
    DATASET_OUTPUTS_ROOT,
    SIM_HZ,
    TABLE_LENGTH,
    TABLE_THICKNESS,
    TABLE_WIDTH,
    _repo_path,
    _scene_name,
    _simulate_variation_scene,
    _steps_per_frame,
)
from dataset.export_room_physics_12view import (  # noqa: E402
    SCENARIO_DEFAULTS,
    simulate_scenario,
)

FINAL_ROOT = DATASET_OUTPUTS_ROOT / "phys4d_final"
FINAL_FPS = 60.0
BALL_DROP_RESTITUTIONS = [0.87, 0.90, 0.93]
BALL_DROP_ANGLES_DEG = [-5.0, 0.0, 5.0]
BALL_DROP_DURATION_SEC = 4.0
ROOM_SCENARIOS = ["collision", "stacking", "deformable"]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _camera_names(raw: str) -> set[str] | None:
    value = raw.strip()
    if value.lower() == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _ball_drop_variants() -> list[dict[str, float]]:
    variants = []
    for restitution in BALL_DROP_RESTITUTIONS:
        for angle in BALL_DROP_ANGLES_DEG:
            variants.append(
                {
                    "restitution": float(restitution),
                    "ball_angle_deg": float(angle),
                }
            )
    return variants


def generate_ball_drop(
    *,
    out_dir: Path,
    video_fps: float,
    sim_hz: float,
    duration_sec: float,
    max_frames: int | None,
    render_camera_names: set[str] | None,
    video_camera_names: set[str] | None,
    overwrite: bool,
    write_videos: bool,
    dry_run: bool,
) -> dict[str, Any]:
    variants = _ball_drop_variants()
    scenes = []
    for idx, params in enumerate(variants):
        scene_id = _scene_name(idx, params["restitution"], params["ball_angle_deg"])
        scene_dir = out_dir / scene_id
        row = {
            "scene_id": scene_id,
            "path": _repo_path(scene_dir),
            "restitution": params["restitution"],
            "ball_angle_deg": params["ball_angle_deg"],
            "config": _repo_path(scene_dir / "config.json"),
            "metadata": _repo_path(scene_dir / "metadata.json"),
        }
        scenes.append(row)
        if dry_run:
            continue
        print(
            f"ball_drop [{idx + 1}/{len(variants)}] {scene_id} "
            f"e={params['restitution']:.2f} angle={params['ball_angle_deg']:.1f}",
            flush=True,
        )
        _simulate_variation_scene(
            scene_dir=scene_dir,
            restitution=params["restitution"],
            ball_angle_deg=params["ball_angle_deg"],
            ball_radius_m=BALL_RADIUS,
            environment="room",
            video_fps=video_fps,
            sim_hz=sim_hz,
            duration_sec=duration_sec,
            render_camera_names=render_camera_names,
            max_frames=max_frames,
            overwrite=overwrite,
            write_videos=write_videos,
            video_camera_names=video_camera_names,
        )

    manifest = {
        "scenario": "ball_drop",
        "scenario_dir": _repo_path(out_dir),
        "num_scenes": len(scenes),
        "fps": video_fps,
        "sim_hz": sim_hz,
        "duration_sec": duration_sec,
        "steps_per_frame": _steps_per_frame(video_fps, sim_hz),
        "param_names": ["restitution", "ball_angle_deg"],
        "param_grid": {
            "restitution": BALL_DROP_RESTITUTIONS,
            "ball_angle_deg": BALL_DROP_ANGLES_DEG,
        },
        "fixed_params": {
            "ball_radius_m": BALL_RADIUS,
            "ball_mass_kg": BALL_MASS,
            "drop_height_above_surface_m": BALL_START_HEIGHT_ABOVE_SURFACE,
            "angled_drop_start_x_abs_m": BALL_SIDE_START_X,
            "table_size_m": [TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS],
            "environment": "room",
            "drag_enabled": False,
        },
        "videos_written": write_videos,
        "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
        "scenes": scenes,
    }
    if not dry_run:
        _write_json(out_dir / "scenario_manifest.json", manifest)
    return manifest


def generate_room_base(
    *,
    scenario: str,
    out_dir: Path,
    video_fps: float,
    max_frames: int | None,
    render_camera_names: set[str] | None,
    video_camera_names: set[str] | None,
    overwrite: bool,
    write_videos: bool,
    dry_run: bool,
) -> dict[str, Any]:
    duration_sec = float(SCENARIO_DEFAULTS[scenario]["duration_sec"])
    sim_hz = float(SCENARIO_DEFAULTS[scenario]["sim_hz"])
    scene_id = f"scene_0000_{scenario}_room"
    scene_dir = out_dir / scene_id
    scene = {
        "scene_id": scene_id,
        "path": _repo_path(scene_dir),
        "scenario": scenario,
        "config": _repo_path(scene_dir / "config.json"),
        "metadata": _repo_path(scene_dir / "metadata.json"),
    }
    if not dry_run:
        print(f"{scenario} base -> {scene_id}", flush=True)
        simulate_scenario(
            scenario=scenario,
            scene_dir=scene_dir,
            video_fps=video_fps,
            sim_hz=sim_hz,
            duration_sec=duration_sec,
            max_frames=max_frames,
            render_camera_names=render_camera_names,
            video_camera_names=video_camera_names,
            overwrite=overwrite,
            write_videos=write_videos,
        )

    manifest = {
        "scenario": scenario,
        "scenario_dir": _repo_path(out_dir),
        "num_scenes": 1,
        "fps": video_fps,
        "sim_hz": sim_hz,
        "duration_sec": duration_sec,
        "steps_per_frame": _steps_per_frame(video_fps, sim_hz),
        "videos_written": write_videos,
        "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
        "scenes": [scene],
    }
    if not dry_run:
        _write_json(out_dir / "scenario_manifest.json", manifest)
    return manifest


def generate_final_dataset(
    *,
    output_dir: Path,
    video_fps: float,
    max_frames: int | None,
    render_camera_names: set[str] | None,
    video_camera_names: set[str] | None,
    overwrite: bool,
    write_videos: bool,
    dry_run: bool,
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    scenarios["ball_drop_3x3_60fps"] = generate_ball_drop(
        out_dir=output_dir / "ball_drop_3x3_60fps",
        video_fps=video_fps,
        sim_hz=SIM_HZ,
        duration_sec=BALL_DROP_DURATION_SEC,
        max_frames=max_frames,
        render_camera_names=render_camera_names,
        video_camera_names=video_camera_names,
        overwrite=overwrite,
        write_videos=write_videos,
        dry_run=dry_run,
    )

    for scenario in ROOM_SCENARIOS:
        scenarios[f"{scenario}_base_60fps"] = generate_room_base(
            scenario=scenario,
            out_dir=output_dir / f"{scenario}_base_60fps",
            video_fps=video_fps,
            max_frames=max_frames,
            render_camera_names=render_camera_names,
            video_camera_names=video_camera_names,
            overwrite=overwrite,
            write_videos=write_videos,
            dry_run=dry_run,
        )

    total_scenes = sum(int(item["num_scenes"]) for item in scenarios.values())
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": "phys4d_final",
        "dataset_root": _repo_path(output_dir),
        "fps": video_fps,
        "num_cameras": 12,
        "image_size": [960, 544],
        "total_scenes": total_scenes,
        "videos_written": write_videos,
        "scenarios": {
            name: {
                "scenario_dir": item["scenario_dir"],
                "num_scenes": item["num_scenes"],
                "manifest": _repo_path(output_dir / name / "scenario_manifest.json"),
            }
            for name, item in scenarios.items()
        },
    }
    if not dry_run:
        _write_json(output_dir / "dataset_manifest.json", manifest)
    return {"dataset_manifest": manifest, "scenario_manifests": scenarios}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FINAL_ROOT)
    parser.add_argument("--video-fps", type=float, default=FINAL_FPS)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--render-camera-names", default="all")
    parser.add_argument("--video-camera-names", default="all")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")

    manifest = generate_final_dataset(
        output_dir=args.output_dir.resolve(),
        video_fps=args.video_fps,
        max_frames=args.max_frames,
        render_camera_names=_camera_names(args.render_camera_names),
        video_camera_names=_camera_names(args.video_camera_names),
        overwrite=not args.no_overwrite,
        write_videos=not args.no_videos,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))
    if not args.dry_run:
        print(f"Wrote final PyBullet dataset: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
