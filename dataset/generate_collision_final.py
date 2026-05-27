#!/usr/bin/env python
"""Generate the final 2x2x2 collision variant PyBullet dataset.

Three physical parameters are swept in a 2x2x2 grid:

  mass_a_kg      – mass of object_a (object_b is fixed at 0.160 kg)
  velocity_scale – scalar applied to both objects' initial approach speeds
  restitution    – coefficient of restitution between the two objects

This mirrors the ball-drop structure in ``generate_phys4d_final.py`` and
produces 8 scenes under:

    dataset/outputs/phys4d_final/collision_2x2x2_60fps/
      scene_0000_col_m100_v060_r025/
      ...
      scene_0007_col_m440_v140_r075/
      scenario_manifest.json
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
    DATASET_OUTPUTS_ROOT,
    TABLE_LENGTH,
    TABLE_THICKNESS,
    TABLE_WIDTH,
    _parse_video_camera_names,
    _repo_path,
    _steps_per_frame,
)
from dataset.export_room_physics_12view import (  # noqa: E402
    COLLISION_OBJECT_HALF_EXTENTS_M,
    COLLISION_TABLE_TOP_Z,
    SCENARIO_DEFAULTS,
    simulate_scenario,
)

FINAL_ROOT = DATASET_OUTPUTS_ROOT / "phys4d_final"
DEFAULT_OUTPUT_DIR = FINAL_ROOT / "collision_2x2x2_60fps"
DEFAULT_VIDEO_FPS = 60.0
SCENARIO = "collision"

COLLISION_MASS_A_KG = [0.100, 0.440]
COLLISION_VELOCITY_SCALES = [0.6, 1.4]
COLLISION_RESTITUTIONS = [0.25, 0.75]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _collision_scene_name(idx: int, mass_a_kg: float, velocity_scale: float, restitution: float) -> str:
    m_str = f"{int(round(mass_a_kg * 1000)):03d}"
    v_str = f"{int(round(velocity_scale * 100)):03d}"
    r_str = f"{int(round(restitution * 100)):03d}"
    return f"scene_{idx:04d}_col_m{m_str}_v{v_str}_r{r_str}"


def _collision_variants() -> list[dict[str, float]]:
    variants = []
    for mass_a_kg in COLLISION_MASS_A_KG:
        for velocity_scale in COLLISION_VELOCITY_SCALES:
            for restitution in COLLISION_RESTITUTIONS:
                variants.append(
                    {
                        "mass_a_kg": float(mass_a_kg),
                        "velocity_scale": float(velocity_scale),
                        "restitution": float(restitution),
                    }
                )
    return variants


def generate_collision_final(
    *,
    output_dir: Path,
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
    variants = _collision_variants()
    scenes = []
    for idx, params in enumerate(variants):
        scene_id = _collision_scene_name(idx, params["mass_a_kg"], params["velocity_scale"], params["restitution"])
        scene_dir = output_dir / scene_id
        row = {
            "scene_id": scene_id,
            "path": _repo_path(scene_dir),
            "scenario": SCENARIO,
            "mass_a_kg": params["mass_a_kg"],
            "velocity_scale": params["velocity_scale"],
            "restitution": params["restitution"],
            "config": _repo_path(scene_dir / "config.json"),
            "metadata": _repo_path(scene_dir / "metadata.json"),
            "object_poses": _repo_path(scene_dir / "object_poses.csv"),
        }
        scenes.append(row)
        if dry_run:
            continue
        print(
            f"collision [{idx + 1}/{len(variants)}] {scene_id} "
            f"m_a={params['mass_a_kg']:.3f}kg vel={params['velocity_scale']:.1f} e={params['restitution']:.2f}",
            flush=True,
        )
        simulate_scenario(
            scenario=SCENARIO,
            scene_dir=scene_dir,
            video_fps=video_fps,
            sim_hz=sim_hz,
            duration_sec=duration_sec,
            max_frames=max_frames,
            render_camera_names=render_camera_names,
            video_camera_names=video_camera_names,
            overwrite=overwrite,
            write_videos=write_videos,
            collision_mass_a_kg=params["mass_a_kg"],
            collision_velocity_scale=params["velocity_scale"],
            collision_restitution=params["restitution"],
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": SCENARIO,
        "scenario_dir": _repo_path(output_dir),
        "num_scenes": len(scenes),
        "fps": video_fps,
        "sim_hz": sim_hz,
        "duration_sec": duration_sec,
        "steps_per_frame": _steps_per_frame(video_fps, sim_hz),
        "num_cameras": 12,
        "image_size": [960, 544],
        "videos_written": write_videos,
        "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
        "param_names": ["mass_a_kg", "velocity_scale", "restitution"],
        "param_grid": {
            "mass_a_kg": COLLISION_MASS_A_KG,
            "velocity_scale": COLLISION_VELOCITY_SCALES,
            "restitution": COLLISION_RESTITUTIONS,
        },
        "fixed_params": {
            "environment": "room",
            "object_count": 2,
            "object_kind": "rigid_box",
            "object_full_size_m": [2.0 * value for value in COLLISION_OBJECT_HALF_EXTENTS_M],
            "object_b_mass_kg": 0.160,
            "object_lateral_friction": 0.02,
            "baseline_vel_a_m_s": 0.95,
            "baseline_vel_b_m_s": 0.75,
            "table_top_z_m": COLLISION_TABLE_TOP_Z,
            "table_size_m": [TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS],
        },
        "scenes": scenes,
    }

    if not dry_run:
        _write_json(output_dir / "scenario_manifest.json", manifest)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--sim-hz", type=float, default=float(SCENARIO_DEFAULTS[SCENARIO]["sim_hz"]))
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=float(SCENARIO_DEFAULTS[SCENARIO]["duration_sec"]),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--render-camera-names", default="all")
    parser.add_argument("--video-camera-names", default="all")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    if args.sim_hz <= 0:
        raise ValueError("--sim-hz must be positive.")
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")

    manifest = generate_collision_final(
        output_dir=args.output_dir.resolve(),
        video_fps=args.video_fps,
        sim_hz=args.sim_hz,
        duration_sec=args.duration_sec,
        max_frames=args.max_frames,
        render_camera_names=_parse_video_camera_names(args.render_camera_names),
        video_camera_names=_parse_video_camera_names(args.video_camera_names),
        overwrite=not args.no_overwrite,
        write_videos=not args.no_videos,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))
    if not args.dry_run:
        print(f"Wrote final collision dataset: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
