#!/usr/bin/env python
"""Generate only the final 12-view collision PyBullet dataset.

This is a narrow companion to ``generate_phys4d_final.py`` for when we want to
refresh collision without regenerating ball drop, stacking, or deformable data.
It writes:

    dataset/outputs/phys4d_final/collision_base_60fps/
      scene_0000_collision_room/
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
DEFAULT_OUTPUT_DIR = FINAL_ROOT / "collision_base_60fps"
DEFAULT_VIDEO_FPS = 60.0
SCENARIO = "collision"
SCENE_ID = "scene_0000_collision_room"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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
    geometry_scale: float,
    wall_height_scale: float,
    side_camera_extra_radius: float,
    side_camera_extra_height: float,
    velocity_scale: float,
) -> dict[str, Any]:
    scene_dir = output_dir / SCENE_ID
    scene = {
        "scene_id": SCENE_ID,
        "path": _repo_path(scene_dir),
        "scenario": SCENARIO,
        "config": _repo_path(scene_dir / "config.json"),
        "metadata": _repo_path(scene_dir / "metadata.json"),
        "object_poses": _repo_path(scene_dir / "object_poses.csv"),
    }

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": SCENARIO,
        "scenario_dir": _repo_path(output_dir),
        "num_scenes": 1,
        "fps": video_fps,
        "sim_hz": sim_hz,
        "duration_sec": duration_sec,
        "steps_per_frame": _steps_per_frame(video_fps, sim_hz),
        "num_cameras": 12,
        "image_size": [960, 544],
        "videos_written": write_videos,
        "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
        "fixed_params": {
            "environment": "room",
            "object_count": 2,
            "object_kind": "rigid_box",
            "object_full_size_m": [
                2.0 * value * geometry_scale
                for value in COLLISION_OBJECT_HALF_EXTENTS_M
            ],
            "object_geometry_scale": geometry_scale,
            "wall_height_scale": wall_height_scale,
            "side_camera_extra_radius_m": side_camera_extra_radius,
            "side_camera_extra_height_m": side_camera_extra_height,
            "object_a_mass_kg": 0.220,
            "object_b_mass_kg": 0.160,
            "object_restitution": 0.98,
            "object_lateral_friction": 0.002,
            "object_velocity_scale": velocity_scale,
            "table_top_z_m": COLLISION_TABLE_TOP_Z,
            "table_size_m": [TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS],
        },
        "scenes": [scene],
    }

    if dry_run:
        return manifest

    print(f"collision [1/1] {SCENE_ID}", flush=True)
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
        collision_restitution=0.98,
        collision_velocity_scale=velocity_scale,
        collision_geometry_scale=geometry_scale,
        collision_wall_height_scale=wall_height_scale,
        side_camera_extra_radius=side_camera_extra_radius,
        side_camera_extra_height=side_camera_extra_height,
    )
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
    parser.add_argument(
        "--geometry-scale",
        type=float,
        default=1.0,
        help="Scale collision object sizes, wall sizes, and object start offsets. Cameras stay fixed.",
    )
    parser.add_argument(
        "--wall-height-scale",
        type=float,
        default=1.0,
        help="Scale collision rail height separately. Use with --geometry-scale to avoid rail occlusion.",
    )
    parser.add_argument(
        "--side-camera-extra-radius",
        type=float,
        default=0.0,
        help="Move left/right ring cameras outward by this many meters.",
    )
    parser.add_argument(
        "--side-camera-extra-height",
        type=float,
        default=0.0,
        help="Move left/right ring cameras upward by this many meters.",
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="Scale the initial collision object velocities.",
    )
    args = parser.parse_args()

    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    if args.sim_hz <= 0:
        raise ValueError("--sim-hz must be positive.")
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.geometry_scale <= 0:
        raise ValueError("--geometry-scale must be positive.")
    if args.wall_height_scale <= 0:
        raise ValueError("--wall-height-scale must be positive.")
    if args.velocity_scale <= 0:
        raise ValueError("--velocity-scale must be positive.")

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
        geometry_scale=args.geometry_scale,
        wall_height_scale=args.wall_height_scale,
        side_camera_extra_radius=args.side_camera_extra_radius,
        side_camera_extra_height=args.side_camera_extra_height,
        velocity_scale=args.velocity_scale,
    )
    print(json.dumps(manifest, indent=2))
    if not args.dry_run:
        print(f"Wrote final collision dataset: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
