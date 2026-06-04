#!/usr/bin/env python
"""Generate final 12-view collision variant scenes.

This is the multi-scene companion to ``generate_collision_final.py``.  It keeps
the improved collision visibility setup we converged on for Wu 4DGS:

* 12 cameras, including side cameras moved slightly outward/upward
* enlarged collision boxes and rails
* per-object masks for object A and object B
* 2.6 seconds at 60 FPS by default

The default parameter grid mirrors the final collision naming convention:
mass A x velocity distribution x restitution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DEFAULT_OUTPUT_DIR = FINAL_ROOT / "collision_scale1p75_elastic_2x2x2_60fps"
DEFAULT_VIDEO_FPS = 60.0
SCENARIO = "collision"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float value.")
    return values


def _tag(value: float, scale: int = 100) -> str:
    return f"{int(round(value * scale)):03d}"


def _parse_velocity_modes(raw: str) -> list[dict[str, Any]]:
    modes: list[dict[str, Any]] = []
    for part in [value.strip() for value in raw.split(",") if value.strip()]:
        if part.startswith("asym"):
            velocity_scale = float(part.removeprefix("asym"))
            modes.append(
                {
                    "name": f"asym{_tag(velocity_scale, 100)}",
                    "kind": "asymmetric",
                    "velocity_scale": velocity_scale,
                    "velocity_split": None,
                    "object_a_initial_velocity_m_s": 0.70 * velocity_scale,
                    "object_b_initial_velocity_m_s": 0.55 * velocity_scale,
                }
            )
        elif part.startswith("equal"):
            speed = float(part.removeprefix("equal"))
            # Wu/Fudan exporter parameterizes velocity by closing speed:
            # closing = (0.70 + 0.55) * velocity_scale.
            velocity_scale = (2.0 * speed) / (0.70 + 0.55)
            modes.append(
                {
                    "name": f"equal{_tag(speed, 100)}",
                    "kind": "equal_speed",
                    "velocity_scale": velocity_scale,
                    "velocity_split": 0.5,
                    "object_a_initial_velocity_m_s": speed,
                    "object_b_initial_velocity_m_s": speed,
                }
            )
        else:
            raise ValueError(
                f"Unknown velocity mode '{part}'. Use values like 'asym1.4' or 'equal1.05'."
            )
    if not modes:
        raise ValueError("Expected at least one velocity mode.")
    return modes


def _variants(
    masses: list[float],
    velocity_modes: list[dict[str, Any]],
    restitutions: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mass_a in masses:
        for velocity_mode in velocity_modes:
            for restitution in restitutions:
                rows.append(
                    {
                        "mass_a_kg": float(mass_a),
                        "velocity_mode": str(velocity_mode["name"]),
                        "velocity_kind": str(velocity_mode["kind"]),
                        "velocity_scale": float(velocity_mode["velocity_scale"]),
                        "velocity_split": velocity_mode["velocity_split"],
                        "object_a_initial_velocity_m_s": float(
                            velocity_mode["object_a_initial_velocity_m_s"]
                        ),
                        "object_b_initial_velocity_m_s": float(
                            velocity_mode["object_b_initial_velocity_m_s"]
                        ),
                        "restitution": float(restitution),
                    }
                )
    return rows


def _generate_scene_job(job: dict[str, Any]) -> str:
    params = job["params"]
    scene_id = str(job["scene_id"])
    print(
        f"collision {job['progress_label']} {scene_id} "
        f"m={params['mass_a_kg']:.3f} "
        f"mode={params['velocity_mode']} "
        f"vA={params['object_a_initial_velocity_m_s']:.2f} "
        f"vB={params['object_b_initial_velocity_m_s']:.2f} "
        f"r={params['restitution']:.2f}",
        flush=True,
    )
    simulate_scenario(
        scenario=SCENARIO,
        scene_dir=Path(job["scene_dir"]),
        video_fps=float(job["video_fps"]),
        sim_hz=float(job["sim_hz"]),
        duration_sec=float(job["duration_sec"]),
        max_frames=job["max_frames"],
        render_camera_names=job["render_camera_names"],
        video_camera_names=job["video_camera_names"],
        overwrite=bool(job["overwrite"]),
        write_videos=bool(job["write_videos"]),
        collision_mass_a_kg=params["mass_a_kg"],
        collision_restitution=params["restitution"],
        collision_velocity_scale=params["velocity_scale"],
        collision_velocity_split=params["velocity_split"],
        collision_geometry_scale=float(job["geometry_scale"]),
        collision_wall_height_scale=float(job["wall_height_scale"]),
        side_camera_extra_radius=float(job["side_camera_extra_radius"]),
        side_camera_extra_height=float(job["side_camera_extra_height"]),
    )
    return scene_id


def generate_collision_variants(
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
    masses: list[float],
    velocity_modes: list[dict[str, Any]],
    restitutions: list[float],
    geometry_scale: float,
    wall_height_scale: float,
    side_camera_extra_radius: float,
    side_camera_extra_height: float,
    workers: int,
) -> dict[str, Any]:
    variants = _variants(masses, velocity_modes, restitutions)
    scenes: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []

    for idx, params in enumerate(variants):
        scene_id = (
            f"scene_{idx:04d}_col_"
            f"m{_tag(params['mass_a_kg'], 1000)}_"
            f"{params['velocity_mode']}_"
            f"r{_tag(params['restitution'], 100)}"
        )
        scene_dir = output_dir / scene_id
        scene = {
            "scene_id": scene_id,
            "path": _repo_path(scene_dir),
            "scenario": SCENARIO,
            "mass_a_kg": params["mass_a_kg"],
            "velocity_mode": params["velocity_mode"],
            "velocity_kind": params["velocity_kind"],
            "velocity_scale": params["velocity_scale"],
            "velocity_split": params["velocity_split"],
            "object_a_initial_velocity_m_s": params["object_a_initial_velocity_m_s"],
            "object_b_initial_velocity_m_s": params["object_b_initial_velocity_m_s"],
            "restitution": params["restitution"],
            "config": _repo_path(scene_dir / "config.json"),
            "metadata": _repo_path(scene_dir / "metadata.json"),
            "object_poses": _repo_path(scene_dir / "object_poses.csv"),
        }
        scenes.append(scene)
        jobs.append(
            {
                "progress_label": f"[{idx + 1}/{len(variants)}]",
                "scene_id": scene_id,
                "scene_dir": str(scene_dir),
                "params": params,
                "video_fps": video_fps,
                "sim_hz": sim_hz,
                "duration_sec": duration_sec,
                "max_frames": max_frames,
                "render_camera_names": render_camera_names,
                "video_camera_names": video_camera_names,
                "overwrite": overwrite,
                "write_videos": write_videos,
                "geometry_scale": geometry_scale,
                "wall_height_scale": wall_height_scale,
                "side_camera_extra_radius": side_camera_extra_radius,
                "side_camera_extra_height": side_camera_extra_height,
            }
        )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if workers <= 1:
            for job in jobs:
                _generate_scene_job(job)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_generate_scene_job, job) for job in jobs]
                for future in as_completed(futures):
                    print(f"finished {future.result()}", flush=True)

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
        "param_names": ["mass_a_kg", "velocity_distribution", "restitution"],
        "param_grid": {
            "mass_a_kg": masses,
            "velocity_modes": velocity_modes,
            "restitution": restitutions,
        },
        "fixed_params": {
            "environment": "room",
            "object_count": 2,
            "object_kind": "rigid_box",
            "object_full_size_m": [2.0 * value * geometry_scale for value in COLLISION_OBJECT_HALF_EXTENTS_M],
            "object_geometry_scale": geometry_scale,
            "wall_height_scale": wall_height_scale,
            "side_camera_extra_radius_m": side_camera_extra_radius,
            "side_camera_extra_height_m": side_camera_extra_height,
            "object_b_mass_kg": 0.160,
            "object_lateral_friction": 0.002,
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
    parser.add_argument("--duration-sec", type=float, default=float(SCENARIO_DEFAULTS[SCENARIO]["duration_sec"]))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--render-camera-names", default="all")
    parser.add_argument("--video-camera-names", default="all")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--masses", default="0.22,0.44", help="Comma-separated object A masses in kg.")
    parser.add_argument(
        "--velocity-modes",
        default="asym1.4,equal1.05",
        help="Comma-separated modes, e.g. asym1.4,equal1.05.",
    )
    parser.add_argument("--restitutions", default="0.98,0.90", help="Comma-separated object/rail restitution values.")
    parser.add_argument("--geometry-scale", type=float, default=1.75)
    parser.add_argument("--wall-height-scale", type=float, default=1.0)
    parser.add_argument("--side-camera-extra-radius", type=float, default=0.3)
    parser.add_argument("--side-camera-extra-height", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1, help="Parallel scene workers.")
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
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    manifest = generate_collision_variants(
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
        masses=_parse_float_list(args.masses),
        velocity_modes=_parse_velocity_modes(args.velocity_modes),
        restitutions=_parse_float_list(args.restitutions),
        geometry_scale=args.geometry_scale,
        wall_height_scale=args.wall_height_scale,
        side_camera_extra_radius=args.side_camera_extra_radius,
        side_camera_extra_height=args.side_camera_extra_height,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2))
    if not args.dry_run:
        print(f"Wrote final collision variant dataset: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
