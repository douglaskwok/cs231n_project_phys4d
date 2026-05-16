#!/usr/bin/env python
"""Generate many sphere-bounce scenes with varied physics for cross-scene training.

Reads ``configs/cross_scene_batch.json``, materializes per-scene JSON configs under
``outputs/sphere_bounce_batch/<scene_id>/config.json``, and invokes
``generate_sphere_bounce_dataset.py`` for each.

One scene is a few seconds on CPU; 50–100 scenes overnight is realistic.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scene_grid(manifest: dict) -> list[dict]:
    grid = manifest["grid"]
    keys = ["restitution", "mass_kg", "drop_z_m"]
    combos = []
    for e, m, z in itertools.product(
        grid["restitution"],
        grid["mass_kg"],
        grid["drop_z_m"],
    ):
        combos.append({"restitution": e, "mass_kg": m, "drop_z_m": z})
    max_n = int(manifest.get("max_scenes", len(combos)))
    return combos[:max_n]


def _apply_physics(base: dict, restitution: float, mass_kg: float, drop_z_m: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["scene"]["true_restitution"] = restitution
    for obj in cfg["scene"]["objects"]:
        if obj["name"] == "sphere":
            obj["mass_kg"] = mass_kg
            pos = list(obj["initial_position_m"])
            pos[2] = drop_z_m
            obj["initial_position_m"] = pos
    return cfg


def _scene_name(manifest: dict, index: int, params: dict) -> str:
    template = manifest.get("naming", "scene_{index:04d}")
    return template.format(index=index, **params)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs" / "cross_scene_batch.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List scenes and paths only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of scenes (overrides manifest max_scenes)",
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest.resolve())
    base_path = REPO_ROOT / manifest["base_config"]
    base_cfg = _load_json(base_path)
    batch_root = REPO_ROOT / manifest["batch_root"]
    grid = _scene_grid(manifest)
    if args.limit is not None:
        grid = grid[: args.limit]

    gen_script = REPO_ROOT / "scripts" / "generate_sphere_bounce_dataset.py"
    planned: list[tuple[str, Path]] = []

    for i, params in enumerate(grid):
        name = _scene_name(manifest, i, params)
        scene_dir = batch_root / name
        cfg = _apply_physics(
            base_cfg,
            restitution=float(params["restitution"]),
            mass_kg=float(params["mass_kg"]),
            drop_z_m=float(params["drop_z_m"]),
        )
        cfg["experiment"] = name
        rel = scene_dir.relative_to(REPO_ROOT).as_posix()
        cfg["outputs"] = {
            "rgb_frames": f"{rel}/rgb/",
            "masks": f"{rel}/masks/",
            "camera_poses": f"{rel}/cameras.json",
            "object_poses": f"{rel}/object_poses.csv",
            "metadata": f"{rel}/metadata.json",
            "recovery_report": f"{rel}/restitution_recovery.json",
        }
        cfg_path = scene_dir / "config.json"
        planned.append((name, cfg_path))
        if args.dry_run:
            continue
        scene_dir.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        cmd = [sys.executable, str(gen_script), "--config", str(cfg_path)]
        # Prefer same interpreter; override with PHYS4D_PYTHON if set.
        py = os.environ.get("PHYS4D_PYTHON")
        if py:
            cmd[0] = py
        print(f"[{i + 1}/{len(grid)}] {name}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    print(f"Planned {len(planned)} scenes under {batch_root}")
    for name, cfg_path in planned[:5]:
        print(f"  {name} -> {cfg_path}")
    if len(planned) > 5:
        print(f"  ... and {len(planned) - 5} more")
    if args.dry_run:
        print("(dry-run: no simulation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
