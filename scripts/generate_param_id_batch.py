#!/usr/bin/env python
"""Generate many randomized sphere-bounce scenes for param-ID training.

Reads ``configs/param_id_dataset.json``, samples physics parameters, writes per-scene
configs under ``outputs/param_id_dataset/``, and invokes ``generate_sphere_bounce_dataset.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _linspace_samples(spec: dict) -> list[float]:
    lo, hi = float(spec["min"]), float(spec["max"])
    steps = int(spec.get("steps", 1))
    if steps <= 1:
        return [lo]
    return [float(x) for x in np.linspace(lo, hi, steps)]


def _sample_params(manifest: dict, rng: np.random.Generator) -> list[dict]:
    s = manifest["sampling"]
    if all("steps" in s[k] for k in ("restitution", "mass_kg", "drop_z_m")):
        rows: list[dict] = []
        for e in _linspace_samples(s["restitution"]):
            for m in _linspace_samples(s["mass_kg"]):
                for z in _linspace_samples(s["drop_z_m"]):
                    rows.append(
                        {
                            "restitution": e,
                            "mass_kg": m,
                            "drop_z_m": z,
                        }
                    )
        n = int(manifest.get("target_scenes", len(rows)))
        return rows[:n]

    n = int(manifest["target_scenes"])
    rows = []
    for _ in range(n):
        rows.append(
            {
                "restitution": float(rng.uniform(s["restitution"]["min"], s["restitution"]["max"])),
                "mass_kg": float(rng.uniform(s["mass_kg"]["min"], s["mass_kg"]["max"])),
                "drop_z_m": float(rng.uniform(s["drop_z_m"]["min"], s["drop_z_m"]["max"])),
            }
        )
    return rows


def _apply_cameras(cfg: dict, manifest: dict, *, num_cameras: int | None) -> None:
    cams = cfg.setdefault("cameras", {})
    block = manifest.get("cameras", {})
    n = num_cameras if num_cameras is not None else int(block.get("num_cameras", cams.get("num_cameras", 10)))
    cams["num_cameras"] = n
    if "train_cameras" in block:
        cams["train_cameras"] = list(block["train_cameras"])
    elif n >= 10:
        cams["train_cameras"] = list(range(8))
        cams["test_cameras"] = [8, 9]
    elif n >= 6:
        cams["train_cameras"] = list(range(min(4, n)))
        cams["test_cameras"] = list(range(min(4, n), n))
    if "test_cameras" in block:
        cams["test_cameras"] = list(block["test_cameras"])


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
        default=REPO_ROOT / "configs" / "param_id_dataset.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=None,
        help="Ring camera count (default: manifest cameras.num_cameras or 10)",
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest.resolve())
    base_cfg = _load_json(REPO_ROOT / manifest["base_config"])
    batch_root = REPO_ROOT / manifest["batch_root"]
    rng = np.random.default_rng(int(manifest.get("seed", 42)))
    grid = _sample_params(manifest, rng)
    if args.limit is not None:
        grid = grid[: args.limit]

    gen_script = REPO_ROOT / "scripts" / "generate_sphere_bounce_dataset.py"
    for i, params in enumerate(grid):
        name = _scene_name(manifest, i, params)
        scene_dir = batch_root / name
        cfg = _apply_physics(
            base_cfg,
            restitution=float(params["restitution"]),
            mass_kg=float(params["mass_kg"]),
            drop_z_m=float(params["drop_z_m"]),
        )
        _apply_cameras(cfg, manifest, num_cameras=args.num_cameras)
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
        if args.dry_run:
            print(f"[{i + 1}/{len(grid)}] {name} -> {cfg_path}")
            continue
        scene_dir.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        cmd = [sys.executable, str(gen_script), "--config", str(cfg_path)]
        py = os.environ.get("PHYS4D_PYTHON")
        if py:
            cmd[0] = py
        print(f"[{i + 1}/{len(grid)}] {name}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    print(f"Planned {len(grid)} scenes under {batch_root}")
    if args.dry_run:
        print("(dry-run: no simulation)")
    else:
        print("Next: python scripts/build_param_id_splits.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
