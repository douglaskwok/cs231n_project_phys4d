"""Scene discovery for cross-scene visual dynamics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    scene_dir: Path
    restitution: float
    mass_kg: float
    drop_z_m: float


def discover_batch_scenes(batch_root: Path) -> list[SceneRecord]:
    batch_root = batch_root.resolve()
    scenes: list[SceneRecord] = []
    for cfg_path in sorted(batch_root.glob("*/config.json")):
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        scene_dir = cfg_path.parent
        if not (scene_dir / "object_poses.csv").is_file():
            continue
        scene = cfg["scene"]
        sphere = next(o for o in scene["objects"] if o["name"] == "sphere")
        scenes.append(
            SceneRecord(
                scene_id=scene_dir.name,
                scene_dir=scene_dir,
                restitution=float(scene["true_restitution"]),
                mass_kg=float(sphere["mass_kg"]),
                drop_z_m=float(sphere["initial_position_m"][2]),
            )
        )
    return scenes


def load_manifest_splits(manifest_path: Path, data_root: Path) -> dict[str, list[SceneRecord]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    batch_name = data["batch_root"]
    out: dict[str, list[SceneRecord]] = {}
    for split, rows in data["splits"].items():
        scenes: list[SceneRecord] = []
        for row in rows:
            scene_dir = data_root / batch_name / row["scene_id"]
            scenes.append(
                SceneRecord(
                    scene_id=row["scene_id"],
                    scene_dir=scene_dir,
                    restitution=float(row.get("restitution", 0.0)),
                    mass_kg=float(row.get("mass_kg", 1.0)),
                    drop_z_m=float(row.get("drop_z_m", 1.5)),
                )
            )
        out[split] = scenes
    return out
