#!/usr/bin/env python
"""Build train/val/test manifest (80/10/10 by scene instance) for param-ID dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PREDICT_NAMES = ["restitution", "mass_kg", "drop_z_m"]


def _load_physics(scene_dir: Path) -> dict | None:
    path = scene_dir / "physics_params.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    meta = scene_dir / "metadata.json"
    if meta.is_file():
        with meta.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("physics_params")
    return None


def discover_scenes(batch_root: Path) -> list[dict]:
    scenes: list[dict] = []
    for cfg_path in sorted(batch_root.glob("*/config.json")):
        scene_dir = cfg_path.parent
        physics = _load_physics(scene_dir)
        if physics is None:
            continue
        names = physics["param_names"]
        vec = physics["param_vector"]
        idx = {n: i for i, n in enumerate(names)}
        target = [float(vec[idx[n]]) for n in PREDICT_NAMES]
        scenes.append(
            {
                "scene_id": scene_dir.name,
                "param_names": PREDICT_NAMES,
                "param_vector": target,
                "restitution": target[0],
                "mass_kg": target[1],
                "drop_z_m": target[2],
            }
        )
    return scenes


def assign_splits(scenes: list[dict], ratios: dict, seed: int) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    ids = [s["scene_id"] for s in scenes]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * float(ratios.get("train", 0.8)))
    n_val = int(n * float(ratios.get("val", 0.1)))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in scenes:
        if s["scene_id"] in train_ids:
            buckets["train"].append(s)
        elif s["scene_id"] in val_ids:
            buckets["val"].append(s)
        else:
            buckets["test"].append(s)
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs" / "param_id_dataset.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <batch_root>/dataset_manifest.json",
    )
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    batch_root = REPO_ROOT / manifest["batch_root"]
    scenes = discover_scenes(batch_root)
    if not scenes:
        raise SystemExit(f"No scenes under {batch_root}. Run generate_param_id_batch.py first.")

    buckets = assign_splits(scenes, manifest["split_ratios"], int(manifest.get("seed", 42)))
    out_path = args.out or (batch_root / "dataset_manifest.json")
    payload = {
        "batch_root": batch_root.name,
        "predict_names": PREDICT_NAMES,
        "num_scenes": len(scenes),
        "split_ratios": manifest["split_ratios"],
        "splits": buckets,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_path}")
    for split, rows in buckets.items():
        print(f"  {split}: {len(rows)} scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
