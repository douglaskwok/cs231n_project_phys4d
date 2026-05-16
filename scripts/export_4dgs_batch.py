#!/usr/bin/env python
"""Export every scene under outputs/sphere_bounce_batch to DyNeRF folders for fudan 4DGS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d.dynerf_export import export_dynerf_dataset  # noqa: E402
from phys4d.visual_dynamics.dataset import discover_batch_scenes  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sphere_bounce_batch",
    )
    parser.add_argument(
        "--dynerf-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sphere_bounce_batch_dynerf",
    )
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    scenes = discover_batch_scenes(args.batch_root)
    if not scenes:
        print(f"No scenes under {args.batch_root}. Run generate_sphere_bounce_batch.py first.", file=sys.stderr)
        return 1
    if args.limit:
        scenes = scenes[: args.limit]

    manifest: list[dict] = []
    args.dynerf_root.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        cfg = _load_json(scene.config_path)
        sim = cfg["simulation"]
        cams_cfg = cfg["cameras"]
        out_dir = args.dynerf_root / scene.scene_id
        rgb_root = scene.rgb_root
        cameras_json = scene.config_path.parent / "cameras.json"
        if not rgb_root.is_dir():
            print(f"Skip {scene.scene_id}: missing rgb", file=sys.stderr)
            continue
        dt = float(sim.get("dt_s", 1.0 / 60.0))
        meta = export_dynerf_dataset(
            rgb_root=rgb_root,
            cameras_json=cameras_json,
            out_dir=out_dir.resolve(),
            train_cameras=list(cams_cfg["train_cameras"]),
            test_cameras=list(cams_cfg["test_cameras"]),
            train_frame_range=(int(sim["train_frames"][0]), int(sim["train_frames"][1])),
            test_frame_range=(int(sim["test_frames"][0]), int(sim["test_frames"][1])),
            fps=1.0 / dt,
            fov_deg=float(cams_cfg.get("fov_deg", 60.0)),
            image_size=(
                int(cams_cfg["image_size"][0]),
                int(cams_cfg["image_size"][1]),
            ),
            copy_images=not args.symlink,
        )
        manifest.append(
            {
                "scene_id": scene.scene_id,
                "dynerf_dir": str(out_dir.relative_to(REPO_ROOT)),
                "restitution": scene.restitution,
                "mass_kg": scene.mass_kg,
                "drop_z_m": scene.drop_z_m,
                **meta,
            }
        )
        print(f"Exported {scene.scene_id} -> {out_dir}")

    manifest_path = args.dynerf_root / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"scenes": manifest}, f, indent=2)
    print(f"Wrote {len(manifest)} scenes to {manifest_path}")
    print("Train one scene: modal run modal_app.py --upload-4d  (point at dynerf dir)")
    print("Batch Modal training: use --scene-id flag or loop manifest (see docs/project_direction.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
