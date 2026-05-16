#!/usr/bin/env python
"""Verify sphere_bounce_batch files are readable locally (not iCloud placeholders)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.poses import load_object_poses_csv  # noqa: E402
from phys4d.visual_dynamics.scenes import discover_batch_scenes  # noqa: E402


def _readable(path: Path) -> str | None:
    try:
        data = path.read_bytes()
        if len(data) == 0 and path.stat().st_size > 0:
            return "empty read (iCloud placeholder?)"
    except OSError as exc:
        return str(exc)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=REPO_ROOT / "outputs/sphere_bounce_batch",
    )
    args = parser.parse_args()
    batch_root = args.batch_root.resolve()
    if not batch_root.is_dir():
        print(f"Missing {batch_root}", file=sys.stderr)
        return 1

    bad: list[str] = []
    scenes = discover_batch_scenes(batch_root)
    for scene in scenes:
        poses = scene.scene_dir / "object_poses.csv"
        try:
            load_object_poses_csv(poses)
        except Exception as exc:
            bad.append(f"{scene.scene_id} poses: {exc}")
        for sub in ("rgb", "masks"):
            root = scene.scene_dir / sub
            if not root.is_dir():
                continue
            for png in sorted(root.rglob("*.png"))[:3]:
                err = _readable(png)
                if err:
                    bad.append(f"{scene.scene_id} {png.name}: {err}")
                    break

    print(f"Scenes: {len(scenes)}")
    if bad:
        print(f"FAIL: {len(bad)} issue(s) — fix before Modal upload:", file=sys.stderr)
        for line in bad[:20]:
            print(f"  {line}", file=sys.stderr)
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more", file=sys.stderr)
        return 1
    print("OK: batch looks fully local and readable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
