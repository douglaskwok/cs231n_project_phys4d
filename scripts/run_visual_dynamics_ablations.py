#!/usr/bin/env python
"""Phase 6 ablations: visual vs states-only; horizon sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/visual_dynamics.json")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "outputs/sphere_bounce_batch/dataset_manifest.json")
    parser.add_argument("--epochs-a", type=int, default=15)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    out = REPO_ROOT / "outputs/visual_dynamics/ablations"
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        _run(
            [
                py,
                "scripts/extract_perception.py",
                "--config",
                str(args.config),
                "--limit",
                "10",
            ]
        )
        _run(
            [
                py,
                "scripts/train_visual_dynamics.py",
                "--config",
                str(args.config),
                "--manifest",
                str(args.manifest),
                "--out-dir",
                str(out / "visual"),
            ]
        )
        _run(
            [
                py,
                "scripts/train_visual_dynamics.py",
                "--config",
                str(args.config),
                "--manifest",
                str(args.manifest),
                "--no-visual",
                "--out-dir",
                str(out / "states_only"),
            ]
        )

    _run(
        [
            py,
            "scripts/eval_visual_dynamics.py",
            "--checkpoint",
            str(out / "visual/visual_dynamics.pt"),
            "--out-json",
            str(out / "eval_visual.json"),
        ]
    )
    _run(
        [
            py,
            "scripts/eval_visual_dynamics.py",
            "--checkpoint",
            str(out / "states_only/visual_dynamics_states_only.pt"),
            "--out-json",
            str(out / "eval_states_only.json"),
        ]
    )
    print(f"Ablation outputs under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
