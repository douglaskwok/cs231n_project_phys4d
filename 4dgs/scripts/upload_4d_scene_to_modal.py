#!/usr/bin/env python
"""Upload a prepared 4DGS/DyNeRF scene to the Modal data volume.

This bypasses ``modal run modal_app.py --upload-4d`` so uploading data does not
trigger the expensive 4DGS image build. Use this when you already have a folder
with ``transforms_train.json`` and ``images/``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

_FOURDGS_DIR = Path(__file__).resolve().parents[1]
if str(_FOURDGS_DIR) not in sys.path:
    sys.path.insert(0, str(_FOURDGS_DIR))

from _paths import REPO_ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        type=Path,
        help="Local 4DGS/DyNeRF scene folder containing transforms_train.json.",
    )
    parser.add_argument("--volume", default="phys4d-gs-data")
    parser.add_argument("--remote-path", default="4d_scene")
    parser.add_argument(
        "--modal-cmd",
        default="modal",
        help='Modal command to use, e.g. "modal" or "arch -arm64 modal".',
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove the existing remote folder before upload.",
    )
    args = parser.parse_args()

    scene = args.scene.resolve()
    if not (scene / "transforms_train.json").is_file():
        print(f"Missing transforms_train.json under: {scene}", file=sys.stderr)
        return 1
    if not (scene / "images").is_dir():
        print(f"Missing images/ under: {scene}", file=sys.stderr)
        return 1

    modal_cmd = shlex.split(args.modal_cmd)
    if not args.keep_existing:
        subprocess.run(
            [*modal_cmd, "volume", "rm", args.volume, args.remote_path, "-r"],
            check=False,
            cwd=str(REPO_ROOT),
        )

    subprocess.run(
        [*modal_cmd, "volume", "put", args.volume, str(scene), args.remote_path],
        check=True,
        cwd=str(REPO_ROOT),
    )
    print(f"Uploaded {scene} -> {args.volume}:/{args.remote_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
