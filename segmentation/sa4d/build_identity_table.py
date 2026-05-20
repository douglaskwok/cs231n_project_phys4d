#!/usr/bin/env python
"""Build an SA4D-style Gaussian identity table from multiview masks.

This is a lightweight baseline, not a full SA4D reproduction. It projects trained
Gaussian centers into segmented views over multiple timestamps and records which
Gaussians land inside object masks.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BINDING_DIR = REPO_ROOT / "segmentation" / "01_mask_guided_gaussian_binding"
if str(BINDING_DIR) not in sys.path:
    sys.path.insert(0, str(BINDING_DIR))

from select_ball_gaussians import (  # noqa: E402
    GaussianCloud,
    _opacity_keep,
    _write_tiny_ply,
    filter_points_by_mask_union,
    load_cameras,
    load_gaussian_ply,
    save_gaussian_ply,
)


def _parse_frames(args: argparse.Namespace) -> list[int]:
    if args.frames:
        frames = [int(part.strip()) for part in args.frames.split(",") if part.strip()]
    else:
        end = args.frame_end if args.frame_end is not None else args.frame_start
        frames = list(range(int(args.frame_start), int(end) + 1, int(args.frame_step)))
    if not frames:
        raise ValueError("No frames requested")
    return sorted(set(frames))


def build_identity_table(
    *,
    ply: Path,
    masks_root: Path,
    cameras_json: Path,
    frames: list[int],
    min_camera_hits: int,
    min_temporal_hits: int,
    opacity_percentile: float,
    out_dir: Path,
    write_frame_plys: bool,
) -> dict:
    cloud = load_gaussian_ply(ply)
    cameras = load_cameras(cameras_json)

    opacity_keep = _opacity_keep(cloud, opacity_percentile)
    candidate_idx = np.where(opacity_keep)[0]
    candidates = cloud.xyz[candidate_idx]

    keep_table = np.zeros((len(frames), len(cloud.vertices)), dtype=bool)
    frame_reports: list[dict] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    for row, frame in enumerate(frames):
        print(f"[{row + 1}/{len(frames)}] projecting frame {frame}", flush=True)
        keep_candidates = filter_points_by_mask_union(
            candidates,
            masks_root=masks_root,
            frame=frame,
            min_camera_hits=min_camera_hits,
            pybullet_cameras=cameras,
        )
        keep = np.zeros(len(cloud.vertices), dtype=bool)
        keep[candidate_idx[keep_candidates]] = True
        keep_table[row] = keep

        selected = int(keep.sum())
        frame_report = {
            "frame": int(frame),
            "selected_gaussians": selected,
            "selected_fraction": float(keep.mean()) if len(keep) else 0.0,
        }
        frame_reports.append(frame_report)

        if write_frame_plys:
            frame_dir = out_dir / f"frame_{frame:05d}"
            save_gaussian_ply(
                GaussianCloud(vertices=cloud.vertices[keep].copy()),
                frame_dir / "ball_gaussians.ply",
            )
            save_gaussian_ply(
                GaussianCloud(vertices=cloud.vertices[~keep].copy()),
                frame_dir / "background_gaussians.ply",
            )

    temporal_votes = keep_table.sum(axis=0)
    stable_keep = temporal_votes >= int(min_temporal_hits)
    stable_ball_ply = out_dir / "stable_ball_gaussians.ply"
    stable_background_ply = out_dir / "stable_background_gaussians.ply"
    save_gaussian_ply(
        GaussianCloud(vertices=cloud.vertices[stable_keep].copy()),
        stable_ball_ply,
    )
    save_gaussian_ply(
        GaussianCloud(vertices=cloud.vertices[~stable_keep].copy()),
        stable_background_ply,
    )

    npz_path = out_dir / "identity_table.npz"
    np.savez_compressed(
        npz_path,
        frames=np.asarray(frames, dtype=np.int32),
        keep_table=keep_table.astype(np.uint8),
        temporal_votes=temporal_votes.astype(np.int32),
        stable_keep=stable_keep.astype(np.uint8),
        candidate_indices=candidate_idx.astype(np.int32),
    )

    report = {
        "format": "sa4d_inspired_identity_table_v1",
        "note": "Mask-projection baseline; does not train SA4D's temporal identity feature field.",
        "ply": str(ply),
        "masks_root": str(masks_root),
        "cameras_json": str(cameras_json),
        "frames": [int(f) for f in frames],
        "num_frames": len(frames),
        "min_camera_hits": int(min_camera_hits),
        "min_temporal_hits": int(min_temporal_hits),
        "opacity_percentile": float(opacity_percentile),
        "total_gaussians": int(len(cloud.vertices)),
        "opacity_candidates": int(opacity_keep.sum()),
        "stable_selected_gaussians": int(stable_keep.sum()),
        "stable_selected_fraction": float(stable_keep.mean()) if len(stable_keep) else 0.0,
        "identity_table_npz": str(npz_path),
        "stable_ball_ply": str(stable_ball_ply),
        "stable_background_ply": str(stable_background_ply),
        "frame_reports": frame_reports,
    }
    with (out_dir / "identity_table.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def run_smoke_test() -> int:
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ply = root / "toy.ply"
        masks = root / "masks"
        cam_dir = masks / "cam00"
        cam_dir.mkdir(parents=True)

        for frame in (0, 1):
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[24:40, 24:40] = 255
            imageio.imwrite(cam_dir / f"frame{frame:05d}.png", mask)

        cameras_json = root / "cameras.json"
        camera = {
            "index": 0,
            "image_size": [64, 64],
            "view_matrix_row_major": np.eye(4).reshape(-1).tolist(),
            "projection_matrix_row_major": np.eye(4).reshape(-1).tolist(),
        }
        cameras_json.write_text(json.dumps({"cameras": [camera]}), encoding="utf-8")

        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.15, 0.15, 0.0],
                [0.8, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        _write_tiny_ply(ply, xyz)
        report = build_identity_table(
            ply=ply,
            masks_root=masks,
            cameras_json=cameras_json,
            frames=[0, 1],
            min_camera_hits=1,
            min_temporal_hits=1,
            opacity_percentile=0.0,
            out_dir=root / "out",
            write_frame_plys=True,
        )
        print(json.dumps(report, indent=2))
        if report["stable_selected_gaussians"] < 1:
            raise RuntimeError("Smoke test selected no stable Gaussians")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--masks-root", type=Path)
    parser.add_argument("--cameras-json", type=Path)
    parser.add_argument("--frames", default=None, help="Comma-separated frames, e.g. 0,10,20")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--min-camera-hits", type=int, default=2)
    parser.add_argument("--min-temporal-hits", type=int, default=1)
    parser.add_argument("--opacity-percentile", type=float, default=0.0)
    parser.add_argument("--write-frame-plys", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("segmentation/sa4d/runs/latest"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        return run_smoke_test()

    required = {
        "--ply": args.ply,
        "--masks-root": args.masks_root,
        "--cameras-json": args.cameras_json,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"Missing required args: {', '.join(missing)}")

    frames = _parse_frames(args)
    report = build_identity_table(
        ply=args.ply.resolve(),
        masks_root=args.masks_root.resolve(),
        cameras_json=args.cameras_json.resolve(),
        frames=frames,
        min_camera_hits=args.min_camera_hits,
        min_temporal_hits=args.min_temporal_hits,
        opacity_percentile=args.opacity_percentile,
        out_dir=args.out_dir.resolve(),
        write_frame_plys=args.write_frame_plys,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
