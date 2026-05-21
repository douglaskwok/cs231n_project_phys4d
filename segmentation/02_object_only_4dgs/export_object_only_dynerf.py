#!/usr/bin/env python
"""Export masked/full DyNeRF datasets for fudan 4DGS."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def view_matrix_to_c2w(view_col_major_16: list[float]) -> np.ndarray:
    v = np.array(view_col_major_16, dtype=np.float64).reshape(4, 4, order="F")
    return np.linalg.inv(v)


def camera_angle_x_from_vertical_fov(fov_deg: float, width: int, height: int) -> float:
    v = math.radians(float(fov_deg))
    return float(2.0 * math.atan(math.tan(v / 2.0) * (float(width) / float(height))))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _intrinsics(fov_deg: float, width: int, height: int) -> dict[str, float | int]:
    cam_angle_x = camera_angle_x_from_vertical_fov(fov_deg, width, height)
    return {
        "w": width,
        "h": height,
        "fl_x": float(width / (2.0 * math.tan(cam_angle_x / 2.0))),
        "fl_y": float(height / (2.0 * math.tan(math.radians(fov_deg) / 2.0))),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "camera_angle_x": cam_angle_x,
    }


def _read_rgb(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    arr = imageio.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    mask = imageio.imread(path)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def _apply_mask(rgb: np.ndarray, mask: np.ndarray, background: str, mode: str) -> np.ndarray:
    if mode == "full":
        return rgb.copy()

    bg_value = 255 if background == "white" else 0
    out = np.full_like(rgb, bg_value)
    keep = ~mask if mode == "background" else mask
    out[keep] = rgb[keep]
    return out


def export_object_only_dynerf(
    *,
    config: Path,
    masks_root: Path,
    output: Path,
    background: str,
    mode: str = "object",
) -> dict[str, Any]:
    import imageio.v2 as imageio

    cfg = _load_json(config)
    sim = cfg["simulation"]
    cams_cfg = cfg["cameras"]
    train_frames = sim["train_frames"]
    test_frames = sim["test_frames"]

    rgb_root = (REPO_ROOT / cfg["outputs"]["rgb_frames"]).resolve()
    cameras_json = (REPO_ROOT / cfg["outputs"]["camera_poses"]).resolve()
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"Missing RGB root: {rgb_root}")
    if not cameras_json.is_file():
        raise FileNotFoundError(f"Missing cameras JSON: {cameras_json}")
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Missing masks root: {masks_root}")

    cam_blob = _load_json(cameras_json)
    cam_by_index = {int(rec["index"]): rec for rec in cam_blob["cameras"]}

    width = int(cams_cfg["image_size"][0])
    height = int(cams_cfg["image_size"][1])
    fov_deg = float(cams_cfg.get("fov_deg", 60.0))
    fps = 1.0 / float(sim.get("dt_s", 1.0 / 60.0))

    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def frame_entries(camera_ids: list[int], lo: int, hi: int) -> tuple[list[dict], int]:
        entries: list[dict] = []
        empty_masks = 0
        for cam_id in camera_ids:
            rec = cam_by_index[cam_id]
            c2w = view_matrix_to_c2w(rec["view_matrix_row_major"])
            for frame_idx in range(lo, hi + 1):
                tag = f"{frame_idx:05d}"
                rgb_path = rgb_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
                mask_path = masks_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
                if not rgb_path.is_file():
                    raise FileNotFoundError(f"Missing RGB frame: {rgb_path}")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask frame: {mask_path}")

                rgb = _read_rgb(rgb_path)
                mask = _read_mask(mask_path)
                if not mask.any():
                    empty_masks += 1
                masked = _apply_mask(rgb, mask, background, mode)

                stem = f"images/cam{cam_id:02d}_{frame_idx:05d}"
                imageio.imwrite(output / f"{stem}.png", masked)
                frame_intr = _intrinsics(float(rec.get("fov_deg", fov_deg)), width, height)
                entries.append(
                    {
                        **frame_intr,
                        "file_path": stem,
                        "transform_matrix": c2w.tolist(),
                        "time": float(frame_idx) / float(fps),
                    }
                )
        return entries, empty_masks

    train, train_empty = frame_entries(
        list(cams_cfg["train_cameras"]), int(train_frames[0]), int(train_frames[1])
    )
    test, test_empty = frame_entries(
        list(cams_cfg["test_cameras"]), int(test_frames[0]), int(test_frames[1])
    )

    intr = _intrinsics(fov_deg, width, height)
    with (output / "transforms_train.json").open("w", encoding="utf-8") as f:
        json.dump({**intr, "frames": train}, f, indent=2)
    with (output / "transforms_test.json").open("w", encoding="utf-8") as f:
        json.dump({**intr, "frames": test}, f, indent=2)

    meta = {
        "format": f"dynerf_{mode}",
        "source_config": str(config),
        "rgb_root": str(rgb_root),
        "masks_root": str(masks_root),
        "mode": mode,
        "background": background,
        "fps": fps,
        "train_cameras": list(cams_cfg["train_cameras"]),
        "test_cameras": list(cams_cfg["test_cameras"]),
        "train_frame_range": [int(train_frames[0]), int(train_frames[1])],
        "test_frame_range": [int(test_frames[0]), int(test_frames[1])],
        "num_train_views": len(train),
        "num_test_views": len(test),
        "empty_train_masks": train_empty,
        "empty_test_masks": test_empty,
        "image_size": [width, height],
        "fov_deg": fov_deg,
        "time_duration_suggested": [0.0, int(train_frames[1]) / fps],
    }
    with (output / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def run_smoke_test() -> int:
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rgb = root / "rgb" / "cam00"
        masks = root / "masks" / "cam00"
        rgb.mkdir(parents=True)
        masks.mkdir(parents=True)
        image = np.full((32, 32, 3), 50, dtype=np.uint8)
        image[12:20, 12:20] = [30, 90, 220]
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[12:20, 12:20] = 255
        imageio.imwrite(rgb / "frame00000.png", image)
        imageio.imwrite(masks / "frame00000.png", mask)

        cameras = {
            "cameras": [
                {
                    "index": 0,
                    "view_matrix_row_major": np.eye(4).reshape(-1).tolist(),
                }
            ]
        }
        (root / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
        cfg = {
            "simulation": {"train_frames": [0, 0], "test_frames": [0, 0], "dt_s": 1 / 60},
            "cameras": {
                "train_cameras": [0],
                "test_cameras": [0],
                "image_size": [32, 32],
                "fov_deg": 60.0,
            },
            "outputs": {
                "rgb_frames": str(rgb.parent),
                "camera_poses": str(root / "cameras.json"),
            },
        }
        cfg_path = root / "config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        expectations = {
            "object": ((0, 0, 0), (30, 90, 220)),
            "background": ((50, 50, 50), (0, 0, 0)),
            "full": ((50, 50, 50), (30, 90, 220)),
        }
        for mode, (bg_expected, obj_expected) in expectations.items():
            out_dir = root / f"out_{mode}"
            meta = export_object_only_dynerf(
                config=cfg_path,
                masks_root=masks.parent,
                output=out_dir,
                background="black",
                mode=mode,
            )
            print(json.dumps(meta, indent=2))
            out_img = imageio.imread(out_dir / "images" / "cam00_00000.png")
            if tuple(out_img[0, 0]) != bg_expected or tuple(out_img[15, 15]) != obj_expected:
                raise RuntimeError(f"Smoke masked image failed for mode={mode}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "sphere_bounce_m2.json")
    parser.add_argument("--masks-root", type=Path, default=REPO_ROOT / "outputs" / "sphere_bounce_m2" / "masks")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "segmentation" / "02_object_only_4dgs" / "dynerf_ball_only")
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument(
        "--mode",
        choices=("object", "background", "full"),
        default="object",
        help="object keeps mask pixels, background removes mask pixels, full keeps original RGB.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        return run_smoke_test()

    meta = export_object_only_dynerf(
        config=args.config.resolve(),
        masks_root=args.masks_root.resolve(),
        output=args.output.resolve(),
        background=args.background,
        mode=args.mode,
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote {args.mode} DyNeRF dataset: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
