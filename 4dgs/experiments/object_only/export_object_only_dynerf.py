#!/usr/bin/env python
"""Export masked/full DyNeRF datasets for fudan 4DGS."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_FOURDGS_DIR = Path(__file__).resolve().parents[2]
if str(_FOURDGS_DIR) not in sys.path:
    sys.path.insert(0, str(_FOURDGS_DIR))

from _paths import REPO_ROOT  # noqa: E402

_REPO_SRC = REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from phys4d.bounce.dataset_split import compute_time_split  # noqa: E402


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


def _mask_pixels(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask frame: {path}")
    return int(_read_mask(path).sum())


def _read_frame_list(path: Path) -> list[int]:
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        frames.append(int(line))
    if not frames:
        raise ValueError(f"Frame list is empty: {path}")
    return sorted(dict.fromkeys(frames))


def export_object_only_dynerf(
    *,
    config: Path,
    masks_root: Path,
    output: Path,
    background: str,
    mode: str = "object",
    min_mask_pixels: int = 0,
    min_visible_cameras: int = 1,
    trim_empty_time_ends: bool = False,
    drop_invisible_frames: bool = False,
    drop_invisible_views: bool = False,
    frame_list: Path | None = None,
    all_train: bool = False,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    cfg = _load_json(config)
    sim = cfg["simulation"]
    cams_cfg = cfg["cameras"]
    train_frames = sim["train_frames"]
    test_frames = sim["test_frames"]
    if not all_train:
        num_frames = int(sim.get("num_frames", int(test_frames[1]) + 1))
        canonical = compute_time_split(0, num_frames - 1)
        train_frames = canonical["train"]
        test_frames = canonical["test"]

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

    def visible_camera_count(camera_ids: list[int], frame_idx: int) -> int:
        tag = f"{frame_idx:05d}"
        return sum(
            _mask_pixels(masks_root / f"cam{cam_id:02d}" / f"frame{tag}.png")
            > min_mask_pixels
            for cam_id in camera_ids
        )

    def visible_frame_indices(camera_ids: list[int], lo: int, hi: int) -> list[int]:
        visible = []
        for frame_idx in range(lo, hi + 1):
            if visible_camera_count(camera_ids, frame_idx) >= min_visible_cameras:
                visible.append(frame_idx)
        return visible

    def effective_range(camera_ids: list[int], lo: int, hi: int) -> tuple[int, int]:
        if not trim_empty_time_ends:
            return lo, hi
        visible = visible_frame_indices(camera_ids, lo, hi)
        if not visible:
            raise ValueError(
                f"No visible timestamps in frame range {lo}..{hi} "
                f"for min_mask_pixels={min_mask_pixels}"
            )
        return visible[0], visible[-1]

    def frame_entries(
        camera_ids: list[int],
        frame_indices: list[int],
    ) -> tuple[list[dict], dict[str, int]]:
        entries: list[dict] = []
        stats = {
            "empty_masks": 0,
            "below_threshold_masks": 0,
            "dropped_invisible_views": 0,
            "written_views": 0,
        }
        for cam_id in camera_ids:
            rec = cam_by_index[cam_id]
            c2w = view_matrix_to_c2w(rec["view_matrix_row_major"])
            for frame_idx in frame_indices:
                tag = f"{frame_idx:05d}"
                rgb_path = rgb_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
                mask_path = masks_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
                if not rgb_path.is_file():
                    raise FileNotFoundError(f"Missing RGB frame: {rgb_path}")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask frame: {mask_path}")

                rgb = _read_rgb(rgb_path)
                mask = _read_mask(mask_path)
                mask_area = int(mask.sum())
                if mask_area == 0:
                    stats["empty_masks"] += 1
                if mask_area <= min_mask_pixels:
                    stats["below_threshold_masks"] += 1
                    if drop_invisible_views and mode != "full":
                        stats["dropped_invisible_views"] += 1
                        continue
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
                        "original_frame": int(frame_idx),
                        "mask_pixels": mask_area,
                    }
                )
                stats["written_views"] += 1
        return entries, stats

    if all_train:
        train_cameras = sorted(cam_by_index)
        test_cameras: list[int] = []
        train_lo = 0
        train_hi = int(sim.get("num_frames", max(int(train_frames[1]), int(test_frames[1])) + 1)) - 1
        test_lo = 0
        test_hi = -1
        fixed_frame_indices = _read_frame_list(frame_list) if frame_list is not None else None
        if fixed_frame_indices is not None:
            out_of_range = [idx for idx in fixed_frame_indices if idx < train_lo or idx > train_hi]
            if out_of_range:
                raise ValueError(
                    f"--frame-list contains frames outside dataset range {train_lo}..{train_hi}: "
                    f"{out_of_range[:10]}"
                )
            train_frame_indices = fixed_frame_indices
        elif drop_invisible_frames and mode != "full":
            train_frame_indices = visible_frame_indices(train_cameras, train_lo, train_hi)
        else:
            train_frame_indices = list(range(train_lo, train_hi + 1))
        test_frame_indices: list[int] = []
    else:
        train_cameras = list(cams_cfg["train_cameras"])
        test_cameras = list(cams_cfg["test_cameras"])
        timeline_cameras = sorted(set(train_cameras + test_cameras))
        train_lo, train_hi = effective_range(timeline_cameras, int(train_frames[0]), int(train_frames[1]))
        test_lo, test_hi = effective_range(timeline_cameras, int(test_frames[0]), int(test_frames[1]))
        fixed_frame_indices = _read_frame_list(frame_list) if frame_list is not None else None
        if fixed_frame_indices is not None:
            all_lo = min(int(train_frames[0]), int(test_frames[0]))
            all_hi = max(int(train_frames[1]), int(test_frames[1]))
            out_of_range = [idx for idx in fixed_frame_indices if idx < all_lo or idx > all_hi]
            if out_of_range:
                raise ValueError(
                    f"--frame-list contains frames outside dataset range {all_lo}..{all_hi}: "
                    f"{out_of_range[:10]}"
                )
            train_frame_indices = [
                idx for idx in fixed_frame_indices if int(train_frames[0]) <= idx <= int(train_frames[1])
            ]
            test_frame_indices = [
                idx for idx in fixed_frame_indices if int(test_frames[0]) <= idx <= int(test_frames[1])
            ]
        elif drop_invisible_frames and mode != "full":
            train_frame_indices = visible_frame_indices(timeline_cameras, train_lo, train_hi)
            test_frame_indices = visible_frame_indices(timeline_cameras, test_lo, test_hi)
        else:
            train_frame_indices = list(range(train_lo, train_hi + 1))
            test_frame_indices = list(range(test_lo, test_hi + 1))

    train, train_empty = frame_entries(train_cameras, train_frame_indices)
    test, test_empty = frame_entries(test_cameras, test_frame_indices)

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
        "train_cameras": train_cameras,
        "test_cameras": test_cameras,
        "train_frame_range": [int(train_frames[0]), int(train_frames[1])],
        "test_frame_range": [int(test_frames[0]), int(test_frames[1])],
        "effective_train_frame_range": [train_lo, train_hi],
        "effective_test_frame_range": [test_lo, test_hi],
        "num_train_views": len(train),
        "num_test_views": len(test),
        "num_train_timestamps": len(train_frame_indices),
        "num_test_timestamps": len(test_frame_indices),
        "train_timestamps": train_frame_indices,
        "test_timestamps": test_frame_indices,
        "empty_train_masks": train_empty["empty_masks"],
        "empty_test_masks": test_empty["empty_masks"],
        "below_threshold_train_masks": train_empty["below_threshold_masks"],
        "below_threshold_test_masks": test_empty["below_threshold_masks"],
        "dropped_train_views": train_empty["dropped_invisible_views"],
        "dropped_test_views": test_empty["dropped_invisible_views"],
        "min_mask_pixels": min_mask_pixels,
        "min_visible_cameras": min_visible_cameras,
        "trim_empty_time_ends": trim_empty_time_ends,
        "drop_invisible_frames": drop_invisible_frames,
        "drop_invisible_views": drop_invisible_views,
        "frame_list": str(frame_list) if frame_list is not None else None,
        "all_train": all_train,
        "preserves_original_timestamps": True,
        "preserves_synchronized_multiview_frames": not drop_invisible_views,
        "image_size": [width, height],
        "fov_deg": fov_deg,
        "time_duration_suggested": [
            float(train_frame_indices[0]) / fps if train_frame_indices else 0.0,
            float(train_frame_indices[-1]) / fps if train_frame_indices else 0.0,
        ],
        "effective_time_range_s": [
            float(train_frame_indices[0]) / fps if train_frame_indices else 0.0,
            float(train_frame_indices[-1]) / fps if train_frame_indices else 0.0,
        ],
    }
    with (output / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    frame_map = {
        "fps": fps,
        "train": [
            {
                "kept_index": kept_idx,
                "original_frame": int(frame_idx),
                "original_time_s": float(frame_idx) / fps,
            }
            for kept_idx, frame_idx in enumerate(train_frame_indices)
        ],
        "test": [
            {
                "kept_index": kept_idx,
                "original_frame": int(frame_idx),
                "original_time_s": float(frame_idx) / fps,
            }
            for kept_idx, frame_idx in enumerate(test_frame_indices)
        ],
    }
    with (output / "frame_map.json").open("w", encoding="utf-8") as f:
        json.dump(frame_map, f, indent=2)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "4dgs" / "experiments" / "object_only" / "runs" / "dynerf_ball_only",
    )
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=0,
        help="Masks with <= this many foreground pixels are treated as invisible.",
    )
    parser.add_argument(
        "--min-visible-cameras",
        type=int,
        default=1,
        help=(
            "For --drop-invisible-frames/--trim-empty-time-ends, keep a timestamp "
            "only if at least this many cameras have mask area > --min-mask-pixels."
        ),
    )
    parser.add_argument(
        "--trim-empty-time-ends",
        action="store_true",
        help="Trim leading/trailing timestamps where the mask is invisible in every selected camera.",
    )
    parser.add_argument(
        "--drop-invisible-frames",
        action="store_true",
        help=(
            "Drop whole timestamps that fail --min-visible-cameras. This preserves "
            "synchronized multi-view frame packets for 4DGS."
        ),
    )
    parser.add_argument(
        "--drop-invisible-views",
        action="store_true",
        help=(
            "Drop individual camera-time entries whose mask area is <= --min-mask-pixels. "
            "This breaks synchronized multi-view frame packets; use --drop-invisible-frames for 4DGS."
        ),
    )
    parser.add_argument(
        "--frame-list",
        type=Path,
        default=None,
        help=(
            "Optional text file containing one original frame index per line. "
            "Use this to force multiple per-object exports onto the exact same timeline."
        ),
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help=(
            "Ignore config train/test split: put all selected cameras and all scene frames "
            "into transforms_train.json, leaving transforms_test.json empty. Use this for "
            "final reconstruction/viewing rather than held-out evaluation."
        ),
    )
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
        min_mask_pixels=args.min_mask_pixels,
        min_visible_cameras=args.min_visible_cameras,
        trim_empty_time_ends=args.trim_empty_time_ends,
        drop_invisible_frames=args.drop_invisible_frames,
        drop_invisible_views=args.drop_invisible_views,
        frame_list=args.frame_list.resolve() if args.frame_list is not None else None,
        all_train=args.all_train,
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote {args.mode} DyNeRF dataset: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
