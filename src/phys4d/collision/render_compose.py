"""Step 5 (collision) — composite N translated objects with background and render.

Mirrors ``phys4d.bounce.render_compose`` but loads multiple canonical objects,
translates each by its own predicted trajectory, and merges all objects with the
background into a single Wu rasterization session. Per-object scale / crop /
reference-trajectory handling matches the single-object bounce path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phys4d.bounce.gaussian_ply import (
    crop_gaussian_vertices_radius,
    crop_gaussian_vertices_box,
    filter_gaussian_vertices_max_scale,
    filter_gaussian_vertices_min_opacity,
    gaussian_xyz,
    load_gaussian_vertices,
    merge_gaussian_vertices,
    opacity_weighted_centroid,
    scale_gaussian_vertices,
)
from phys4d.bounce.load_4dgs import default_wu_root, load_cfg_args
from phys4d.bounce.load_4dgs import _ensure_wu_on_path
from phys4d.bounce.physics import load_trajectory_csv
from phys4d.bounce.render_compose import (
    _WuCloudSession,
    _boost_opacity,
    _build_minicam_from_frame,
    _composite_layers_alpha,
    _composite_layers_depth,
    _composite_layers_threshold,
    _load_vertices_into_wu_model,
    _parse_cam_frame,
    _read_predicted_csv,
    _render_overlay_fallback,
    _wu_rasterize_available,
    _write_rgb_png,
)


@dataclass
class Step5Result:
    """Artifacts written under ``step5/``."""

    out_dir: Path
    renders_dir: Path
    num_rendered: int
    render_mode: str
    meta_json: Path


@dataclass
class _ObjectAsset:
    """Prepared per-object Gaussians + placement bookkeeping."""

    vertices: np.ndarray
    placement_center: np.ndarray
    centroid: np.ndarray
    p_ref: np.ndarray
    predicted: dict[int, np.ndarray]
    scaled: bool


class _MultiCompositeSession:
    """GPU session: load all objects + background once, translate per object per frame."""

    def __init__(
        self,
        *,
        cfg_args_path: Path,
        object_vertices: list[np.ndarray],
        background_vertices: np.ndarray,
        wu_root: Path | None,
        white_background: bool,
    ) -> None:
        wu = Path(wu_root or default_wu_root())
        _ensure_wu_on_path(wu)

        import torch
        from argparse import Namespace
        from gaussian_renderer import render as wu_render
        from scene.gaussian_model import GaussianModel

        self._torch = torch
        self._wu_render = wu_render
        self._obj_xyz = [gaussian_xyz(v).astype(np.float32) for v in object_vertices]
        counts = [int(x.shape[0]) for x in self._obj_xyz]
        self._offsets = np.cumsum([0, *counts]).astype(np.int64)  # start indices per object

        args = load_cfg_args(cfg_args_path)
        self._gaussians = GaussianModel(3, args)
        merged_objects = object_vertices[0]
        for verts in object_vertices[1:]:
            merged_objects = merge_gaussian_vertices(merged_objects, verts)
        merged = merge_gaussian_vertices(merged_objects, background_vertices)
        _load_vertices_into_wu_model(self._gaussians, merged)

        self._pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self._bg = torch.tensor(
            [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )

    def set_object_delta(self, obj_index: int, delta: np.ndarray) -> None:
        start = int(self._offsets[obj_index])
        end = int(self._offsets[obj_index + 1])
        new_xyz = self._obj_xyz[obj_index] + np.asarray(delta, dtype=np.float32).reshape(1, 3)
        self._gaussians._xyz.data[start:end] = self._torch.from_numpy(new_xyz).cuda()

    def render(self, camera_frame: dict) -> np.ndarray:
        cam = _build_minicam_from_frame(camera_frame)
        with self._torch.no_grad():
            pkg = self._wu_render(
                cam, self._gaussians, self._pipe, self._bg, stage="coarse", cam_type=None
            )
            rgb = pkg["render"].clamp(0.0, 1.0).detach().cpu().numpy()
        return (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def _prepare_object(
    *,
    canonical_ply: Path,
    predicted_csv: Path,
    ref_traj_csv: Path,
    object_scale: float | None,
    object_crop_radius: float | None,
    object_crop_box_half_extents_m: np.ndarray | None = None,
    object_max_scale: float | None = None,
    object_min_opacity: float | None = 0.05,
    opacity_boost: float = 1.0,
) -> _ObjectAsset:
    canonical = load_gaussian_vertices(canonical_ply)
    predicted = _read_predicted_csv(predicted_csv)
    _, _, ref_positions = load_trajectory_csv(ref_traj_csv)
    p_ref = ref_positions[0].copy()

    anchor = opacity_weighted_centroid(canonical)
    if object_crop_box_half_extents_m is not None and object_scale is not None:
        half_src = np.asarray(object_crop_box_half_extents_m, dtype=np.float64) / float(object_scale)
        canonical = crop_gaussian_vertices_box(canonical, anchor, half_src)
    elif object_crop_radius is not None and object_crop_radius > 0:
        canonical = crop_gaussian_vertices_radius(canonical, anchor, float(object_crop_radius))
    if object_min_opacity is not None and object_min_opacity > 0:
        canonical = filter_gaussian_vertices_min_opacity(canonical, float(object_min_opacity))
    canonical = _boost_opacity(canonical, opacity_boost)
    if object_scale is not None:
        canonical = scale_gaussian_vertices(canonical, float(object_scale), center=anchor)
    if object_max_scale is not None and object_max_scale > 0:
        canonical = filter_gaussian_vertices_max_scale(canonical, float(object_max_scale))

    return _ObjectAsset(
        vertices=canonical,
        placement_center=anchor.copy(),
        centroid=opacity_weighted_centroid(canonical),
        p_ref=p_ref,
        predicted=predicted,
        scaled=object_scale is not None,
    )


def run_step5(
    *,
    canonical_plys: list[Path],
    bg_ply: Path,
    predicted_csvs: list[Path],
    ref_traj_csvs: list[Path],
    dynerf_export: Path,
    out_dir: Path,
    cfg_args: Path | None = None,
    scene_dir: Path | None = None,
    wu_root: Path | None = None,
    white_background: bool = True,
    force_overlay: bool = False,
    sphere_radius_m: float = 0.1,
    object_scales: list[float | None] | None = None,
    object_crop_radii: list[float | None] | None = None,
    object_crop_box_half_extents_m: list[np.ndarray | None] | None = None,
    object_max_scales: list[float | None] | None = None,
    object_min_opacities: list[float | None] | None = None,
    object_opacity_boost: float = 1.0,
    composite_2d: bool = True,
    composite_mode: str = "alpha",
    composite_threshold: int = 48,
    composite_alpha_gamma: float = 1.0,
    cfg_args_list: list[Path] | None = None,
    cameras: list[int] | None = None,
    skip_existing: bool = False,
) -> Step5Result:
    """Render held-out (test) views compositing N translated objects + background."""

    n_obj = len(canonical_plys)
    if not (len(predicted_csvs) == len(ref_traj_csvs) == n_obj):
        raise ValueError("canonical_plys, predicted_csvs, ref_traj_csvs must have equal length")
    if object_scales is None:
        object_scales = [None] * n_obj
    if object_crop_radii is None:
        object_crop_radii = [None] * n_obj
    if object_crop_box_half_extents_m is None:
        object_crop_box_half_extents_m = [None] * n_obj
    if object_max_scales is None:
        object_max_scales = [None] * n_obj
    if object_min_opacities is None:
        object_min_opacities = [0.05] * n_obj
    if len(object_scales) != n_obj:
        raise ValueError("object_scales must match number of objects")
    if len(object_crop_radii) != n_obj:
        raise ValueError("object_crop_radii must match number of objects")
    if len(object_crop_box_half_extents_m) == 1 and n_obj > 1:
        object_crop_box_half_extents_m = object_crop_box_half_extents_m * n_obj
    if len(object_crop_box_half_extents_m) != n_obj:
        raise ValueError("object_crop_box_half_extents_m must match number of objects")
    if len(object_max_scales) != n_obj:
        raise ValueError("object_max_scales must match number of objects")
    if len(object_min_opacities) != n_obj:
        raise ValueError("object_min_opacities must match number of objects")
    if cfg_args_list is None:
        cfg_args_list = [cfg_args] * n_obj if cfg_args is not None else [None] * n_obj
    if len(cfg_args_list) == 1 and n_obj > 1:
        cfg_args_list = cfg_args_list * n_obj
    if len(cfg_args_list) != n_obj:
        raise ValueError("cfg_args_list must match number of objects")
    composite_mode = str(composite_mode).lower()
    if composite_mode not in {"alpha", "depth", "threshold", "merge3d"}:
        raise ValueError(f"unsupported composite_mode: {composite_mode}")
    if not composite_2d:
        composite_mode = "merge3d"

    out_dir = out_dir.resolve()
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    assets = [
        _prepare_object(
            canonical_ply=canonical_plys[k],
            predicted_csv=predicted_csvs[k],
            ref_traj_csv=ref_traj_csvs[k],
            object_scale=object_scales[k],
            object_crop_radius=object_crop_radii[k],
            object_crop_box_half_extents_m=object_crop_box_half_extents_m[k],
            object_max_scale=object_max_scales[k],
            object_min_opacity=object_min_opacities[k],
            opacity_boost=object_opacity_boost,
        )
        for k in range(n_obj)
    ]

    transforms_test = dynerf_export.resolve() / "transforms_test.json"
    if not transforms_test.is_file():
        raise FileNotFoundError(f"Missing {transforms_test}")
    blob = json.loads(transforms_test.read_text(encoding="utf-8"))
    frames = blob.get("frames") or []
    if not frames:
        raise ValueError(f"No frames in {transforms_test}")

    use_wu = (not force_overlay) and _wu_rasterize_available()
    if use_wu and cfg_args_list[0] is None:
        raise ValueError("Wu rasterization requires --cfg-args from the training run")
    if use_wu:
        import torch

        if not torch.cuda.is_available():
            use_wu = False

    if use_wu and composite_mode in {"alpha", "depth", "threshold"}:
        render_mode = f"wu_3dgs_{composite_mode}_composite"
    elif use_wu:
        render_mode = "wu_3dgs"
    else:
        render_mode = "overlay_fallback"
    num_rendered = 0
    num_skipped = 0
    missing_pred = 0

    wu_session: _MultiCompositeSession | None = None
    bg_session: _WuCloudSession | None = None
    obj_sessions: list[_WuCloudSession] | None = None
    if use_wu:
        total_obj_gauss = sum(int(a.vertices.shape[0]) for a in assets)
        background = load_gaussian_vertices(bg_ply)
        if composite_mode in {"alpha", "depth", "threshold"}:
            print(
                f"Step 5 Wu {composite_mode} composite: background ({background.shape[0]} Gaussians) + "
                f"{n_obj} object layers ({total_obj_gauss} Gaussians total)",
                flush=True,
            )
            bg_session = _WuCloudSession(
                cfg_args_path=cfg_args_list[0],
                vertices=background,
                wu_root=wu_root,
                white_background=white_background,
            )
            obj_sessions = [
                _WuCloudSession(
                    cfg_args_path=cfg_args_list[k],
                    vertices=a.vertices,
                    wu_root=wu_root,
                    white_background=False,
                )
                for k, a in enumerate(assets)
            ]
        else:
            print(
                f"Step 5 Wu session: {total_obj_gauss} object ({n_obj} objects) + "
                f"{background.shape[0]} background Gaussians",
                flush=True,
            )
            wu_session = _MultiCompositeSession(
                cfg_args_path=cfg_args_list[0],
                object_vertices=[a.vertices for a in assets],
                background_vertices=background,
                wu_root=wu_root,
                white_background=white_background,
            )

    cam_filter = set(cameras) if cameras else None
    total_views = len(frames)

    for view_i, frame_rec in enumerate(frames):
        cam_idx, frame_idx = _parse_cam_frame(str(frame_rec["file_path"]))
        if cam_filter is not None and cam_idx not in cam_filter:
            continue

        preds = [a.predicted.get(frame_idx) for a in assets]
        if any(p is None for p in preds):
            missing_pred += 1
            continue

        out_path = renders_dir / f"{frame_idx:04d}_cam{cam_idx}.png"
        if skip_existing and out_path.is_file():
            num_skipped += 1
            num_rendered += 1
            continue

        camera_blob = {**blob, **frame_rec}
        if use_wu:
            deltas = [
                (preds[k] - asset.placement_center) if asset.scaled else (preds[k] - asset.p_ref)
                for k, asset in enumerate(assets)
            ]
            if composite_mode in {"alpha", "depth", "threshold"}:
                assert bg_session is not None and obj_sessions is not None
                bg_rgb, bg_depth = bg_session.render(camera_blob)
                obj_layers: list[tuple[np.ndarray, np.ndarray]] = []
                obj_rgbs: list[np.ndarray] = []
                for session, delta in zip(obj_sessions, deltas):
                    session.set_delta(delta)
                    obj_rgb, obj_depth = session.render(camera_blob)
                    obj_layers.append((obj_rgb, obj_depth))
                    obj_rgbs.append(obj_rgb)
                if composite_mode == "alpha":
                    rgb = _composite_layers_alpha(
                        bg_rgb,
                        obj_rgbs,
                        threshold=composite_threshold,
                        alpha_gamma=composite_alpha_gamma,
                    )
                elif composite_mode == "depth":
                    rgb = _composite_layers_depth(
                        bg_rgb,
                        bg_depth,
                        obj_layers,
                        fg_threshold=composite_threshold,
                        alpha_gamma=composite_alpha_gamma,
                    )
                else:
                    rgb = _composite_layers_threshold(
                        bg_rgb, obj_rgbs, threshold=composite_threshold
                    )
            else:
                assert wu_session is not None
                for k, delta in enumerate(deltas):
                    wu_session.set_object_delta(k, delta)
                rgb = wu_session.render(camera_blob)
        else:
            # Overlay fallback: draw each object's projected disk in turn.
            rgb = None
            for asset in assets:
                rgb = _render_overlay_fallback(
                    predicted=asset.predicted,
                    frame_rec=frame_rec if rgb is None else {**frame_rec},
                    scene_dir=scene_dir if rgb is None else None,
                    sphere_radius_m=sphere_radius_m,
                    white_background=white_background,
                )
                if rgb is None:
                    break
            if rgb is None:
                missing_pred += 1
                continue

        _write_rgb_png(out_path, rgb)
        num_rendered += 1
        if num_rendered % 10 == 0 or num_rendered == 1:
            print(
                f"  rendered {num_rendered} views (frame {frame_idx}, cam {cam_idx}, "
                f"{view_i + 1}/{total_views})",
                flush=True,
            )

    meta = {
        "render_mode": render_mode,
        "composite_2d": composite_mode in {"alpha", "depth", "threshold"},
        "composite_mode": composite_mode,
        "composite_threshold": int(composite_threshold),
        "composite_alpha_gamma": float(composite_alpha_gamma),
        "num_objects": n_obj,
        "object_scales": [float(s) if s is not None else None for s in object_scales],
        "object_crop_radii": [float(r) if r else None for r in object_crop_radii],
        "object_max_scales": [float(s) if s is not None else None for s in object_max_scales],
        "object_min_opacities": [float(o) if o is not None else None for o in object_min_opacities],
        "object_opacity_boost": float(object_opacity_boost),
        "num_object_gaussians": [int(a.vertices.shape[0]) for a in assets],
        "num_rendered": num_rendered,
        "num_skipped_existing": num_skipped,
        "cameras_filter": sorted(cam_filter) if cam_filter else None,
        "num_test_views": len(frames),
        "num_missing_predictions": missing_pred,
        "p_ref": [a.p_ref.tolist() for a in assets],
        "placement_center": [a.placement_center.tolist() for a in assets],
        "canonical_centroid": [a.centroid.tolist() for a in assets],
        "canonical_plys": [str(p.resolve()) for p in canonical_plys],
        "bg_ply": str(bg_ply.resolve()),
        "predicted_csvs": [str(p.resolve()) for p in predicted_csvs],
        "ref_traj_csvs": [str(p.resolve()) for p in ref_traj_csvs],
        "dynerf_export": str(dynerf_export.resolve()),
    }
    meta_json = out_dir / "step5_meta.json"
    meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if num_rendered == 0:
        raise RuntimeError("No views rendered; check predictions and transforms_test.json")

    return Step5Result(
        out_dir=out_dir,
        renders_dir=renders_dir,
        num_rendered=num_rendered,
        render_mode=render_mode,
        meta_json=meta_json,
    )
