"""Step 5 — translate canonical object Gaussians, merge background, rasterize test views."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gaussian_ply import (
    crop_gaussian_vertices_box,
    crop_gaussian_vertices_radius,
    filter_gaussian_vertices_max_scale,
    filter_gaussian_vertices_min_opacity,
    gaussian_xyz,
    load_gaussian_vertices,
    merge_gaussian_vertices,
    opacity_weighted_centroid,
    save_gaussian_vertices,
    scale_gaussian_vertices,
    translate_gaussian_vertices,
)
from .load_4dgs import default_wu_root, load_cfg_args
from .load_4dgs import _ensure_wu_on_path  # Wu path setup for rasterizer imports
from .physics import load_trajectory_csv

_CAM_FRAME_RE = re.compile(r"cam(\d+)_(\d+)$")


@dataclass
class Step5Result:
    """Artifacts written under ``step5/``."""

    out_dir: Path
    renders_dir: Path
    num_rendered: int
    render_mode: str
    meta_json: Path


def _read_predicted_csv(path: Path) -> dict[int, np.ndarray]:
    path = path.resolve()
    out: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["frame"]))
            out[frame] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=np.float64,
            )
    if not out:
        raise ValueError(f"No predictions in {path}")
    return out


def _parse_cam_frame(file_path: str) -> tuple[int, int]:
    stem = Path(file_path).name
    match = _CAM_FRAME_RE.search(stem)
    if not match:
        raise ValueError(f"Unsupported file_path stem: {file_path}")
    return int(match.group(1)), int(match.group(2))


def _c2w_to_rt(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-world (OpenCV/COLMAP convention) → 3DGS ``getWorld2View2`` (R, T).

    The DyNeRF export ``transforms_*.json`` already stores OpenCV camera-to-world
    matrices (camera looks down +z, y down), the same frame the background 3DGS and
    the metric object Gaussians live in, so **no** Blender/OpenGL axis flip is applied.
    ``getWorld2View2`` expects ``R`` = camera rotation (it transposes it internally to
    build world→view) and ``T`` = the world→view translation.

    The previous implementation negated ``T`` (and mirrored x), which placed every
    Gaussian *behind* the camera (view-space z < 0, ndc_z > 1) so the rasterizer culled
    the entire scene and produced a blank/white frame.
    """

    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    return R, T


def _build_minicam_from_frame(frame: dict) -> object:
    """Build Wu ``MiniCam`` from a DyNeRF ``transforms_*.json`` frame record."""

    import torch
    from scene.cameras import MiniCam
    from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix, getWorld2View2

    w = int(frame.get("w") or frame.get("width") or 0)
    h = int(frame.get("h") or frame.get("height") or 0)
    if "fl_x" in frame:
        fl_x = float(frame["fl_x"])
        fl_y = float(frame.get("fl_y", fl_x))
        fovx = focal2fov(fl_x, w)
        fovy = focal2fov(fl_y, h)
    else:
        fovx = float(frame["camera_angle_x"])
        fovy = focal2fov(fov2focal(fovx, w), h)

    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    R, T = _c2w_to_rt(c2w)
    world_view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).float().cuda()
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    time_val = float(frame.get("time", 0.0))
    return MiniCam(w, h, fovy, fovx, 0.01, 100.0, world_view, full_proj, time_val)


def _wu_rasterize_available() -> bool:
    try:
        import diff_gaussian_rasterization  # noqa: F401

        return True
    except ImportError:
        return False


def render_merged_ply_wu(
    merged_vertices: np.ndarray,
    *,
    cfg_args_path: Path,
    camera_frame: dict,
    wu_root: Path | None = None,
    white_background: bool = True,
) -> np.ndarray:
    """Rasterize merged Gaussians with Wu renderer (``stage=coarse`` — no deformation)."""

    wu = Path(wu_root or default_wu_root())
    _ensure_wu_on_path(wu)

    import torch
    from argparse import Namespace
    from gaussian_renderer import render as wu_render
    from scene.gaussian_model import GaussianModel

    args = load_cfg_args(cfg_args_path)
    gaussians = GaussianModel(3, args)
    _load_vertices_into_wu_model(gaussians, merged_vertices)

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    cam = _build_minicam_from_frame(camera_frame)
    with torch.no_grad():
        pkg = wu_render(cam, gaussians, pipe, bg, stage="coarse", cam_type=None)
        rgb = pkg["render"].clamp(0.0, 1.0).detach().cpu().numpy()
    return (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def _load_vertices_into_wu_model(gaussians: object, vertices: np.ndarray) -> None:
    """Load a merged numpy vertex buffer into an already-constructed Wu ``GaussianModel``."""

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        save_gaussian_vertices(vertices, tmp_path)
        gaussians.load_ply(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    gaussians.active_sh_degree = gaussians.max_sh_degree


class _WuCloudSession:
    """GPU session for a single Gaussian cloud (background or one object)."""

    def __init__(
        self,
        *,
        cfg_args_path: Path,
        vertices: np.ndarray,
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
        self._canonical_xyz = gaussian_xyz(vertices).astype(np.float32)

        args = load_cfg_args(cfg_args_path)
        self._gaussians = GaussianModel(3, args)
        _load_vertices_into_wu_model(self._gaussians, vertices)

        self._pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self._bg = torch.tensor(
            [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )

    def set_delta(self, delta: np.ndarray) -> None:
        delta = np.asarray(delta, dtype=np.float32).reshape(3)
        new_xyz = self._canonical_xyz + delta[None, :]
        self._gaussians._xyz.data[:] = self._torch.from_numpy(new_xyz).cuda()

    def render(self, camera_frame: dict) -> tuple[np.ndarray, np.ndarray]:
        cam = _build_minicam_from_frame(camera_frame)
        with self._torch.no_grad():
            pkg = self._wu_render(
                cam, self._gaussians, self._pipe, self._bg, stage="coarse", cam_type=None
            )
            rgb = pkg["render"].clamp(0.0, 1.0).detach().cpu().numpy()
            depth = pkg["depth"].detach().cpu().numpy()
        rgb_u8 = (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        depth_map = np.transpose(depth, (1, 2, 0)) if depth.ndim == 3 else depth
        return rgb_u8, depth_map


class _WuCompositeSession:
    """GPU session that loads object+background once and only translates object xyz per frame."""

    def __init__(
        self,
        *,
        cfg_args_path: Path,
        canonical_vertices: np.ndarray,
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
        self._canonical_xyz = gaussian_xyz(canonical_vertices).astype(np.float32)
        self._n_obj = int(self._canonical_xyz.shape[0])

        args = load_cfg_args(cfg_args_path)
        self._gaussians = GaussianModel(3, args)
        merged = merge_gaussian_vertices(canonical_vertices, background_vertices)
        _load_vertices_into_wu_model(self._gaussians, merged)

        self._pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self._bg = torch.tensor(
            [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )

    def set_object_delta(self, delta: np.ndarray) -> None:
        """Rigid translation applied equally to every object Gaussian center."""

        delta = np.asarray(delta, dtype=np.float32).reshape(3)
        new_xyz = self._canonical_xyz + delta[None, :]
        self._gaussians._xyz.data[: self._n_obj] = self._torch.from_numpy(new_xyz).cuda()

    def render(self, camera_frame: dict) -> np.ndarray:
        cam = _build_minicam_from_frame(camera_frame)
        with self._torch.no_grad():
            pkg = self._wu_render(
                cam, self._gaussians, self._pipe, self._bg, stage="coarse", cam_type=None
            )
            rgb = pkg["render"].clamp(0.0, 1.0).detach().cpu().numpy()
        return (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def _depth_scalar(depth: np.ndarray) -> np.ndarray:
    """Reduce a Wu depth buffer to a single ``(H, W)`` plane."""

    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    return arr


def _composite_layers_threshold(
    background_rgb: np.ndarray,
    object_rgbs: list[np.ndarray],
    *,
    threshold: int,
) -> np.ndarray:
    """Hard-mask object layers over background (legacy; can look sticker-like)."""

    out = background_rgb.copy()
    thr = int(threshold)
    for obj_rgb in object_rgbs:
        mask = np.max(obj_rgb, axis=2) > thr
        out[mask] = obj_rgb[mask]
    return out


def _object_alpha_from_black_bg(
    rgb: np.ndarray, threshold: int, *, gamma: float = 1.0
) -> np.ndarray:
    """Estimate an alpha matte from an object-only render on black."""

    t = float(threshold) / 255.0
    lum = np.max(rgb.astype(np.float32) / 255.0, axis=2)
    alpha = np.clip((lum - t) / max(1e-6, 1.0 - t), 0.0, 1.0)
    if gamma != 1.0:
        alpha = np.power(alpha, float(gamma))
    return alpha[..., np.newaxis]


def _composite_layers_alpha(
    background_rgb: np.ndarray,
    object_rgbs: list[np.ndarray],
    *,
    threshold: int,
    alpha_gamma: float = 1.0,
) -> np.ndarray:
    """Alpha-matte each object layer over the background."""

    out = background_rgb.astype(np.float32)
    for obj_rgb in object_rgbs:
        obj = obj_rgb.astype(np.float32)
        alpha = _object_alpha_from_black_bg(obj_rgb, threshold, gamma=alpha_gamma)
        out = obj * alpha + out * (1.0 - alpha)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _composite_layers_depth(
    background_rgb: np.ndarray,
    background_depth: np.ndarray,
    object_layers: list[tuple[np.ndarray, np.ndarray]],
    *,
    fg_threshold: int,
    alpha_gamma: float = 1.0,
    depth_slack: float = 0.08,
    alpha_fill_threshold: int | None = None,
    min_alpha: float = 0.04,
) -> np.ndarray:
    """Depth-guided alpha composite for separately rasterized layers.

    Wu/3DGS depth is a transmittance-weighted mean, so object interiors often fail
    a strict ``obj_depth < bg_depth`` test even when the alpha matte is solid (only
    silhouette edges survive).  Keep depth ordering where reliable and alpha-fill
    the interior with a softer matte threshold.
    """

    if alpha_fill_threshold is None:
        alpha_fill_threshold = max(4, int(fg_threshold) // 4)

    out = background_rgb.astype(np.float32)
    best_depth = _depth_scalar(background_depth)
    best_depth = np.where(best_depth > 1e-6, best_depth, np.inf)
    slack = float(depth_slack)

    for obj_rgb, obj_depth in object_layers:
        obj = obj_rgb.astype(np.float32)
        obj_d = _depth_scalar(obj_depth)
        valid_d = obj_d > 1e-6

        alpha_strict = _object_alpha_from_black_bg(
            obj_rgb, int(fg_threshold), gamma=alpha_gamma
        )
        alpha_fill = _object_alpha_from_black_bg(
            obj_rgb, int(alpha_fill_threshold), gamma=alpha_gamma
        )
        alpha = np.maximum(alpha_strict, alpha_fill)

        in_front_strict = valid_d & (obj_d < best_depth)
        in_front_soft = valid_d & (obj_d < best_depth * (1.0 + slack))
        strong_alpha = alpha[..., 0] >= float(min_alpha)
        # Depth wins at edges; soft depth + alpha recover the interior fill.
        use = strong_alpha & (in_front_strict | in_front_soft | (alpha[..., 0] > 0.18))
        blended = obj * alpha + out * (1.0 - alpha)
        out = np.where(use[..., np.newaxis], blended, out)

        placed = use & valid_d
        best_depth = np.where(placed, np.minimum(obj_d, best_depth), best_depth)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _boost_opacity(vertices: np.ndarray, boost: float) -> np.ndarray:
    """Multiply per-Gaussian activated opacity by ``boost`` and clamp to (0, 1)."""

    if boost is None or boost == 1.0:
        return vertices
    out = vertices.copy()
    logit = out["opacity"].astype(np.float64)
    alpha = 1.0 / (1.0 + np.exp(-logit))
    alpha = np.clip(alpha * float(boost), 1e-4, 1.0 - 1e-4)
    out["opacity"] = np.log(alpha / (1.0 - alpha)).astype(np.float32)
    return out


def _read_rgb_png(path: Path) -> np.ndarray:
    try:
        import imageio.v2 as imageio

        arr = imageio.imread(path)
    except Exception:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)
    except Exception:
        from PIL import Image

        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def _project_world_to_pixel(
    point_world: np.ndarray,
    c2w: np.ndarray,
    *,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[float, float, bool]:
    """Pinhole project; try common axis sign flips."""

    w2c = np.linalg.inv(c2w)
    ph = np.concatenate([point_world, [1.0]])
    pc = w2c @ ph
    for sx, sy, sz in ((1, 1, 1), (1, 1, -1), (1, -1, -1)):
        x, y, z = sx * pc[0], sy * pc[1], sz * pc[2]
        if z <= 1e-6:
            continue
        u = fl_x * (x / z) + cx
        v = fl_y * (y / z) + cy
        if 0 <= u < width and 0 <= v < height:
            return float(u), float(v), True
    return 0.0, 0.0, False


def _draw_disk(rgb: np.ndarray, u: float, v: float, radius: int, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    cu, cv = int(round(u)), int(round(v))
    r = max(1, int(radius))
    for yy in range(max(0, cv - r), min(h, cv + r + 1)):
        for xx in range(max(0, cu - r), min(w, cu + r + 1)):
            if (xx - cu) ** 2 + (yy - cv) ** 2 <= r * r:
                out[yy, xx] = color
    return out


def _render_overlay_fallback(
    *,
    predicted: dict[int, np.ndarray],
    frame_rec: dict,
    scene_dir: Path | None,
    sphere_radius_m: float,
    white_background: bool,
) -> np.ndarray | None:
    """Milestone3 fallback: predicted center drawn on GT / blank background."""

    cam_idx, frame_idx = _parse_cam_frame(str(frame_rec["file_path"]))
    pred = predicted.get(frame_idx)
    if pred is None:
        return None

    w = int(frame_rec.get("w", 960))
    h = int(frame_rec.get("h", 544))
    fl_x = float(frame_rec.get("fl_x", 500.0))
    fl_y = float(frame_rec.get("fl_y", fl_x))
    cx = float(frame_rec.get("cx", w / 2))
    cy = float(frame_rec.get("cy", h / 2))
    c2w = np.asarray(frame_rec["transform_matrix"], dtype=np.float64)

    if scene_dir is not None:
        rgb_path = scene_dir / "rgb" / f"cam{cam_idx:02d}" / f"frame{frame_idx:05d}.png"
        bg = _read_rgb_png(rgb_path) if rgb_path.is_file() else np.zeros((h, w, 3), dtype=np.uint8)
    else:
        bg = np.zeros((h, w, 3), dtype=np.uint8) if not white_background else np.full((h, w, 3), 255, dtype=np.uint8)

    u, v, ok = _project_world_to_pixel(
        pred, c2w, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, width=w, height=h
    )
    if not ok:
        return bg
    w2c = np.linalg.inv(c2w)
    z = float((w2c @ np.append(pred, 1.0))[2])
    z = max(abs(z), 0.05)
    radius_px = max(3, int(round(fl_x * sphere_radius_m / z)))
    return _draw_disk(bg, u, v, radius_px, (255, 64, 64))


def run_step5(
    *,
    canonical_ply: Path,
    bg_ply: Path,
    predicted_csv: Path,
    ref_traj_csv: Path,
    dynerf_export: Path,
    out_dir: Path,
    cfg_args: Path | None = None,
    scene_dir: Path | None = None,
    wu_root: Path | None = None,
    white_background: bool = True,
    force_overlay: bool = False,
    sphere_radius_m: float = 0.1,
    object_scale: float | None = None,
    object_crop_radius: float | None = None,
    object_crop_box_half_extents_m: np.ndarray | None = None,
    object_max_scale: float | None = None,
    object_min_opacity: float | None = 0.05,
    object_opacity_boost: float = 1.0,
    composite_2d: bool = True,
    composite_mode: str = "alpha",
    composite_threshold: int = 32,
    composite_alpha_gamma: float = 1.0,
    cameras: list[int] | None = None,
    skip_existing: bool = False,
) -> Step5Result:
    """Render held-out (test) views: translated object ∪ background.

    ``object_scale`` (e.g. the phase1 ``align_scale``) brings an object reconstructed
    in an arbitrary 4DGS world scale into the metric background frame: the canonical
    object is scaled about its own centroid and then placed at the predicted metric
    position each frame. When ``None`` (object already metric, e.g. the big ball) the
    object is simply translated by ``pred - p_ref``.

    By default uses layered alpha compositing (same as collision): render background
    and object in separate 3D passes, then alpha-matte the object over the background.
    """

    composite_mode = str(composite_mode).lower()
    if composite_mode not in {"alpha", "depth", "threshold", "merge3d"}:
        raise ValueError(f"unsupported composite_mode: {composite_mode}")
    if not composite_2d:
        composite_mode = "merge3d"

    out_dir = out_dir.resolve()
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    canonical = load_gaussian_vertices(canonical_ply)
    background = load_gaussian_vertices(bg_ply)
    predicted = _read_predicted_csv(predicted_csv)

    _, _, ref_positions = load_trajectory_csv(ref_traj_csv)
    p_ref = ref_positions[0].copy()
    placement_center = opacity_weighted_centroid(canonical)
    if object_crop_box_half_extents_m is not None and object_scale is not None:
        half_src = np.asarray(object_crop_box_half_extents_m, dtype=np.float64) / float(object_scale)
        canonical = crop_gaussian_vertices_box(canonical, placement_center, half_src)
    elif object_crop_radius is not None and object_crop_radius > 0:
        canonical = crop_gaussian_vertices_radius(canonical, placement_center, float(object_crop_radius))
    if object_min_opacity is not None and object_min_opacity > 0:
        canonical = filter_gaussian_vertices_min_opacity(canonical, float(object_min_opacity))
    canonical = _boost_opacity(canonical, object_opacity_boost)
    if object_scale is not None:
        canonical = scale_gaussian_vertices(canonical, float(object_scale), center=placement_center)
    if object_max_scale is not None and object_max_scale > 0:
        canonical = filter_gaussian_vertices_max_scale(canonical, float(object_max_scale))
    canonical_centroid = opacity_weighted_centroid(canonical)
    scaled = object_scale is not None

    transforms_test = dynerf_export.resolve() / "transforms_test.json"
    if not transforms_test.is_file():
        raise FileNotFoundError(f"Missing {transforms_test}")
    blob = json.loads(transforms_test.read_text(encoding="utf-8"))
    frames = blob.get("frames") or []
    if not frames:
        raise ValueError(f"No frames in {transforms_test}")

    use_wu = (not force_overlay) and _wu_rasterize_available()
    if use_wu and cfg_args is None:
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

    wu_session: _WuCompositeSession | None = None
    bg_session: _WuCloudSession | None = None
    obj_session: _WuCloudSession | None = None
    if use_wu:
        if composite_mode in {"alpha", "depth", "threshold"}:
            print(
                f"Step 5 Wu {composite_mode} composite: background ({background.shape[0]} Gaussians) + "
                f"object ({canonical.shape[0]} Gaussians)",
                flush=True,
            )
            bg_session = _WuCloudSession(
                cfg_args_path=cfg_args,
                vertices=background,
                wu_root=wu_root,
                white_background=white_background,
            )
            obj_session = _WuCloudSession(
                cfg_args_path=cfg_args,
                vertices=canonical,
                wu_root=wu_root,
                white_background=False,
            )
        else:
            print(
                f"Step 5 Wu session: {canonical.shape[0]} object + {background.shape[0]} background Gaussians",
                flush=True,
            )
            wu_session = _WuCompositeSession(
                cfg_args_path=cfg_args,
                canonical_vertices=canonical,
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
        pred = predicted.get(frame_idx)
        if pred is None:
            missing_pred += 1
            continue

        out_path = renders_dir / f"{frame_idx:04d}_cam{cam_idx}.png"
        if skip_existing and out_path.is_file():
            num_skipped += 1
            num_rendered += 1
            continue

        # Scaled object: place the (already-scaled) placement center on the predicted
        # metric position. Unscaled object (metric): pred - p_ref translation.
        delta = (pred - placement_center) if scaled else (pred - p_ref)
        camera_blob = {**blob, **frame_rec}

        if use_wu:
            if composite_mode in {"alpha", "depth", "threshold"}:
                assert bg_session is not None and obj_session is not None
                bg_rgb, bg_depth = bg_session.render(camera_blob)
                obj_session.set_delta(delta)
                obj_rgb, obj_depth = obj_session.render(camera_blob)
                if composite_mode == "alpha":
                    rgb = _composite_layers_alpha(
                        bg_rgb,
                        [obj_rgb],
                        threshold=composite_threshold,
                        alpha_gamma=composite_alpha_gamma,
                    )
                elif composite_mode == "depth":
                    rgb = _composite_layers_depth(
                        bg_rgb,
                        bg_depth,
                        [(obj_rgb, obj_depth)],
                        fg_threshold=composite_threshold,
                        alpha_gamma=composite_alpha_gamma,
                    )
                else:
                    rgb = _composite_layers_threshold(
                        bg_rgb,
                        [obj_rgb],
                        threshold=composite_threshold,
                    )
            else:
                assert wu_session is not None
                wu_session.set_object_delta(delta)
                rgb = wu_session.render(camera_blob)
        else:
            rgb = _render_overlay_fallback(
                predicted=predicted,
                frame_rec=frame_rec,
                scene_dir=scene_dir,
                sphere_radius_m=sphere_radius_m,
                white_background=white_background,
            )
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
        "object_scale": float(object_scale) if object_scale is not None else None,
        "object_crop_radius": float(object_crop_radius) if object_crop_radius else None,
        "object_crop_box_half_extents_m": (
            np.asarray(object_crop_box_half_extents_m, dtype=np.float64).tolist()
            if object_crop_box_half_extents_m is not None
            else None
        ),
        "object_max_scale": float(object_max_scale) if object_max_scale is not None else None,
        "object_min_opacity": float(object_min_opacity) if object_min_opacity is not None else None,
        "object_opacity_boost": float(object_opacity_boost),
        "num_object_gaussians": int(canonical.shape[0]),
        "num_rendered": num_rendered,
        "num_skipped_existing": num_skipped,
        "cameras_filter": sorted(cam_filter) if cam_filter else None,
        "num_test_views": len(frames),
        "num_missing_predictions": missing_pred,
        "p_ref": p_ref.tolist(),
        "p_ref_source": "ref_traj_first_row",
        "placement_center": placement_center.tolist(),
        "canonical_centroid": canonical_centroid.tolist(),
        "canonical_ply": str(canonical_ply.resolve()),
        "bg_ply": str(bg_ply.resolve()),
        "predicted_csv": str(predicted_csv.resolve()),
        "ref_traj_csv": str(ref_traj_csv.resolve()),
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
