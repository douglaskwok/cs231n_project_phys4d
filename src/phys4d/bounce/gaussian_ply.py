"""Load, translate, and merge 3D Gaussian Splatting PLY files (graphdeco / Wu layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussian_vertices(path: Path) -> np.ndarray:
    """Read the ``vertex`` table from a 3DGS ``point_cloud.ply``."""

    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plyfile is required: pip install plyfile") from exc

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    ply = PlyData.read(str(path))
    return np.asarray(ply["vertex"].data).copy()


def save_gaussian_vertices(vertices: np.ndarray, path: Path) -> None:
    """Write a 3DGS vertex buffer to PLY."""

    from plyfile import PlyData, PlyElement

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = PlyElement.describe(vertices, "vertex")
    PlyData([vertex], text=False).write(str(path))


def gaussian_xyz(vertices: np.ndarray) -> np.ndarray:
    return np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
        dtype=np.float64,
    )


def opacity_weighted_centroid(vertices: np.ndarray) -> np.ndarray:
    """Centroid using activated opacity (sigmoid on stored logits)."""

    xyz = gaussian_xyz(vertices)
    w = _sigmoid(np.asarray(vertices["opacity"], dtype=np.float64).reshape(-1))
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total < 1e-12:
        return xyz.mean(axis=0)
    return (xyz * w[:, None]).sum(axis=0) / total


def translate_gaussian_vertices(vertices: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Rigid translation of all Gaussian centers; other attributes unchanged."""

    delta = np.asarray(delta, dtype=np.float64).reshape(3)
    out = vertices.copy()
    out["x"] = out["x"] + np.float32(delta[0])
    out["y"] = out["y"] + np.float32(delta[1])
    out["z"] = out["z"] + np.float32(delta[2])
    return out


def crop_gaussian_vertices_radius(
    vertices: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Keep only Gaussians whose centers lie within ``radius`` of ``center``.

    Object-only 4DGS runs often leave a diffuse halo of low-opacity floaters around
    the reconstructed ball; cropping to a ball-sized sphere (in the *source* 4DGS
    units, before scaling) removes them so the composite shows a clean object.
    """

    center = np.asarray(center, dtype=np.float64).reshape(3)
    d = np.linalg.norm(gaussian_xyz(vertices) - center[None, :], axis=1)
    return vertices[d <= float(radius)].copy()


def scale_gaussian_vertices(
    vertices: np.ndarray,
    scale: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Isotropically scale a Gaussian set about ``center`` (default: origin).

    Used to bring an object reconstructed in an arbitrary 4DGS world scale into the
    metric/background frame. Positions scale about ``center``; the per-Gaussian
    log-scale fields (``scale_0/1/2``) get ``+ln(scale)`` so the splats keep the
    correct physical size. Centroid is preserved when ``center`` is the centroid.
    """

    import math

    s = float(scale)
    out = vertices.copy()
    center = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64).reshape(3)
    xyz = gaussian_xyz(out)
    new = center[None, :] + s * (xyz - center[None, :])
    out["x"] = new[:, 0].astype(np.float32)
    out["y"] = new[:, 1].astype(np.float32)
    out["z"] = new[:, 2].astype(np.float32)
    ln_s = math.log(s) if s > 0 else 0.0
    for field in ("scale_0", "scale_1", "scale_2"):
        if field in out.dtype.names:
            out[field] = (out[field].astype(np.float64) + ln_s).astype(np.float32)
    return out


def merge_gaussian_vertices(object_vertices: np.ndarray, background_vertices: np.ndarray) -> np.ndarray:
    """Concatenate object + background Gaussian sets (same dtype / fields)."""

    if object_vertices.dtype != background_vertices.dtype:
        raise ValueError("Object and background PLY dtypes must match")
    return np.concatenate([object_vertices, background_vertices], axis=0)
