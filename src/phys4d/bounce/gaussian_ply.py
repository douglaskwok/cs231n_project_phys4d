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


def merge_gaussian_vertices(object_vertices: np.ndarray, background_vertices: np.ndarray) -> np.ndarray:
    """Concatenate object + background Gaussian sets (same dtype / fields)."""

    if object_vertices.dtype != background_vertices.dtype:
        raise ValueError("Object and background PLY dtypes must match")
    return np.concatenate([object_vertices, background_vertices], axis=0)
