"""Unit tests for Step 5 Gaussian PLY helpers (no GPU)."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HAS_PLYFILE = importlib.util.find_spec("plyfile") is not None

from phys4d.bounce.gaussian_ply import (
    gaussian_xyz,
    load_gaussian_vertices,
    merge_gaussian_vertices,
    save_gaussian_vertices,
    translate_gaussian_vertices,
)


def _make_vertices(n: int = 5) -> np.ndarray:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("opacity", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )
    v = np.zeros(n, dtype=dtype)
    v["x"] = np.linspace(0, 1, n)
    v["opacity"] = 2.0
    v["rot_0"] = 1.0
    return v


class TestStep5GaussianPly(unittest.TestCase):
    def test_translate_and_merge(self) -> None:
        obj = _make_vertices(3)
        bg = _make_vertices(2)
        shifted = translate_gaussian_vertices(obj, np.array([1.0, 2.0, 3.0]))
        xyz = gaussian_xyz(shifted)
        np.testing.assert_allclose(xyz[0], [1.0, 2.0, 3.0])
        merged = merge_gaussian_vertices(shifted, bg)
        self.assertEqual(merged.shape[0], 5)

    @unittest.skipUnless(_HAS_PLYFILE, "plyfile not installed")
    def test_ply_roundtrip(self) -> None:
        v = _make_vertices(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            save_gaussian_vertices(v, path)
            loaded = load_gaussian_vertices(path)
            self.assertEqual(loaded.shape[0], 4)
            np.testing.assert_allclose(loaded["x"], v["x"])


if __name__ == "__main__":
    unittest.main()
