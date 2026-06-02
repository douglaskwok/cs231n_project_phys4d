#!/usr/bin/env python
"""Select likely ball Gaussians by projecting centers into multiview masks."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from dataclasses import dataclass


@dataclass
class GaussianCloud:
    vertices: np.ndarray

    @property
    def xyz(self) -> np.ndarray:
        return np.stack(
            [self.vertices["x"], self.vertices["y"], self.vertices["z"]], axis=1
        )


def load_cameras(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(json.load(f)["cameras"])


def load_gaussian_ply(path: Path) -> GaussianCloud:
    try:
        from plyfile import PlyData

        return GaussianCloud(vertices=np.asarray(PlyData.read(str(path))["vertex"].data))
    except ImportError:
        return GaussianCloud(vertices=_read_ply_without_plyfile(path))


def save_gaussian_ply(cloud: GaussianCloud, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from plyfile import PlyData, PlyElement

        PlyData([PlyElement.describe(cloud.vertices, "vertex")], text=False).write(str(path))
    except ImportError:
        _write_ascii_ply(path, cloud.vertices)


def _read_ply_without_plyfile(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_bytes = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Malformed PLY with no end_header: {path}")
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break
        payload_offset = f.tell()

    header = header_bytes.decode("ascii", errors="strict").splitlines()
    if not header or header[0].strip() != "ply":
        raise ValueError(f"Not a PLY file: {path}")
    fmt = _parse_ply_format(header, path)
    if fmt == "ascii":
        return _read_ascii_ply_from_header(path, header, payload_offset)
    return _read_binary_ply_from_header(path, header, payload_offset, fmt)


def _parse_ply_format(header: list[str], path: Path) -> str:
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "format":
            if parts[1] in {"ascii", "binary_little_endian", "binary_big_endian"}:
                return parts[1]
            raise ValueError(f"Unsupported PLY format {parts[1]!r}: {path}")
    raise ValueError(f"Malformed PLY with no format line: {path}")


def _vertex_properties_from_header(header: list[str], path: Path) -> tuple[int, list[tuple[str, str]]]:
    props: list[str] = []
    count = None
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
            in_vertex = True
        elif parts and parts[0] == "element":
            in_vertex = False
        elif in_vertex and len(parts) == 3 and parts[0] == "property":
            props.append((parts[2], parts[1]))
        elif in_vertex and len(parts) > 3 and parts[0] == "property" and parts[1] == "list":
            raise ValueError(f"Unsupported list property in vertex element: {path}")
    if count is None:
        raise ValueError(f"Malformed PLY with no vertex element: {path}")
    return count, props


def _read_ascii_ply_from_header(path: Path, header: list[str], payload_offset: int) -> np.ndarray:
    count, props = _vertex_properties_from_header(header, path)
    dtype = [(name, _numpy_dtype_for_ply_type(ply_type, "=")) for name, ply_type in props]
    out = np.zeros(count, dtype=dtype)
    with path.open("rb") as f:
        f.seek(payload_offset)
        for row_idx in range(count):
            line = f.readline().decode("ascii", errors="strict")
            values = line.split()
            if len(values) < len(props):
                raise ValueError(f"Malformed ASCII PLY vertex row {row_idx}: {path}")
            for (name, _ply_type), value in zip(props, values):
                out[name][row_idx] = value
    return out


def _read_binary_ply_from_header(
    path: Path,
    header: list[str],
    payload_offset: int,
    fmt: str,
) -> np.ndarray:
    count, props = _vertex_properties_from_header(header, path)
    endian = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(name, _numpy_dtype_for_ply_type(ply_type, endian)) for name, ply_type in props])
    with path.open("rb") as f:
        f.seek(payload_offset)
        out = np.fromfile(f, dtype=dtype, count=count)
    if out.shape[0] != count:
        raise ValueError(f"Binary PLY ended before reading {count} vertices: {path}")
    return out


def _numpy_dtype_for_ply_type(ply_type: str, endian: str) -> str:
    aliases = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    code = aliases.get(ply_type)
    if code is None:
        raise ValueError(f"Unsupported PLY property type: {ply_type}")
    if code.endswith("1"):
        return code
    return endian + code


def _read_ascii_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_bytes = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Malformed ASCII PLY with no end_header: {path}")
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break
        payload_offset = f.tell()
    header = header_bytes.decode("ascii", errors="strict").splitlines()
    return _read_ascii_ply_from_header(path, header, payload_offset)


def _write_ascii_ply(path: Path, vertices: np.ndarray) -> None:
    names = list(vertices.dtype.names or [])
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        for name in names:
            f.write(f"property {_ply_type_for_numpy_dtype(vertices.dtype[name])} {name}\n")
        f.write("end_header\n")
        for row in vertices:
            f.write(" ".join(_format_ply_scalar(row[name]) for name in names) + "\n")


def _ply_type_for_numpy_dtype(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype.kind == "f" and dtype.itemsize <= 4:
        return "float"
    if dtype.kind == "f":
        return "double"
    if dtype.kind == "i":
        return {1: "char", 2: "short", 4: "int"}.get(dtype.itemsize, "int")
    if dtype.kind == "u":
        return {1: "uchar", 2: "ushort", 4: "uint"}.get(dtype.itemsize, "uint")
    return "float"


def _format_ply_scalar(value: np.ndarray | np.generic | float | int) -> str:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (int, np.integer)):
        return str(int(scalar))
    return repr(float(scalar))


def _read_mask(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    mask = imageio.imread(path)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def _project_pybullet_points(points: np.ndarray, camera_record: dict) -> tuple[np.ndarray, np.ndarray]:
    view = np.array(camera_record["view_matrix_row_major"], dtype=np.float64).reshape(4, 4)
    proj = np.array(camera_record["projection_matrix_row_major"], dtype=np.float64).reshape(4, 4)
    width, height = int(camera_record["image_size"][0]), int(camera_record["image_size"][1])
    pts = np.asarray(points, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    clip = (hom @ view) @ proj
    w = clip[:, 3]
    valid_w = np.abs(w) > 1e-8
    ndc = np.zeros((pts.shape[0], 3), dtype=np.float64)
    ndc[valid_w] = clip[valid_w, :3] / w[valid_w, None]
    u = (ndc[:, 0] * 0.5 + 0.5) * width
    v = (1.0 - ndc[:, 1]) * 0.5 * height
    visible = (
        valid_w
        & (ndc[:, 2] >= -1.0)
        & (ndc[:, 2] <= 1.0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    return np.stack([u, v], axis=1), visible


def filter_points_by_mask_union(
    points: np.ndarray,
    *,
    masks_root: Path,
    frame: int,
    min_camera_hits: int,
    pybullet_cameras: list[dict],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    hits = np.zeros(pts.shape[0], dtype=np.int32)
    frame_tag = f"frame{frame:05d}.png"
    for cam in pybullet_cameras:
        ci = int(cam["index"])
        mask_path = masks_root / f"cam{ci:02d}" / frame_tag
        if not mask_path.is_file():
            continue
        mask = _read_mask(mask_path)
        uv, visible = _project_pybullet_points(pts, cam)
        vis_idx = np.where(visible)[0]
        if vis_idx.size == 0:
            continue
        u = np.clip(np.round(uv[vis_idx, 0]).astype(np.int32), 0, mask.shape[1] - 1)
        v = np.clip(np.round(uv[vis_idx, 1]).astype(np.int32), 0, mask.shape[0] - 1)
        hits[vis_idx[mask[v, u] > 0]] += 1
    return hits >= int(min_camera_hits)


def _opacity_keep(cloud: GaussianCloud, percentile: float) -> np.ndarray:
    if "opacity" not in cloud.vertices.dtype.names:
        return np.ones(len(cloud.vertices), dtype=bool)
    op = np.asarray(cloud.vertices["opacity"], dtype=np.float64)
    return op >= np.percentile(op, float(percentile))


def select_by_masks(
    *,
    ply: Path,
    masks_root: Path,
    cameras_json: Path,
    frame: int,
    min_camera_hits: int,
    opacity_percentile: float,
    out_dir: Path,
) -> dict:
    cloud = load_gaussian_ply(ply)
    cameras = load_cameras(cameras_json)

    opacity_keep = _opacity_keep(cloud, opacity_percentile)
    candidate_idx = np.where(opacity_keep)[0]
    candidates = cloud.xyz[candidate_idx]

    mask_keep_candidates = filter_points_by_mask_union(
        candidates,
        masks_root=masks_root,
        frame=frame,
        min_camera_hits=min_camera_hits,
        pybullet_cameras=cameras,
    )
    ball_keep = np.zeros(len(cloud.vertices), dtype=bool)
    ball_keep[candidate_idx[mask_keep_candidates]] = True

    ball = GaussianCloud(vertices=cloud.vertices[ball_keep].copy())
    background = GaussianCloud(vertices=cloud.vertices[~ball_keep].copy())

    out_dir.mkdir(parents=True, exist_ok=True)
    ball_ply = out_dir / "ball_gaussians.ply"
    bg_ply = out_dir / "background_gaussians.ply"
    save_gaussian_ply(ball, ball_ply)
    save_gaussian_ply(background, bg_ply)

    report = {
        "ply": str(ply),
        "masks_root": str(masks_root),
        "cameras_json": str(cameras_json),
        "frame": int(frame),
        "min_camera_hits": int(min_camera_hits),
        "opacity_percentile": float(opacity_percentile),
        "total_gaussians": int(len(cloud.vertices)),
        "opacity_candidates": int(opacity_keep.sum()),
        "selected_ball_gaussians": int(ball_keep.sum()),
        "selected_fraction": float(ball_keep.mean()) if len(ball_keep) else 0.0,
        "ball_ply": str(ball_ply),
        "background_ply": str(bg_ply),
    }
    with (out_dir / "selection_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _write_tiny_ply(path: Path, xyz: np.ndarray) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("opacity", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    verts = np.zeros(len(xyz), dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["opacity"] = 10.0
    verts["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_ascii_ply(path, verts)


def run_smoke_test() -> int:
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ply = root / "toy.ply"
        masks = root / "masks"
        cam_dir = masks / "cam00"
        cam_dir.mkdir(parents=True)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[24:40, 24:40] = 255
        imageio.imwrite(cam_dir / "frame00000.png", mask)

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
        report = select_by_masks(
            ply=ply,
            masks_root=masks,
            cameras_json=cameras_json,
            frame=0,
            min_camera_hits=1,
            opacity_percentile=0.0,
            out_dir=root / "out",
        )
        print(json.dumps(report, indent=2))
        if report["selected_ball_gaussians"] < 1:
            raise RuntimeError("Smoke test selected no Gaussians")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--masks-root", type=Path)
    parser.add_argument("--cameras-json", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--min-camera-hits", type=int, default=2)
    parser.add_argument("--opacity-percentile", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("segmentation/01_mask_guided_gaussian_binding/runs/latest"))
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

    report = select_by_masks(
        ply=args.ply.resolve(),
        masks_root=args.masks_root.resolve(),
        cameras_json=args.cameras_json.resolve(),
        frame=args.frame,
        min_camera_hits=args.min_camera_hits,
        opacity_percentile=args.opacity_percentile,
        out_dir=args.out_dir.resolve(),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
