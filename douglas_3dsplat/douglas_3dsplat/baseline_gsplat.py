"""Direct gsplat baseline for the extracted 12-view static frame."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _look_at_w2c(
    eye: list[float],
    target: list[float],
    up: list[float],
    device: torch.device,
) -> torch.Tensor:
    """OpenCV-style world-to-camera matrix: x right, y down, z forward."""
    eye_t = torch.tensor(eye, dtype=torch.float32, device=device)
    target_t = torch.tensor(target, dtype=torch.float32, device=device)
    up_t = torch.tensor(up, dtype=torch.float32, device=device)

    forward = F.normalize(target_t - eye_t, dim=0)
    right = F.normalize(torch.linalg.cross(forward, up_t), dim=0)
    down = F.normalize(torch.linalg.cross(forward, right), dim=0)

    rot = torch.stack([right, down, forward], dim=0)
    trans = -(rot @ eye_t)

    view = torch.eye(4, dtype=torch.float32, device=device)
    view[:3, :3] = rot
    view[:3, 3] = trans
    return view


def _load_image(path: Path, width: int, height: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    return data.reshape(height, width, 3).to(device=device, dtype=torch.float32) / 255.0


def load_scene(
    data_dir: Path,
    *,
    downscale: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Load RGB targets, view matrices, and intrinsics from transforms.json."""
    data_dir = Path(data_dir)
    with (data_dir / "transforms.json").open("r", encoding="utf-8") as f:
        scene = json.load(f)

    base_w = int(scene["w"])
    base_h = int(scene["h"])
    width = max(1, base_w // int(downscale))
    height = max(1, base_h // int(downscale))

    images: list[torch.Tensor] = []
    viewmats: list[torch.Tensor] = []
    intrinsics: list[torch.Tensor] = []

    for frame in scene["frames"]:
        image_path = data_dir / f"{frame['file_path']}.png"
        images.append(_load_image(image_path, width, height, device))

        viewmats.append(
            _look_at_w2c(
                eye=[float(x) for x in frame["eye"]],
                target=[float(x) for x in frame["target"]],
                up=[float(x) for x in frame["up"]],
                device=device,
            )
        )

        scale_x = width / base_w
        scale_y = height / base_h
        k = torch.tensor(
            [
                [float(frame["fl_x"]) * scale_x, 0.0, float(frame["cx"]) * scale_x],
                [0.0, float(frame["fl_y"]) * scale_y, float(frame["cy"]) * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        intrinsics.append(k)

    return (
        torch.stack(images, dim=0),
        torch.stack(viewmats, dim=0),
        torch.stack(intrinsics, dim=0),
        width,
        height,
    )


def _init_gaussians(
    num_gaussians: int,
    device: torch.device,
    *,
    room_shell: bool = False,
) -> dict[str, torch.nn.Parameter]:
    means = torch.empty(num_gaussians, 3, device=device)
    if not room_shell:
        # Original 12-view scene bounds: table plus bouncing ball from frame 0.
        means[:, 0].uniform_(-0.75, 0.75)
        means[:, 1].uniform_(-0.55, 0.55)
        means[:, 2].uniform_(0.65, 1.65)
    else:
        # Room variant: seed most splats near the table/ball, with the rest on the shell.
        focus_count = max(1, int(num_gaussians * 0.65))
        room_count = num_gaussians - focus_count

        means[:focus_count, 0].uniform_(-0.75, 0.75)
        means[:focus_count, 1].uniform_(-0.55, 0.55)
        means[:focus_count, 2].uniform_(0.65, 1.65)

        if room_count > 0:
            room_half_x = 2.20
            room_half_y = 2.20
            room_height = 2.85
            room = means[focus_count:]
            room[:, 0].uniform_(-room_half_x, room_half_x)
            room[:, 1].uniform_(-room_half_y, room_half_y)
            room[:, 2].uniform_(0.0, room_height)

            faces = torch.randint(0, 6, (room_count,), device=device)
            for face_id in range(6):
                mask = faces == face_id
                if not bool(mask.any()):
                    continue
                if face_id == 0:
                    room[mask, 2] = 0.0
                elif face_id == 1:
                    room[mask, 2] = room_height
                elif face_id == 2:
                    room[mask, 1] = -room_half_y
                elif face_id == 3:
                    room[mask, 1] = room_half_y
                elif face_id == 4:
                    room[mask, 0] = -room_half_x
                else:
                    room[mask, 0] = room_half_x

    quats = torch.zeros(num_gaussians, 4, device=device)
    quats[:, 0] = 1.0
    quats += 0.01 * torch.randn_like(quats)

    log_scales = torch.full((num_gaussians, 3), math.log(0.025), device=device)
    opacity_logits = torch.full((num_gaussians,), -2.0, device=device)
    color_logits = torch.randn(num_gaussians, 3, device=device)

    return {
        "means": torch.nn.Parameter(means),
        "quats": torch.nn.Parameter(quats),
        "log_scales": torch.nn.Parameter(log_scales),
        "opacity_logits": torch.nn.Parameter(opacity_logits),
        "color_logits": torch.nn.Parameter(color_logits),
    }


def _render(
    params: dict[str, torch.nn.Parameter],
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    from gsplat.rendering import rasterization

    quats = F.normalize(params["quats"], dim=-1)
    scales = torch.exp(params["log_scales"]).clamp(0.002, 0.20)
    opacities = torch.sigmoid(params["opacity_logits"])
    colors = torch.sigmoid(params["color_logits"])
    backgrounds = torch.ones(3, device=viewmats.device)

    rendered, _alphas, _meta = rasterization(
        means=params["means"],
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        packed=True,
        backgrounds=backgrounds,
        render_mode="RGB",
        radius_clip=0.0,
    )
    return rendered[..., :3].clamp(0.0, 1.0)


def _save_grid(images: torch.Tensor, path: Path, cols: int = 4) -> None:
    images = images.detach().clamp(0.0, 1.0).cpu()
    count, height, width, channels = images.shape
    rows = math.ceil(count / cols)
    grid = torch.ones(rows * height, cols * width, channels)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = image

    array = (grid.numpy() * 255.0).round().astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def export_gaussians_ply(
    params: dict[str, torch.Tensor],
    path: Path,
) -> None:
    """Export trained params to the common 3DGS PLY layout for web viewers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sh_c0 = 0.28209479177387814
    means = params["means"].detach().cpu().numpy().astype(np.float32)
    quats = F.normalize(params["quats"].detach().cpu(), dim=-1).numpy().astype(np.float32)
    log_scales = params["log_scales"].detach().cpu().numpy().astype(np.float32)
    opacity_logits = params["opacity_logits"].detach().cpu().numpy().astype(np.float32)
    rgb = torch.sigmoid(params["color_logits"].detach().cpu()).numpy().astype(np.float32)
    f_dc = ((rgb - 0.5) / sh_c0).astype(np.float32)

    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    dtype = [(name, "f4") for name in names]
    vertices = np.empty(means.shape[0], dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    vertices["opacity"] = opacity_logits
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = (
        log_scales[:, 0],
        log_scales[:, 1],
        log_scales[:, 2],
    )
    vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = (
        quats[:, 0],
        quats[:, 1],
        quats[:, 2],
        quats[:, 3],
    )

    try:
        from plyfile import PlyData, PlyElement

        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
    except Exception:
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertices)}",
            *[f"property float {name}" for name in names],
            "end_header",
        ]
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            for row in vertices:
                f.write(" ".join(str(float(row[name])) for name in names) + "\n")


def export_debug_rgb_pointcloud_ply(
    params: dict[str, torch.Tensor],
    path: Path,
    opacity_threshold: float = 0.02,
) -> None:
    """Export a simple ASCII RGB point cloud for debugging in generic viewers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    means = params["means"].detach().cpu().numpy().astype(np.float32)
    colors = torch.sigmoid(params["color_logits"].detach().cpu()).numpy()
    opacities = torch.sigmoid(params["opacity_logits"].detach().cpu()).numpy()

    mask = opacities >= float(opacity_threshold)
    means = means[mask]
    colors = np.clip(colors[mask] * 255.0, 0.0, 255.0).round().astype(np.uint8)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(means)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(means, colors):
            f.write(
                f"{float(xyz[0])} {float(xyz[1])} {float(xyz[2])} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def train_baseline(
    *,
    data_dir: Path,
    output_dir: Path,
    steps: int = 300,
    num_gaussians: int = 1500,
    downscale: int = 4,
    lr: float = 0.02,
    seed: int = 7,
) -> dict[str, Any]:
    """Train a small static Gaussian scene with gsplat as the rasterizer."""
    if not torch.cuda.is_available():
        raise RuntimeError("This baseline expects a CUDA GPU")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    targets, viewmats, intrinsics, width, height = load_scene(
        data_dir,
        downscale=downscale,
        device=device,
    )
    with (data_dir / "transforms.json").open("r", encoding="utf-8") as f:
        scene_meta = json.load(f)
    scene_text = " ".join(
        str(scene_meta.get(key, "")) for key in ("description", "scene_context")
    ).lower()
    room_shell = "room shell" in scene_text
    params = _init_gaussians(num_gaussians, device, room_shell=room_shell)

    optimizer = torch.optim.Adam(
        [
            {"params": [params["means"]], "lr": lr},
            {"params": [params["quats"], params["log_scales"]], "lr": lr * 0.5},
            {"params": [params["opacity_logits"], params["color_logits"]], "lr": lr},
        ]
    )

    losses: list[float] = []
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        rendered = _render(params, viewmats, intrinsics, width, height)

        foreground_weight = 1.0 + 4.0 * (targets.sub(1.0).abs().mean(dim=-1, keepdim=True) > 0.05)
        l1 = (foreground_weight * (rendered - targets).abs()).mean()
        mse = (foreground_weight * (rendered - targets).square()).mean()
        loss = 0.8 * l1 + 0.2 * mse

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            params["log_scales"].clamp_(math.log(0.002), math.log(0.20))
            if room_shell:
                params["means"][:, 0].clamp_(-2.30, 2.30)
                params["means"][:, 1].clamp_(-2.30, 2.30)
                params["means"][:, 2].clamp_(-0.05, 2.95)
            else:
                params["means"][:, 2].clamp_(0.40, 2.20)

        losses.append(float(loss.detach().cpu()))
        if step % 25 == 0 or step == steps - 1:
            print(f"step {step:04d} loss={losses[-1]:.6f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    final = _render(params, viewmats, intrinsics, width, height)
    _save_grid(targets, output_dir / "targets_grid.png")
    _save_grid(final, output_dir / "render_grid.png")

    torch.save(
        {name: value.detach().cpu() for name, value in params.items()},
        output_dir / "gaussians.pt",
    )
    export_gaussians_ply(params, output_dir / "gaussians.ply")
    export_debug_rgb_pointcloud_ply(params, output_dir / "gaussians_debug_rgb.ply")
    metrics = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "steps": int(steps),
        "num_gaussians": int(num_gaussians),
        "downscale": int(downscale),
        "width": int(width),
        "height": int(height),
        "room_shell": bool(room_shell),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({**metrics, "losses": losses}, f, indent=2)
    return metrics
