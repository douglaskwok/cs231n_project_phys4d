#!/usr/bin/env python
"""Build slide-ready infographics from generated dataset frames."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "presentation_assets"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE = _font(54, bold=True)
SUBTITLE = _font(28)
LABEL = _font(30, bold=True)
SMALL = _font(22)


def _fit_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    img = img.convert("RGB")
    scale = max(target_w / img.width, target_h / img.height)
    resized = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _overlay_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    subtext: str | None = None,
    *,
    width: int,
) -> None:
    x, y = xy
    pad = 18
    label_h = 72 if subtext else 54
    draw.rounded_rectangle(
        [x, y, x + width, y + label_h],
        radius=18,
        fill=(8, 11, 18, 205),
    )
    draw.text((x + pad, y + 10), text, fill=(255, 255, 255), font=LABEL)
    if subtext:
        draw.text((x + pad, y + 44), subtext, fill=(210, 219, 232), font=SMALL)


def _frame(path: str) -> Image.Image:
    full = REPO_ROOT / path
    if not full.is_file():
        raise FileNotFoundError(full)
    return Image.open(full)


def build_physics_scenarios() -> Path:
    canvas = Image.new("RGB", (1920, 1080), (244, 246, 250))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((80, 48), "Physics Scenarios", fill=(18, 24, 38), font=TITLE)
    draw.text(
        (82, 108),
        "Synthetic 12-view PyBullet scenes used for dynamic reconstruction tests",
        fill=(83, 94, 112),
        font=SUBTITLE,
    )

    scenarios = [
        (
            "Bouncing Ball",
            "restitution + trajectory variation",
            "dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0/rgb/cam00/frame00045.png",
        ),
        (
            "Collision",
            "two sliding rigid boxes",
            "dataset/outputs/room_physics_12view_base/scene_0000_collision_room/rgb/cam00/frame00030.png",
        ),
        (
            "Stacking",
            "sequential block contacts",
            "dataset/outputs/room_physics_12view_base/scene_0001_stacking_room/rgb/cam00/frame00220.png",
        ),
        (
            "Deformable",
            "soft torus drop",
            "dataset/outputs/room_physics_12view_base/scene_0002_deformable_room/rgb/cam00/frame00080.png",
        ),
    ]

    tile_w, tile_h = 840, 370
    x0, y0 = 80, 180
    gap_x, gap_y = 80, 56
    for idx, (title, sub, path) in enumerate(scenarios):
        row, col = divmod(idx, 2)
        x = x0 + col * (tile_w + gap_x)
        y = y0 + row * (tile_h + gap_y)
        draw.rounded_rectangle(
            [x - 10, y - 10, x + tile_w + 10, y + tile_h + 10],
            radius=26,
            fill=(255, 255, 255, 255),
            outline=(216, 223, 235, 255),
            width=2,
        )
        img = _fit_crop(_frame(path), (tile_w, tile_h))
        canvas.paste(img, (x, y))
        _overlay_label(draw, (x + 18, y + tile_h - 92), title, sub, width=tile_w - 36)

    dst = OUT_DIR / "physics_scenarios_2x2.png"
    canvas.save(dst)
    return dst


def build_twelve_views() -> Path:
    scene = REPO_ROOT / "dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0"
    config = json.loads((scene / "config.json").read_text(encoding="utf-8"))
    names = config["cameras"]["camera_names"]
    frame = "frame00045.png"

    canvas = Image.new("RGB", (1920, 1080), (244, 246, 250))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((70, 42), "12-View Camera Rig", fill=(18, 24, 38), font=TITLE)
    draw.text(
        (72, 102),
        "Same simulation frame rendered from calibrated training and held-out viewpoints",
        fill=(83, 94, 112),
        font=SUBTITLE,
    )

    cols, rows = 4, 3
    tile_w, tile_h = 430, 244
    gap_x, gap_y = 22, 24
    x0, y0 = 60, 168
    for cam in range(12):
        row, col = divmod(cam, cols)
        x = x0 + col * (tile_w + gap_x)
        y = y0 + row * (tile_h + gap_y)
        path = scene / "rgb" / f"cam{cam:02d}" / frame
        img = _fit_crop(Image.open(path), (tile_w, tile_h))
        draw.rounded_rectangle(
            [x - 6, y - 6, x + tile_w + 6, y + tile_h + 6],
            radius=18,
            fill=(255, 255, 255, 255),
            outline=(216, 223, 235, 255),
            width=2,
        )
        canvas.paste(img, (x, y))
        label = f"cam{cam:02d}  {names[cam].replace('_', ' ')}"
        _overlay_label(draw, (x + 12, y + tile_h - 58), label, width=tile_w - 24)

    dst = OUT_DIR / "twelve_views_grid.png"
    canvas.save(dst)
    return dst


def _mask(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("L")


def _color_mask(
    *,
    ball_mask: Image.Image,
    table_mask: Image.Image | None,
    size: tuple[int, int],
) -> Image.Image:
    ball = _fit_crop(ball_mask.convert("RGB"), size).convert("L")
    table = _fit_crop(table_mask.convert("RGB"), size).convert("L") if table_mask is not None else None
    out = Image.new("RGB", size, (8, 11, 18))
    pix = out.load()
    ball_px = ball.load()
    table_px = table.load() if table is not None else None
    for y in range(size[1]):
        for x in range(size[0]):
            if table_px is not None and table_px[x, y] > 0:
                pix[x, y] = (36, 165, 104)
            if ball_px[x, y] > 0:
                pix[x, y] = (38, 124, 230)
    return out


def build_segmentation_mask_comparison() -> Path:
    scene = REPO_ROOT / "dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0"
    frame = "frame00045.png"
    cam = "cam00"
    ball_mask = _mask(scene / "masks" / cam / frame)
    table_mask = _mask(scene / "masks_table" / cam / frame)

    canvas = Image.new("RGB", (1920, 1080), (244, 246, 250))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((80, 52), "Segmentation Mask Variants", fill=(18, 24, 38), font=TITLE)
    draw.text(
        (82, 112),
        "Foreground definitions used to train object-only 4DGS inputs",
        fill=(83, 94, 112),
        font=SUBTITLE,
    )

    tile_w, tile_h = 820, 560
    y = 250
    panels = [
        ("Ball Only", "dynamic object mask", None, 80),
        ("Ball + Table", "dynamic object plus static contact surface", table_mask, 1020),
    ]
    for title, sub, maybe_table, x in panels:
        draw.rounded_rectangle(
            [x - 12, y - 12, x + tile_w + 12, y + tile_h + 120],
            radius=26,
            fill=(255, 255, 255, 255),
            outline=(216, 223, 235, 255),
            width=2,
        )
        mask_img = _color_mask(ball_mask=ball_mask, table_mask=maybe_table, size=(tile_w, tile_h))
        canvas.paste(mask_img, (x, y))
        draw.text((x, y + tile_h + 28), title, fill=(18, 24, 38), font=LABEL)
        draw.text((x, y + tile_h + 68), sub, fill=(83, 94, 112), font=SMALL)

    # Legend.
    legend_y = 190
    draw.rounded_rectangle([80, legend_y, 600, legend_y + 42], radius=16, fill=(255, 255, 255, 255))
    draw.rounded_rectangle([102, legend_y + 12, 126, legend_y + 36], radius=6, fill=(38, 124, 230, 255))
    draw.text((138, legend_y + 8), "ball foreground", fill=(42, 51, 67), font=SMALL)
    draw.rounded_rectangle([340, legend_y + 12, 364, legend_y + 36], radius=6, fill=(36, 165, 104, 255))
    draw.text((376, legend_y + 8), "table foreground", fill=(42, 51, 67), font=SMALL)

    dst = OUT_DIR / "segmentation_masks_ball_vs_ball_table.png"
    canvas.save(dst)
    return dst


def build_ball_only_no_table() -> Path:
    scene = REPO_ROOT / "dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0"
    frame = "frame00045.png"
    cam = "cam00"
    rgb = Image.open(scene / "rgb" / cam / frame)
    ball_mask = _mask(scene / "masks" / cam / frame)

    canvas = Image.new("RGB", (1920, 1080), (244, 246, 250))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((80, 52), "Ball-Only Segmentation", fill=(18, 24, 38), font=TITLE)
    draw.text(
        (82, 112),
        "Foreground contains only the dynamic object; the table and room are excluded",
        fill=(83, 94, 112),
        font=SUBTITLE,
    )

    tile_w, tile_h = 820, 560
    y = 250
    panels = [
        ("Source Frame", "RGB input from cam00", _fit_crop(rgb, (tile_w, tile_h)), 80),
        (
            "Ball Mask",
            "ball foreground only; table removed",
            _color_mask(ball_mask=ball_mask, table_mask=None, size=(tile_w, tile_h)),
            1020,
        ),
    ]
    for title, sub, img, x in panels:
        draw.rounded_rectangle(
            [x - 12, y - 12, x + tile_w + 12, y + tile_h + 120],
            radius=26,
            fill=(255, 255, 255, 255),
            outline=(216, 223, 235, 255),
            width=2,
        )
        canvas.paste(img, (x, y))
        draw.text((x, y + tile_h + 28), title, fill=(18, 24, 38), font=LABEL)
        draw.text((x, y + tile_h + 68), sub, fill=(83, 94, 112), font=SMALL)

    legend_y = 190
    draw.rounded_rectangle([80, legend_y, 350, legend_y + 42], radius=16, fill=(255, 255, 255, 255))
    draw.rounded_rectangle([102, legend_y + 12, 126, legend_y + 36], radius=6, fill=(38, 124, 230, 255))
    draw.text((138, legend_y + 8), "ball foreground", fill=(42, 51, 67), font=SMALL)

    dst = OUT_DIR / "segmentation_mask_ball_only_no_table.png"
    canvas.save(dst)
    return dst


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        build_physics_scenarios(),
        build_twelve_views(),
        build_segmentation_mask_comparison(),
        build_ball_only_no_table(),
    ]
    for path in paths:
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
