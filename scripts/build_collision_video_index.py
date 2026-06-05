#!/usr/bin/env python3
"""Build a lightweight HTML preview page for collision scene videos."""

from __future__ import annotations

import argparse
import html
from pathlib import Path


DEFAULT_CAMERAS = ("cam00_front", "cam02_right", "cam08_top", "cam10_low_front_left")
VIDEO_GROUPS = (
    ("rgb", "RGB"),
    ("masks_object_a", "Object A mask"),
    ("masks_object_b", "Object B mask"),
    ("masks_object_table", "Objects + table"),
)


def video_tag(path: Path, root: Path) -> str:
    if not path.is_file():
        return '<div class="missing">missing</div>'
    rel = html.escape(path.relative_to(root).as_posix())
    return f'<video controls muted loop preload="metadata" src="{rel}"></video>'


def build_index(root: Path, cameras: tuple[str, ...]) -> str:
    rows: list[str] = []
    for scene in sorted(root.glob("scene_*")):
        if not scene.is_dir():
            continue
        rows.append(f'<section><h2>{html.escape(scene.name)}</h2>')
        for cam in cameras:
            rows.append(f'<h3>{html.escape(cam)}</h3><div class="grid">')
            for group, label in VIDEO_GROUPS:
                mp4 = scene / "videos" / group / f"{cam}.mp4"
                rows.append(
                    '<div class="card">'
                    f'<div class="label">{html.escape(label)}</div>'
                    f"{video_tag(mp4, root)}"
                    "</div>"
                )
            rows.append("</div>")
        rows.append("</section>")

    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Collision 2x2x2 Preview</title>
<style>
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #111; color: #eee; }
header { position: sticky; top: 0; background: #181818; border-bottom: 1px solid #333; padding: 16px 20px; z-index: 2; }
h1 { margin: 0; font-size: 20px; }
h2 { margin: 28px 0 12px; font-size: 18px; }
h3 { margin: 18px 0 8px; font-size: 14px; color: #bbb; }
section { padding: 0 20px 28px; border-bottom: 1px solid #2a2a2a; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; }
.card { background: #1c1c1c; border: 1px solid #333; border-radius: 6px; overflow: hidden; }
.label { padding: 8px 10px; font-size: 12px; color: #ddd; border-bottom: 1px solid #333; }
video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #000; }
.missing { display: grid; place-items: center; aspect-ratio: 16 / 9; color: #888; background: #080808; }
@media (max-width: 1000px) { .grid { grid-template-columns: repeat(2, minmax(160px, 1fr)); } }
@media (max-width: 560px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
<h1>Collision 2x2x2 Preview</h1>
</header>
""" + "\n".join(rows) + """
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Merged collision dataset root.")
    parser.add_argument("--out", default="index.html", help="Output HTML path, relative to root unless absolute.")
    parser.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.write_text(build_index(root, tuple(args.cameras)), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
