#!/usr/bin/env python
"""Build an interactive HTML viewer to scrub 4DGS renders through time.

Works with PNG folders from ``render_4dgs_trajectory.py`` (Modal ``--render-4d``)
or downloaded ``4dgs_renders/latest``.

Example::

    python 4dgs/scripts/view_4dgs_time.py build \\
      --render-dir 4dgs_renders/latest \\
      --dataset outputs/sphere_bounce_m2/dynerf_sphere_bounce \\
      --out 4dgs_renders/latest/viewer

    python 4dgs/scripts/view_4dgs_time.py build --render-dir path/to/pngs --open
    python 4dgs/scripts/view_4dgs_time.py serve --dir 4dgs_renders/latest/viewer
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import sys
import webbrowser
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_FOURDGS_DIR = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from render_index import index_render_dir  # noqa: E402

_TEMPLATE = _FOURDGS_DIR / "viewer" / "viewer_template.html"


def _build_html(manifest: dict, *, render_prefix: str, gt_prefix: str) -> str:
    template = _TEMPLATE.read_text(encoding="utf-8")
    manifest_json = json.dumps(manifest, indent=2)
    return (
        template.replace("__MANIFEST_JSON__", manifest_json)
        .replace("__RENDER_PREFIX__", json.dumps(render_prefix))
        .replace("__GT_PREFIX__", json.dumps(gt_prefix))
    )


def cmd_build(args: argparse.Namespace) -> int:
    render_dir = args.render_dir.resolve()
    out_dir = (args.out or render_dir / "viewer").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset.resolve() if args.dataset else None
    manifest = index_render_dir(render_dir, dataset=dataset)

    if args.copy_renders:
        dest = out_dir / "renders"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(render_dir, dest)
        render_prefix = "renders/"
        manifest["render_dir"] = str(dest)
    else:
        try:
            rel = Path(os.path.relpath(render_dir, out_dir))
            render_prefix = f"{rel.as_posix()}/" if rel != Path(".") else ""
        except ValueError:
            render_prefix = ""

    gt_prefix = ""
    if dataset and manifest.get("gt_frames"):
        try:
            rel_ds = Path(os.path.relpath(dataset, out_dir))
            gt_prefix = f"{rel_ds.as_posix()}/" if rel_ds != Path(".") else ""
        except ValueError:
            gt_prefix = ""

    html = _build_html(manifest, render_prefix=render_prefix, gt_prefix=gt_prefix)
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {index_path}")
    print(f"  {len(manifest['frames'])} time steps, cameras {manifest['camera_ids']}")
    if manifest.get("gt_frames"):
        print("  Ground-truth comparison enabled (toggle in viewer)")
    print("Open in browser:")
    print(f"  file://{index_path}")
    print("Or serve (recommended if images do not load via file://):")
    print(f"  python 4dgs/scripts/view_4dgs_time.py serve --dir {out_dir}")

    if args.open:
        webbrowser.open(index_path.as_uri())

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    root = args.dir.resolve()
    if not (root / "index.html").is_file():
        print(f"Missing {root / 'index.html'} — run build first.", file=sys.stderr)
        return 1

    root_str = str(root)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, request, client_address, server):
            super().__init__(request, client_address, server, directory=root_str)

        def log_message(self, fmt, *fargs):  # noqa: D102
            if args.verbose:
                super().log_message(fmt, *fargs)

    httpd = http.server.ThreadingHTTPServer((args.host, args.port), QuietHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {root} at {url}")
    print("Ctrl+C to stop")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Generate index.html + manifest.json")
    build.add_argument(
        "--render-dir",
        type=Path,
        required=True,
        help="Folder of PNGs from render_4dgs_trajectory (e.g. 4dgs_renders/latest).",
    )
    build.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="DyNeRF folder (transforms_*.json + images/) for timestamps and GT compare.",
    )
    build.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output viewer directory (default: <render-dir>/viewer).",
    )
    build.add_argument(
        "--copy-renders",
        action="store_true",
        help="Copy PNGs into viewer/renders/ for a self-contained folder.",
    )
    build.add_argument("--open", action="store_true", help="Open index.html in the default browser.")
    build.set_defaults(func=cmd_build)

    serve = sub.add_parser("serve", help="HTTP server for the viewer directory")
    serve.add_argument("--dir", type=Path, required=True, help="Viewer directory containing index.html.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true")
    serve.add_argument("--verbose", action="store_true")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
