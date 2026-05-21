#!/usr/bin/env python
"""Deprecated wrapper — use ``4dgs/scripts/build_4dgs_render_videos.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent.parent / "4dgs" / "scripts" / "build_4dgs_render_videos.py"
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
