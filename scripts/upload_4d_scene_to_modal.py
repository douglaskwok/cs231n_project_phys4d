#!/usr/bin/env python
"""Deprecated wrapper — use ``4dgs/scripts/upload_4d_scene_to_modal.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent.parent / "4dgs" / "scripts" / "upload_4d_scene_to_modal.py"
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
