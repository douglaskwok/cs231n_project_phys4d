"""Collision pipeline: multi-body (N-sphere) extension of the bounce pipeline.

Reuses ``phys4d.bounce`` for extraction, Gaussian PLY handling, and per-object
metrics; adds a coupled multi-body simulator and multi-object compositing.
"""

from .physics import Body, SceneParams, count_pair_collisions, load_trajectory_csv, simulate_scene
from .render_compose import Step5Result, run_step5

__all__ = [
    "Body",
    "SceneParams",
    "Step5Result",
    "count_pair_collisions",
    "load_trajectory_csv",
    "run_step5",
    "simulate_scene",
]
