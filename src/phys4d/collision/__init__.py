"""Collision pipeline: multi-body extension of the bounce pipeline.

The package is intentionally lazy: metric/refit scripts should be able to use
the lightweight physics helpers without importing Wu 4DGS/Torch render code.
"""

from __future__ import annotations

_EXPORTS = {
    "Body": ("physics", "Body"),
    "SceneParams": ("physics", "SceneParams"),
    "Step5Result": ("render_compose", "Step5Result"),
    "count_pair_collisions": ("physics", "count_pair_collisions"),
    "load_trajectory_csv": ("physics", "load_trajectory_csv"),
    "run_step5": ("render_compose", "run_step5"),
    "simulate_scene": ("physics", "simulate_scene"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _EXPORTS[name]
    module = __import__(f"{__name__}.{module_name}", fromlist=[attr])
    value = getattr(module, attr)
    globals()[name] = value
    return value
