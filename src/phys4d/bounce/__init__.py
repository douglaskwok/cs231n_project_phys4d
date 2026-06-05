"""Steps 4-6: trajectory extract, physics fit, extrapolate, render, eval.

The package is intentionally lazy: lightweight scripts such as metric refits
should not import Wu 4DGS/Torch dependencies unless they ask for Step 4a.
"""

from __future__ import annotations

_EXPORTS = {
    "LoadedWu4DGS": ("load_4dgs", "LoadedWu4DGS"),
    "PhysicsParams": ("physics", "PhysicsParams"),
    "Step4aResult": ("extract", "Step4aResult"),
    "Step4bResult": ("physics", "Step4bResult"),
    "Step4cResult": ("extrapolate", "Step4cResult"),
    "Step5Result": ("render_compose", "Step5Result"),
    "Step6Result": ("metrics", "Step6Result"),
    "fit_physics_params": ("physics", "fit_physics_params"),
    "load_physics_params_json": ("physics", "load_physics_params_json"),
    "load_trajectory_csv": ("physics", "load_trajectory_csv"),
    "load_wu_4dgs": ("load_4dgs", "load_wu_4dgs"),
    "run_step4a": ("extract", "run_step4a"),
    "run_step4b": ("physics", "run_step4b"),
    "run_step4c": ("extrapolate", "run_step4c"),
    "run_step5": ("render_compose", "run_step5"),
    "run_step6": ("metrics", "run_step6"),
    "simulate_trajectory": ("physics", "simulate_trajectory"),
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
