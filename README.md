# cs231n_project_phys4d

**Visually conditioned dynamics on 4D Gaussian scenes** — follow [`project.md`](project.md).

Multi-view video → **4DGS** (perception) → per-object **states** + **t=0 visual features** → learned **dynamics rollout** → warp **Gaussians** → evaluate on future frames.

| Doc | Purpose |
|-----|---------|
| [`project.md`](project.md) | **Authoritative** phases 1–6 |
| [`docs/WORKFLOW_COMMANDS.md`](docs/WORKFLOW_COMMANDS.md) | **All commands** (local + Modal) |
| [`docs/PIPELINE_FROM_SCRATCH.md`](docs/PIPELINE_FROM_SCRATCH.md) | High-level flow |
| [`docs/PROJECT_CHECKLIST.md`](docs/PROJECT_CHECKLIST.md) | Implementation status |

## Quick start

```bash
conda activate phys4d
python -m unittest tests/test_visual_dynamics.py -q
```

Then follow **[`docs/WORKFLOW_COMMANDS.md`](docs/WORKFLOW_COMMANDS.md)** (data → Modal train → E2E → eval).

Modal summary: [`docs/modal_scripts.md`](docs/modal_scripts.md).

## Legacy

Param-ID scripts (`param_ident/`, `train_param_predictor.py`) remain for ablations but are **not** the primary story in `project.md`.
