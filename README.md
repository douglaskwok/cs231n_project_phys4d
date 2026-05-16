# cs231n_project_phys4d

**Visually conditioned dynamics on 4D Gaussian scenes** — follow [`project.md`](project.md).

Multi-view video → **4DGS** (perception) → per-object **states** + **t=0 visual features** → learned **dynamics rollout** → warp **Gaussians** → evaluate on future frames.

| Doc | Purpose |
|-----|---------|
| [`project.md`](project.md) | **Authoritative** phases 1–6 |
| [`docs/PIPELINE_FROM_SCRATCH.md`](docs/PIPELINE_FROM_SCRATCH.md) | Commands |
| [`docs/PROJECT_CHECKLIST.md`](docs/PROJECT_CHECKLIST.md) | Status |

## Quick start

```bash
conda activate phys4d
python -m unittest tests/test_visual_dynamics.py -q

# Phase 1–2 (data + perception cache)
python scripts/build_param_id_splits.py \
  --batch-root outputs/sphere_bounce_batch \
  --out outputs/sphere_bounce_batch/dataset_manifest.json
python scripts/extract_perception.py --limit 5

# Phase 3–4 (train dynamics)
python scripts/train_visual_dynamics.py \
  --manifest outputs/sphere_bounce_batch/dataset_manifest.json

# Phase 5 (E2E on one scene; needs checkpoint + gs_sphere_bounce PLY)
python scripts/run_visual_dynamics_pipeline.py

# Phase 6
python scripts/eval_visual_dynamics.py
python scripts/run_visual_dynamics_ablations.py --skip-train  # after training both models
```

## Modal (4DGS perception)

```bash
python scripts/export_4dgs_dataset.py
modal run modal_app.py --upload-4d && modal run modal_app.py --train-4d
```

See [`docs/modal_scripts.md`](docs/modal_scripts.md).

## Legacy

Param-ID scripts (`param_ident/`, `train_param_predictor.py`) remain for ablations but are **not** the primary story in `project.md`.
