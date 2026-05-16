# cs231n_project_phys4d

**Inverse physics from multi-view video:** PyBullet data → CNN predicts physical parameters → sim rollout → warp 3D/4D Gaussians → render.

**M2 checklist (deck spine):** [`docs/PROJECT_CHECKLIST.md`](docs/PROJECT_CHECKLIST.md)  
**Direction:** [`docs/project_direction.md`](docs/project_direction.md)  
**Original claim:** [`docs/project_claim.md`](docs/project_claim.md)

## Quick start

```bash
conda activate phys4d   # torch + pybullet + imageio
python -m unittest tests/test_param_ident_shapes.py

# Phase 1 — data (smoke: 5 scenes)
python scripts/generate_param_id_batch.py --limit 5
python scripts/build_param_id_splits.py

# Phase 2–3 — train CNN
python scripts/train_param_predictor.py --epochs 50
```

## Core scripts

| Script | Purpose |
|--------|---------|
| `generate_sphere_bounce_dataset.py` | Single-scene PyBullet RGB + masks + poses + `physics_params.json` |
| `generate_param_id_batch.py` | 150 randomized scenes (configurable) |
| `build_param_id_splits.py` | 80/10/10 `dataset_manifest.json` |
| `train_param_predictor.py` | ResNet18 + Transformer param regression |
| `export_4dgs_dataset.py` | DyNeRF export for 4DGS (phase 4 substrate) |
| `warp_gaussians_to_frame.py` | Rigid warp static 3DGS with object poses |
| `modal_app.py` | GPU 3DGS / 4DGS train + render |

## Modal

```bash
pip install -r requirements-modal.txt && modal setup
python scripts/export_4dgs_dataset.py
modal run modal_app.py --upload-4d
modal run modal_app.py --train-4d
```

See [`docs/modal.md`](docs/modal.md) and [`docs/modal_scripts.md`](docs/modal_scripts.md) (what runs locally vs Modal).

## Phase 4 (E2E)

```bash
python scripts/run_param_id_pipeline.py
# or on Modal after --upload-pipeline:
modal run modal_app.py --pipeline
```
