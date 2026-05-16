# What runs on Modal vs locally

| Script / job | Where | Modal flag |
|--------------|-------|------------|
| `generate_sphere_bounce_dataset.py` | **Local CPU** (PyBullet) | — |
| `generate_param_id_batch.py` | **Local CPU** | — |
| `build_param_id_splits.py` | **Local** | — |
| `export_gs_blender_scene.py` | **Local** (prep) | `--upload` |
| `export_4dgs_dataset.py` | **Local** (prep) | `--upload-4d` |
| `train_param_predictor.py` | **Local GPU** or Modal | `--train-param-id` |
| `run_param_id_pipeline.py` | **Local GPU** or Modal | `--pipeline` |
| `train` (3DGS) | Modal GPU | `--train` |
| `train_4dgs` | Modal GPU | `--train-4d` |
| `render_4dgs_trajectory.py` | Modal GPU | `--render-4d` |
| `eval_4dgs_metrics.py` | Modal GPU | `--eval-4d` |
| `warp_gaussians_to_frame.py` | Local CPU (in pipeline on Modal too) | via `--pipeline` |
| `recover_restitution*.py` | Local | — |
| unittests | Modal T4 | `--tests` |

## Phase 4 on Modal (typical)

```bash
# 1) Local data + optional local train
python scripts/generate_sphere_bounce_dataset.py
modal run modal_app.py --upload && modal run modal_app.py --train   # 3DGS

# 2) Upload scene + artifacts for pipeline
modal run modal_app.py --upload-pipeline

# 3) Train CNN on volume (after batch dataset upload)
modal volume put phys4d-gs-data outputs/param_id_dataset param_id_dataset
modal run modal_app.py --train-param-id

# 4) E2E pipeline
modal run modal_app.py --pipeline
modal volume get phys4d-gs-output param_id_pipeline .
```
