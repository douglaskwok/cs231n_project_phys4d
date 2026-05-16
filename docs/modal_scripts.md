# Modal vs local (project.md / visual dynamics)

| Step | Local | Modal |
|------|-------|-------|
| PyBullet data | `generate_sphere_bounce_dataset.py` | — |
| Upload batch | — | `--upload-batch` |
| Perception cache | `extract_perception.py` | `--extract-perception` |
| **Train dynamics** | `train_visual_dynamics.py` | **`--train-visual-dynamics`** |
| 3DGS | `export_gs_blender_scene.py` | `--upload` + `--train` |
| 4DGS | `export_4dgs_dataset.py` | `--upload-4d` + `--train-4d` |
| E2E pipeline | `run_visual_dynamics_pipeline.py` | `--upload-visual-pipeline` + `--visual-pipeline` |

## Visual dynamics on Modal (recommended)

```bash
# 1) Upload 45-scene batch (+ manifest) to volume
modal run modal_app.py --upload-batch

# 2) Cache perception on GPU machine (states + t=0 visual features)
modal run modal_app.py --extract-perception

# 3) Train (A10G, ~hours depending on epochs in visual_dynamics.json)
modal run modal_app.py --train-visual-dynamics

# 4) Download checkpoint
modal volume get phys4d-gs-output visual_dynamics . --force

# 5) E2E demo scene (upload m2 + 3DGS + ckpt first)
modal run modal_app.py --upload-visual-pipeline
modal run modal_app.py --visual-pipeline
```

States-only ablation on Modal:

```bash
modal run modal_app.py --train-visual-dynamics --visual-states-only
```

## Legacy param-ID (not project.md primary)

`--train-param-id`, `--pipeline`, `--upload-pipeline`
