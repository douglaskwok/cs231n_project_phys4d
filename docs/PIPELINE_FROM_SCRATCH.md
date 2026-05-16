# Pipeline from scratch (project.md)

**Read [`../project.md`](../project.md) first.** This file is command-oriented.

## 0. Environment

```bash
conda activate phys4d
pip install -r requirements-m2.txt  # pybullet, imageio, torch, torchvision
```

## 1. Data (Phase 1)

```bash
# Single scene
python scripts/generate_sphere_bounce_dataset.py

# Batch (or use existing outputs/sphere_bounce_batch/)
python scripts/generate_param_id_batch.py --limit 10
python scripts/build_param_id_splits.py \
  --batch-root outputs/sphere_bounce_batch \
  --out outputs/sphere_bounce_batch/dataset_manifest.json
```

## 2. Perception (Phase 2)

```bash
# Cache states.npy + visual_feat_t0.npy per scene
python scripts/extract_perception.py

# 4DGS (Modal)
python scripts/export_4dgs_dataset.py
modal run modal_app.py --upload-4d && modal run modal_app.py --train-4d
modal volume get phys4d-gs-output gs_sphere_bounce . --force
```

## 3–4. Train dynamics (Phases 3–4)

**Local (MPS/CUDA):**

```bash
python scripts/extract_perception.py
python scripts/train_visual_dynamics.py \
  --manifest outputs/sphere_bounce_batch/dataset_manifest.json
```

**Modal (GPU):**

```bash
modal run modal_app.py --upload-batch
modal run modal_app.py --extract-perception
modal run modal_app.py --train-visual-dynamics
modal volume get phys4d-gs-output visual_dynamics . --force
```

States-only ablation: add `--no-visual` locally or `--visual-states-only` on Modal.

## 5. Inference E2E (Phase 5)

```bash
python scripts/run_visual_dynamics_pipeline.py \
  --checkpoint outputs/visual_dynamics/visual_dynamics.pt \
  --ply gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply
```

## 6. Evaluation (Phase 6)

```bash
python scripts/eval_visual_dynamics.py
python scripts/run_visual_dynamics_ablations.py
```

Overnight (train both visual + states-only):

```bash
OVERNIGHT_EPOCHS=40 bash scripts/overnight_run.sh  # legacy; prefer commands above
```
