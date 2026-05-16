# Project direction — inverse physics from video (M2)

**Checklist:** [`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md)

## One paragraph

We simulate calibrated multi-view bounces in PyBullet with known physics parameters and trajectories. A **CNN** (ResNet18 + temporal Transformer + multi-view attention) predicts physical parameters from video. Predicted parameters drive a **PyBullet rollout**; rigid transforms warp **3D/4D Gaussian** object clusters and re-render for evaluation. The contribution is **visually grounded system identification** on Gaussian scene representations — not vanilla 4DGS reconstruction alone. v1 predicts **restitution, mass, and drop height** (no friction). Cross-scene training on 100–150 randomized scenarios with an 80/10/10 instance split is the default regime.

## Pipeline

```
PyBullet (random e, m, z) → multi-view video + GT params + poses
        → CNN param predictor (phases 2–3)
        → PyBullet rollout from predicted params (phase 4)
        → warp object Gaussians (4DGS/3DGS substrate) → render (phase 4–5)
```

## What we removed (May 2026 pivot)

The prior **learned visual dynamics / temporal extrapolation** branch (`visual_dynamics`, extrapolation baselines, batch 4DGS smoke) is archived in git history. It did not match the M2 deck spine (param ID → sim → render).

## Quick commands

```bash
# Phase 1
python scripts/generate_param_id_batch.py --limit 10
python scripts/build_param_id_splits.py

# Phase 2–3
python scripts/train_param_predictor.py --epochs 50

# 4DGS substrate (one demo scene)
python scripts/export_4dgs_dataset.py
modal run modal_app.py --upload-4d && modal run modal_app.py --train-4d
```
