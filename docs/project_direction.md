# Project direction — visually conditioned dynamics on 4DGS (project.md)

**Authoritative spec:** [`../project.md`](../project.md)  
**Checklist:** [`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md)  
**Commands:** [`PIPELINE_FROM_SCRATCH.md`](PIPELINE_FROM_SCRATCH.md)

## One paragraph

We reconstruct dynamic scenes with **4D Gaussian Splatting** on the training video, extract per-object **states** (pose + finite-difference velocity) and **fixed t=0 visual features** (ResNet18 on masked crops). A **visually conditioned dynamics model** (object tokens + Transformer) autoregressively predicts future **SE(3) states** from past state history and appearance — not a neural ODE on trajectories alone, and not CNN→physics-params→simulator. Predicted rigid transforms **warp per-object Gaussian clusters** and re-render for evaluation on held-out future frames.

## Pipeline

```
PyBullet data (train 0–59, test 60–89)
  → extract_perception.py (states + visual_feat@t=0)
  → train_visual_dynamics.py (Stage A: 1-step; Stage B: rollout curriculum)
  → run_visual_dynamics_pipeline.py (rollout + warp 3DGS + metrics)
  → eval_visual_dynamics.py + run_visual_dynamics_ablations.py
```

4DGS training remains on Modal (`--upload-4d`, `--train-4d`, `--render-4d`) as the perception substrate.

## CV justification

Appearance at **t=0** conditions future motion: the model must use pixels (not only past poses) to distinguish regimes that look different (mass/restitution proxies). **Ablation:** `--no-visual` states-only baseline should lose on rollout/render metrics.
