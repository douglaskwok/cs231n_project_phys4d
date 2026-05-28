# Bounce pipeline (Phase 1+)

Uses **[fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)** checkpoints (`chkpnt*.pth`) from Modal training — same stack as `4dgs/scripts/train_one_final_scene_4dgs.sh` and `--render-4d`.

## Phase 1 on Modal (recommended — Mac needs Fudan CUDA extensions)

Prerequisites (same as train/render):

1. DyNeRF export on `phys4d-gs-data:/4d_scene` (the export you trained on):

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py \
  4dgs/experiments/object_only/runs/ball_drop_e0p78_a0p0_object_all12_allframes \
  --modal-cmd "arch -arm64 modal"
```

2. Trained model on `phys4d-gs-output:/4dgs_ball_drop_e0p78_a0p0_object_all12_allframes` (already done if you trained/rendered).

Run Phase 1:

```bash
cd cs231n_project_phys4d.nosync

arch -arm64 modal run modal_app.py --bounce-phase1 \
  --bounce-phase1-config room_physics_4dgs_4p0s.yaml \
  --bounce-phase1-model 4dgs_ball_drop_e0p78_a0p0_object_all12_allframes \
  --bounce-phase1-checkpoint chkpnt15000.pth \
  --bounce-phase1-gt-poses dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_0004_e0p78_a0p0/object_poses.csv \
  --bounce-phase1-out-rel bounce_pipeline/ball_drop_e0p78_a0p0_object_all12_allframes/phase1
```

Download results:

```bash
mkdir -p outputs/bounce_pipeline/ball_drop_e0p78_a0p0_object_all12_allframes/phase1
arch -arm64 modal volume get phys4d-gs-output \
  bounce_pipeline/ball_drop_e0p78_a0p0_object_all12_allframes/phase1 \
  outputs/bounce_pipeline/ball_drop_e0p78_a0p0_object_all12_allframes/phase1 \
  --force
```

Outputs: `trajectory_raw.csv`, `trajectory_smoothed.csv`, `trajectory_plot.png`, `phase1_meta.json`.

## Local Phase 1 (optional — needs Fudan clone + CUDA ops)

Local Mac CPU usually fails on `pointops2_cuda`. Prefer Modal above.

```bash
git clone --recursive https://github.com/fudan-zvg/4d-gaussian-splatting.git
export FOURDGS_ROOT="/path/to/4d-gaussian-splatting"
python -m pip install omegaconf scipy matplotlib  # into the same env as `python`
```

Spec: [`bouncepipeline.md`](../../bouncepipeline.md).
