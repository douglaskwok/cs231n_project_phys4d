# Phys4D — Ball Bounce Pipeline

**Authoritative spec:** [`milestone3.md`](milestone3.md)

Multi-view PyBullet ball-drop video → SAM2 masks → Wu 4DGS (train frames only) → **physics trajectory fit & extrapolation** → rigid Gaussian warp + render → metrics.

## Status

| Step | Owner | Status |
|------|-------|--------|
| 1. PyBullet data | teammate | external (not in this tree yet) |
| 2. SAM2 segmentation | teammate | external |
| 3. Wu 4DGS training | teammate | external |
| **4a–4c** Trajectory extract / fit / predict | us | **next** |
| **5** Rebuild scene (bg + warp + render) | us | planned |
| **6** Evaluation | us | planned |

## Repo layout

```
third_party/4DGaussians/     # git submodule (hustvl/4DGaussians)
src/phys4d/bounce/           # load_4dgs, extract, physics, render_compose, metrics
scripts/bounce/              # step4a_extract … step6_eval CLIs
outputs/bounce_pipeline/     # per-run artifacts (gitignored)
milestone3.md                  # full pipeline spec + acceptance criteria
```

## Setup

```bash
pip install -r requirements.txt
git submodule add https://github.com/hustvl/4DGaussians third_party/4DGaussians
```

End-to-end commands are in **milestone3.md** § End-to-end CLI.

### Step 4a (implemented)

```bash
export WU_4DGS_ROOT=third_party/4DGaussians   # hustvl clone (already vendored)
pip install -r requirements.txt

python scripts/bounce/step4a_extract.py \
  --canonical .../point_cloud/iteration_30000/point_cloud.ply \
  --deform    .../point_cloud/iteration_30000/deformation.pth \
  --cfg-args  .../cfg_args \
  --dynerf-export 4dgs/experiments/object_only/runs/<RUN_NAME> \
  --gt-poses dataset/.../object_poses.csv \
  --out outputs/bounce_pipeline/<RUN_NAME>/step4a
```

### Step 4b (implemented)

```bash
python scripts/bounce/step4b_fit.py \
  --traj outputs/bounce_pipeline/<RUN_NAME>/step4a/trajectory_smoothed.csv \
  --out  outputs/bounce_pipeline/<RUN_NAME>/step4b
```

### Step 4c (implemented)

```bash
python scripts/bounce/step4c_predict.py \
  --params outputs/bounce_pipeline/<RUN_NAME>/step4b/physics_params.json \
  --frame-map 4dgs/experiments/object_only/runs/<RUN_NAME>/frame_map.json \
  --scene-config dataset/.../scene_<id>/config.json \
  --traj outputs/bounce_pipeline/<RUN_NAME>/step4a/trajectory_smoothed.csv \
  --out outputs/bounce_pipeline/<RUN_NAME>/step4c
```

### Step 5 (implemented)

```bash
python scripts/bounce/step5_render.py \
  --canonical .../point_cloud/iteration_30000/point_cloud.ply \
  --bg-ply dataset/.../background_3dgs/background.ply \
  --predicted outputs/bounce_pipeline/<RUN>/step4c/trajectory_predicted.csv \
  --ref-traj outputs/bounce_pipeline/<RUN>/step4a/trajectory_smoothed.csv \
  --dynerf-export 4dgs/experiments/object_only/runs/<RUN> \
  --cfg-args .../cfg_args \
  --scene-dir dataset/.../scene_<id> \
  --out outputs/bounce_pipeline/<RUN>/step5
```

Uses Wu 3DGS rasterizer (`stage=coarse`, no deformation) when CUDA + `diff_gaussian_rasterization` are available; otherwise **`--force-overlay`** or automatic fallback draws the predicted ball on GT RGB.

### Step 6 (implemented)

```bash
python scripts/bounce/step6_eval.py \
  --predicted outputs/bounce_pipeline/<RUN>/step4c/trajectory_predicted.csv \
  --gt-poses dataset/.../object_poses.csv \
  --rendered outputs/bounce_pipeline/<RUN>/step5/renders \
  --gt-rgb-root dataset/.../rgb \
  --out outputs/bounce_pipeline/<RUN>/step6
```

**Command reference (Steps 4a–6):** see [`STAGES_4_6_COMMANDS.txt`](STAGES_4_6_COMMANDS.txt).

Unit tests (no GPU):

```bash
PYTHONPATH=src python -m unittest \
  tests.test_step4a_timestamps tests.test_step4b_physics tests.test_step4c_extrapolate \
  tests.test_step5_gaussian_ply tests.test_step5_overlay tests.test_step6_metrics -q
```
