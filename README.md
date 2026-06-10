# Phys4D — Physics-Aware 4D Scene Reconstruction

**CS231N final project:** multi-view synthetic video → 4D Gaussian Splatting (4DGS) → analytical physics fit → forward prediction → composited novel-view renders.

Two evaluation scenarios:

- **Ball bounce** — single sphere, restitution × launch-angle grid  
- **Object collision** — two rigid boxes, mass × restitution × velocity grid  

Full write-up: [`CS231n_Final_Report-9.pdf`](CS231n_Final_Report-9.pdf) (if present locally).

---

## Pipeline rollout (Steps 1–6)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA          PyBullet 12-view RGB, masks, poses, cameras                │
│    (local CPU)   dataset/generate_*.py, dataset/export_*_12view.py          │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. SEGMENTATION  Object-only masks (PyBullet oracle for paper experiments)  │
│                  Optional SAM2 path: segmentation/                          │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. 4DGS / 3DGS   Object 4DGS (Wu) on train frames + static background 3DGS  │
│    (Modal GPU)   4dgs/scripts/train_one_scene_wu4dgs.sh, modal_app.py       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4a. EXTRACT      Centroid trajectory from deformed Gaussians (train window) │
│    (Modal GPU)   scripts/{bounce,collision}/step4a_extract.py               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4b. FIT          Umeyama align → least-squares physics params               │
│    (local CPU)   scripts/{bounce,collision}/refit_metric_gt.py            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4c. PREDICT      Euler forward integration on held-out frames              │
│    (local CPU)   scripts/{bounce,collision}/predict_metric_from_refit.py  │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. COMPOSE       Rigid translate object Gaussians + alpha-blend background  │
│    (Modal GPU)   scripts/{bounce,collision}/step5_render.py                 │
│                  → src/phys4d/{bounce,collision}/render_compose.py          │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. EVAL          Trajectory RMSE / velocity R² + PSNR / SSIM / MAE          │
│    (local CPU)   scripts/{bounce,collision}/step6_eval.py                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Train / test split:** defined per scene in `config.json` (`train_frames`, `test_frames`). Steps 4b–4c fit on the train window and predict the held-out future; Step 6 scores against PyBullet ground truth.

---

## Supplementary materials (Gaussian inputs after segmentation)

Trained **object 4DGS checkpoints**, **background 3DGS PLYs**, and **DyNeRF exports** used in the paper are too large for git. They live on Google Drive:

**[code-supplementary (Google Drive)](https://drive.google.com/drive/folders/1ifHAAlGXUgIfkS3nYqiCJQ84KqBUk_I6?usp=sharing)**

Use this folder as **proof and staging** for Gaussian assets **after segmentation** (Steps 2–3):

| Expected contents | Used in |
|-------------------|---------|
| Per-scene `rgb/`, `masks/` (or object-only masked exports) | Step 3 training input |
| Wu 4DGS outputs: `point_cloud.ply`, `deformation.pth`, `cfg_args` | Steps 4a, 5 |
| DyNeRF export: `frame_map.json`, `transforms_train.json`, `transforms_test.json` | Steps 4a, 4c, 5 |
| Static background `background.ply` (3DGS) | Step 5 compositing |

Download the scene(s) you need, then point `SCENE_DIR`, `WU_OUT`, and `EXPORT` at the local paths (see [`STAGES_4_6_COMMANDS.txt`](STAGES_4_6_COMMANDS.txt)).

**In-repo generated data** (optional refresh): run the `dataset/` scripts below; outputs go to `dataset/outputs/` (gitignored).

---

## Quick start

### 1. Clone and setup

```bash
git clone <repo-url> cs231n_project_phys4d
cd cs231n_project_phys4d
git submodule update --init third_party/4DGaussians

python3 -m venv .venv_pipeline
source .venv_pipeline/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
export WU_4DGS_ROOT=third_party/4DGaussians
```

For PyBullet data generation, use a separate env with `pybullet`, `imageio`, and optionally `imageio-ffmpeg` (see [`dataset/README.md`](dataset/README.md)).

For GPU steps (3, 4a, 5): install [Modal](https://modal.com), configure credentials, and use `modal_app.py` (see [`context.md`](context.md)).

### 2. Generate or download data

**From Drive:** download the supplementary folder linked above.

**From scratch (local CPU):**

```bash
# Full bundle: 9 bounce + 1 collision + stacking + deformable
python dataset/generate_phys4d_final.py --video-fps 60

# Report collision grid (8 scenes, 2×2×2 parameters)
python dataset/generate_collision_variants_final.py

# Single bounce scene (custom flags — see dataset/README.md)
python dataset/export_ping_pong_12view.py \
  --variation-set single \
  --output-dir dataset/outputs/my_bounce_scene \
  --video-fps 60 --duration-sec 2.0
```

### 3. Train 4DGS (Step 3)

```bash
# Export object-only DyNeRF folder from a scene
python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config dataset/outputs/phys4d_final/.../config.json \
  --masks-root dataset/outputs/phys4d_final/.../masks \
  --output 4dgs/experiments/object_only/runs/<RUN_NAME> \
  --mode object --background black

# Train on Modal (see 4dgs/scripts/train_one_scene_wu4dgs.sh)
bash 4dgs/scripts/train_one_scene_wu4dgs.sh <SCENE_DIR> <RUN_NAME>
```

Collision: train **one 4DGS model per object** with `masks_object_a` / `masks_object_b` (see [`docs/COLLISION_4DGS_COMPOSITION.md`](docs/COLLISION_4DGS_COMPOSITION.md)).

### 4. Physics pipeline (Steps 4a–6)

Set paths once (edit for your run):

```bash
SCENE_DIR="dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_0004_e0p93_a0p0"
RUN_NAME="ball_drop_e0p93_a0p0_object"
WU_OUT="${SCENE_DIR}/4dgs_wu/${RUN_NAME}"          # or path from Google Drive
EXPORT="4dgs/experiments/object_only/runs/${RUN_NAME}"
OUT="outputs/bounce_pipeline/${RUN_NAME}"
```

**Bounce (canonical path):**

```bash
# 4a — GPU (Modal recommended)
modal run modal_app.py --upload-step4a --step4a-dir <staging_dir>
modal run modal_app.py --step4a --bounce-out-rel <RUN>/step4a

# 4b — metric-frame physics fit
python scripts/bounce/refit_metric_gt.py \
  --traj "${OUT}/step4a/trajectory_smoothed.csv" \
  --gt-poses "${SCENE_DIR}/object_poses.csv" \
  --out "${OUT}/step4b_metric"

# 4c — forward predict held-out window
python scripts/bounce/predict_metric_from_refit.py \
  --refit "${OUT}/step4b_metric/refit_metric.json" \
  --out "${OUT}/step4c/trajectory_predicted.csv" \
  --gt-poses "${SCENE_DIR}/object_poses.csv"

# 5 — composite render (GPU; use --composite-mode alpha)
python scripts/bounce/step5_render.py \
  --canonical "${WU_OUT}/point_cloud/iteration_30000/point_cloud.ply" \
  --bg-ply "${SCENE_DIR}/background_3dgs/background.ply" \
  --predicted "${OUT}/step4c/trajectory_predicted.csv" \
  --ref-traj "${OUT}/step4a/trajectory_smoothed.csv" \
  --dynerf-export "${EXPORT}" \
  --cfg-args "${WU_OUT}/cfg_args" \
  --scene-dir "${SCENE_DIR}" \
  --out "${OUT}/step5"

# 6 — metrics
python scripts/bounce/step6_eval.py \
  --predicted "${OUT}/step4c/trajectory_predicted.csv" \
  --gt-poses "${SCENE_DIR}/object_poses.csv" \
  --rendered "${OUT}/step5/renders" \
  --gt-rgb-root "${SCENE_DIR}/rgb" \
  --out "${OUT}/step6"
```

**Collision:** same structure under `scripts/collision/`; see [`scripts/collision/workflow.md`](scripts/collision/workflow.md).

Step-by-step bounce notes: [`scripts/bounce/workflow.md`](scripts/bounce/workflow.md).  
Full flag reference: [`STAGES_4_6_COMMANDS.txt`](STAGES_4_6_COMMANDS.txt).  
Artifact schemas: [`DATA_CONTRACT.md`](DATA_CONTRACT.md).

### 5. Tests (no GPU)

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

---

## Repo layout

```
dataset/                  Step 1 — PyBullet exporters & generators
segmentation/             Step 2 — mask-guided binding utilities
4dgs/                     Step 3 — DyNeRF export, Wu training scripts, experiment logs
third_party/4DGaussians/  Wu et al. 4DGS submodule (required for 3–5)
src/phys4d/
  bounce/                 extract, physics, extrapolate, render_compose, metrics
  collision/              multi-object physics + compositing
scripts/
  bounce/                 Step 4–6 CLIs + workflow helpers
  collision/              collision analogue
  diagnostics/            analysis / plotting (post-hoc)
modal_app.py              Modal GPU orchestration (train, 4a, 5)
outputs/                  Per-run pipeline artifacts (gitignored)
docs/                     Design notes (e.g. collision compositing)
tests/                    Unit tests for Steps 4a–6
```

---

## What is / is not in git

| In git | Not in git (by design) |
|--------|-------------------------|
| All pipeline source code | `outputs/` — step 4–6 run trees |
| `dataset/*.py` generators | `dataset/outputs/` — rendered scenes |
| Tests, specs, commands | `*.ply`, `*.pth`, `*.mp4`, `*.png` checkpoints & media |
| Submodule pointer | Large trained Gaussians → **[Google Drive supplementary](https://drive.google.com/drive/folders/1ifHAAlGXUgIfkS3nYqiCJQ84KqBUk_I6?usp=sharing)** |

---

## Scenario summary (paper)

| Scenario | Scenes | Key parameters | Step scripts |
|----------|--------|----------------|--------------|
| Ball bounce | 12 (4×3 grid) | restitution, launch angle | `scripts/bounce/` |
| Object collision | 8 (2×2×2 grid) | mass A, restitution, velocity mode | `scripts/collision/` |

Data generation defaults for collision: `dataset/generate_collision_variants_final.py`  
(`--geometry-scale 1.75`, `--masses 0.22,0.44`, `--restitutions 0.98,0.90`, etc.)

---

## Further reading

| Document | Purpose |
|----------|---------|
| [`docs/GENAI_CODING_AGENT_USAGE.md`](docs/GENAI_CODING_AGENT_USAGE.md) | Coding-agent usage statement (CS231N submission) |
| [`STAGES_4_6_COMMANDS.txt`](STAGES_4_6_COMMANDS.txt) | Copy-paste CLI for Steps 4a–6 |
| [`DATA_CONTRACT.md`](DATA_CONTRACT.md) | Required file names & JSON schemas |
| [`METRICS.md`](METRICS.md) | RMSE, R², PSNR, SSIM definitions |
| [`context.md`](context.md) | Worked bounce reproduction (Modal paths) |
| [`dataset/README.md`](dataset/README.md) | Data generation flags & layouts |
| [`docs/COLLISION_4DGS_COMPOSITION.md`](docs/COLLISION_4DGS_COMPOSITION.md) | Per-object 4DGS + compositing |

---

## Contributors

- **Simon Casper** — Steps 4–6 (extract, fit, predict, compose, eval), Modal integration  
- **Sze Heng Douglas Kwok** — data generation, segmentation, 4DGS training  
- **Janhavi Purkar** — collision data generation, segmentation, literature review  
