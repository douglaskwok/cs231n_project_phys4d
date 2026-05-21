# Workflow & commands (visual dynamics)

**Spec:** [`../project.md`](../project.md)  
**Repo:** `cs231n_project_phys4d` — run all commands from the repo root unless noted.

---

## What this pipeline does (one paragraph)

PyBullet generates multi-view bounce videos + poses → we cache **states** and **t=0 visual features** → a **Transformer** predicts future object states → a trained **3D Gaussian** scene is **warped** to those poses → metrics on held-out frames. Training and heavy GPU work are intended for **Modal**; data generation is **local**.

---

## 0. One-time setup

```bash
cd /path/to/cs231n_project_phys4d
conda activate phys4d
pip install -r requirements-m2.txt
pip install -r requirements-modal.txt   # only for Modal
modal setup                             # only for Modal
python -m unittest tests/test_visual_dynamics.py -q
```

**Avoid iCloud stalls:** generate data under `~/phys4d_data/`, not only `~/Desktop/.../outputs/`.

---

## 1. Phase 1 — Generate data (local, PyBullet)

### Single demo scene (E2E / 3DGS)

Uses **10 ring cameras** (`configs/sphere_bounce_m2.json`).

```bash
python scripts/generate_sphere_bounce_dataset.py
# -> outputs/sphere_bounce_m2/{rgb,masks,object_poses.csv,cameras.json}
```

### Training batch (45 scenes, 10 cameras)

```bash
# Recommended: write directly to local disk (edit batch_root in manifest if needed)
# configs/sphere_bounce_batch.json -> K=10, 45-scene grid

python scripts/generate_param_id_batch.py \
  --manifest configs/sphere_bounce_batch.json

python scripts/build_param_id_splits.py \
  --batch-root outputs/sphere_bounce_batch \
  --out outputs/sphere_bounce_batch/dataset_manifest.json
```

**Smoke (1 scene):**

```bash
python scripts/generate_param_id_batch.py \
  --manifest configs/sphere_bounce_batch.json --limit 1
ls outputs/sphere_bounce_batch/scene_*/rgb | wc -l   # expect 10 (cam00..cam09)
```

**Override camera count:**

```bash
python scripts/generate_sphere_bounce_dataset.py --num-cameras 10
python scripts/generate_param_id_batch.py --num-cameras 10
```

### Copy batch off Desktop / iCloud (before upload)

```bash
bash scripts/materialize_batch_local.sh ~/phys4d_data/sphere_bounce_batch
python scripts/verify_batch_local.py --batch-root ~/phys4d_data/sphere_bounce_batch
```

---

## 2. Phase 2 — Perception cache (states + visual features)

Writes per scene: `perception/states.npy`, `perception/visual_feat_t0.npy`.

**Local:**

```bash
python scripts/extract_perception.py --device mps --skip-existing
# or: --device cuda
```

**Modal (GPU ResNet, after batch is on volume):**

```bash
bash scripts/upload_batch_to_modal.sh
# or: modal volume put phys4d-gs-data ~/phys4d_data/sphere_bounce_batch sphere_bounce_batch

modal run modal_app.py --extract-perception
```

---

## 3. Phases 3–4 — Train visual dynamics (local or Modal)

**Local:**

```bash
python scripts/train_visual_dynamics.py \
  --manifest outputs/sphere_bounce_batch/dataset_manifest.json
# checkpoint -> outputs/visual_dynamics/visual_dynamics.pt
```

**Modal (recommended):**

```bash
modal run modal_app.py --train-visual-dynamics
modal volume get phys4d-gs-output visual_dynamics . --force
```

**States-only ablation:**

```bash
python scripts/train_visual_dynamics.py --no-visual \
  --out-dir outputs/visual_dynamics/states_only
# Modal:
modal run modal_app.py --train-visual-dynamics --visual-states-only
```

---

## 4. Static 3DGS — for Phase 5 warp (Modal)

Needed for `run_visual_dynamics_pipeline` (warp PLY).

```bash
python scripts/export_gs_blender_scene.py

modal run modal_app.py --upload
modal run modal_app.py --train
modal volume get phys4d-gs-output gs_sphere_bounce . --force
```

---

## 5. Phase 5 — E2E inference (rollout + warp)

**Local:**

```bash
python scripts/run_visual_dynamics_pipeline.py \
  --checkpoint outputs/visual_dynamics/visual_dynamics.pt \
  --ply gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply \
  --scene-dir outputs/sphere_bounce_m2
# -> outputs/visual_dynamics_pipeline/pipeline_report.json
```

**Modal:**

```bash
modal run modal_app.py --upload-visual-pipeline
modal run modal_app.py --visual-pipeline
modal volume get phys4d-gs-output visual_dynamics_pipeline . --force
```

---

## 6. Phase 6 — Evaluation (local)

```bash
python scripts/eval_visual_dynamics.py \
  --manifest outputs/sphere_bounce_batch/dataset_manifest.json \
  --checkpoint outputs/visual_dynamics/visual_dynamics.pt

python scripts/run_visual_dynamics_ablations.py
```

---

## Optional — 4DGS reconstruction (not required for dynamics training)

Separate track: reconstruct dynamic scene with fudan 4DGS.

```bash
python 4dgs/scripts/export_4dgs_dataset.py
modal run modal_app.py --upload-4d
modal run modal_app.py --train-4d
modal run modal_app.py --render-4d
modal run modal_app.py --eval-4d
modal volume get phys4d-gs-output 4dgs_sphere_bounce . --force
```

---

## Recommended Modal path (copy-paste)

Assumes batch is verified at `~/phys4d_data/sphere_bounce_batch`.

```bash
cd /path/to/cs231n_project_phys4d
conda activate phys4d

# 1) Upload data (~81k files, ~280 MB with 10 cams)
modal volume rm phys4d-gs-data sphere_bounce_batch -r 2>/dev/null || true
modal volume put phys4d-gs-data ~/phys4d_data/sphere_bounce_batch sphere_bounce_batch

# 2) Perception on volume
modal run modal_app.py --extract-perception

# 3) Train dynamics
modal run modal_app.py --train-visual-dynamics
modal volume get phys4d-gs-output visual_dynamics . --force

# 4) 3DGS + E2E
python scripts/export_gs_blender_scene.py
modal run modal_app.py --upload
modal run modal_app.py --train
modal run modal_app.py --upload-visual-pipeline
modal run modal_app.py --visual-pipeline
```

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| Upload stuck ~1k–2k files | iCloud placeholders — use `materialize_batch_local.sh` + `verify_batch_local.py` |
| `modal run --upload-batch` silent forever | First deploy builds huge Docker images — use `bash scripts/upload_batch_to_modal.sh` instead |
| `extract_perception` CSV error | Empty `object_poses.csv` — download from iCloud or regenerate batch |
| `visual-pipeline` missing PLY | Run Modal `--upload` + `--train` (3DGS) first |
| Only 6 cameras in old batch | Regenerate with `configs/sphere_bounce_batch.json` (10 cameras) |

---

## What is *not* wired yet (vs full `project.md`)

- 4DGS does **not** feed dynamics training (poses/features from PyBullet).
- Training Stage B is still **1-step loss** (horizon only in eval).
- No full **render PSNR/SSIM** on warped frames (mask-coverage proxy only).
- Batch is **45 scenes × 10 cams**, not 150 scenes.

Legacy **param-ID** scripts exist locally but are **not** on Modal.
