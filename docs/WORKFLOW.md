# Phys4D — workflow (Milestone 2)

Concise runbook for the sphere-bounce experiment: sim data → static 3DGS → view result.

## 1. Environment (local)

```bash
conda activate phys4d   # torch + pybullet + imageio; pin numpy<2 if torch breaks
cd /path/to/cs231n_project_phys4d
```

On macOS: `conda install -c conda-forge pybullet` (avoid `pip install pybullet` from source).

## 2. Generate PyBullet dataset

```bash
python scripts/generate_sphere_bounce_dataset.py
```

Writes under `outputs/sphere_bounce_m2/` (gitignored): `rgb/`, `masks/`, `cameras.json`, `object_poses.csv`.

## 3. Package for 3D Gaussian Splatting

```bash
python scripts/export_gs_blender_scene.py
```

Default: **frame 0**, **6 cameras** → `outputs/sphere_bounce_m2/gs_blender_frame00000/` (`transforms_train.json` + `train/cam*.png`).

This is **static** 3DGS (one timestep), not the full 90-frame bounce video.

## 3b. Package for 4D Gaussian Splatting (full bounce video)

Uses [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) (DyNeRF-style layout):

```bash
python 4dgs/scripts/export_4dgs_dataset.py
```

Writes `outputs/sphere_bounce_m2/dynerf_sphere_bounce/`:

| Split | Cameras | Frames | Views |
|-------|---------|--------|-------|
| train | 0–3 | 0–59 | 240 |
| test | 4–5 | 60–89 | 60 |

Each frame has a `time` field (seconds at 60 fps). Config: `4dgs/configs/sphere_bounce_4dgs.yaml`.

## 4. Train on Modal (GPU)

```bash
pip install -r requirements-modal.txt
modal setup
modal run modal_app.py --upload
modal run modal_app.py --train

# 4DGS (full video):
modal run modal_app.py --upload-4d
modal run modal_app.py --train-4d
```

- **Static 3DGS:** upload → `phys4d-gs-data:/scene`; train → [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (7k iters, A10G).
- **4DGS:** upload → `phys4d-gs-data:/4d_scene`; train → fudan 4DGS (15k iters default). First run is slow (CUDA extension build).
- Results on volume `phys4d-gs-output`: `gs_sphere_bounce/` or `4dgs_sphere_bounce/`.

## 5. Download trained model

```bash
cd /path/to/cs231n_project_phys4d
modal volume get phys4d-gs-output gs_sphere_bounce . --force
```

Do **not** create an empty `local_gs_out/` folder first (Modal errors). Optional rename:

```bash
mv gs_sphere_bounce local_gs_out
```

Trained splat file:

`gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply`

(Use this file, not `input.ply` — that is the noisy COLMAP init.)

## 6. View the result (SuperSplat)

1. Open [superspl.at/editor](https://superspl.at/editor).
2. **File → Open** → `point_cloud/iteration_7000/point_cloud.ply`.
3. Press **`Space`** to hide Centers/Rings edit overlays (normal rendered view).
4. Orbit: left-drag; pan: right-drag; zoom: scroll.
5. To **translate** the scene: select layer in Scene Manager → **Move** gizmo (arrows) → **File → Save** if you want to keep edits.

Compare to a source image: `outputs/sphere_bounce_m2/rgb/cam00/frame00000.png`.

Xcode / generic point viewers only show Gaussian **centers** as dots — they will look noisy; SuperSplat is required for proper splat rendering.

## 7. Quick physics sanity checks (local)

```bash
python scripts/recover_restitution.py
python -m unittest tests/test_restitution_recovery.py
python scripts/recover_restitution_from_poses.py   # pose-only; bias vs true e expected
```

## Status (2026-05-15)

| Done | Not yet |
|------|---------|
| PyBullet multi-view data + masks | Joint physics + rendering optimization |
| Static 3DGS train (~PSNR 46.6 @ 7k iters) | 4DGS Modal train + eval/orbit/render scripts |
| Toy restitution recovery | Extrapolation metrics on held-out frames |
| Modal upload/train pipeline | Counterfactual restitution edits in render |

## 8. Pose-linked Gaussians + eval (new)

Requires trained `point_cloud.ply` and `outputs/sphere_bounce_m2/object_poses.csv`:

```bash
pip install plyfile   # if missing

# Warp canonical 3DGS (frame 0) to another timestep
python scripts/warp_gaussians_to_frame.py --target-frame 30 --sphere-only

# Optional: mask-based selection (uses trained gs_sphere_bounce/cameras.json)
python scripts/warp_gaussians_to_frame.py --target-frame 30 --mask-filter

# Physics: fit e on train frames 0–59, MSE on test 60–89
python scripts/eval_physics_trajectory_split.py

# Proxy render check (3DGS projection + PyBullet masks; best after warp)
python scripts/eval_warped_mask_coverage.py --target-frame 30
```

Open warped PLY in SuperSplat and compare to `rgb/cam00/frame00030.png`.

## 9. 4DGS train + download

```bash
python 4dgs/scripts/export_4dgs_dataset.py
modal run modal_app.py --upload-4d
modal run modal_app.py --train-4d
modal volume get phys4d-gs-output 4dgs_sphere_bounce . --force
```

Open `4dgs_sphere_bounce/point_cloud/iteration_*/point_cloud.ply` in SuperSplat; compare to `rgb/cam00/frame00030.png` at the same time.

## 10. 4DGS metrics, calibrated video, orbit video (Modal or local CUDA)

After training, measure reconstruction error vs simulator RGB (train / test splits):

```bash
# Modal (runs on volume data + checkpoint)
modal run modal_app.py --eval-4d
modal volume get phys4d-gs-output 4dgs_eval/metrics_4dgs.json . --force

# Local (after downloading checkpoint + same dynerf folder layout)
export FOURDGS_ROOT=/path/to/4d-gaussian-splatting
python 4dgs/scripts/eval_4dgs_metrics.py \
  --config 4dgs/configs/sphere_bounce_4dgs.yaml \
  --dataset outputs/sphere_bounce_m2/dynerf_sphere_bounce \
  --checkpoint 4dgs_sphere_bounce/chkpnt15000.pth \
  --out-json metrics_4dgs.json
```

Calibrated multi-view + time (matches `transforms_*.json`):

```bash
modal run modal_app.py --render-4d
modal volume get phys4d-gs-output 4dgs_renders/latest . --force   # trajectory.mp4
```

Novel horizontal orbit (smooth camera path; time sweeps by default over `time_duration`):

```bash
modal run modal_app.py --render-4d-orbit
modal volume get phys4d-gs-output 4dgs_renders/orbit_latest . --force   # orbit.mp4
```

Optional Modal flags: `--render-4d-checkpoint chkpnt15000.pth`, `--render-fps 30`, `--orbit-frames 240`, `--orbit-time-start 0`, `--orbit-time-end 0.5`, `--eval-4d-dry-run-max 60` (cap views for a quick eval).

## Next session

1. Log `metrics_4dgs.json` train/test PSNR in `docs/PROGRESS.md`; compare to TensorBoard if available.
2. Differentiable / image loss on held-out views (beyond PSNR script).
3. Joint optimize restitution with rendering + physics losses.

More detail: `docs/modal.md`, `docs/PROGRESS.md`, `configs/sphere_bounce_m2.json`.
