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

## 4. Train on Modal (GPU)

```bash
pip install -r requirements-modal.txt
modal setup
modal run modal_app.py --upload
modal run modal_app.py --train
```

- Upload copies your Blender scene to volume `phys4d-gs-data:/scene`.
- Train runs official [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (7000 iters, A10G). First run is slow (CUDA extension build).
- Result on volume `phys4d-gs-output` at `gs_sphere_bounce/`.

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
| Static 3DGS train (~PSNR 46.6 @ 7k iters) | 4DGS / per-frame or dynamic splats |
| Toy restitution recovery | Extrapolation metrics on held-out frames |
| Modal upload/train pipeline | Counterfactual restitution edits in render |

## Next session

1. Spot-check SuperSplat vs `cam00/frame00000.png` (axis flip? → fix `colmap_export.py` if needed).
2. Connect object poses from `object_poses.csv` to rigid Gaussian transforms.
3. Pixel / mask loss on test frames + recover restitution with differentiable sim.
4. Update `docs/PROGRESS.md` when you have numbers.

More detail: `docs/modal.md`, `docs/PROGRESS.md`, `configs/sphere_bounce_m2.json`.
