# cs231n_project_phys4d

Focused scaffold for the CS231N Milestone 2 direction: object-centric Gaussian scenes for inverse system identification, extrapolation, and counterfactual editing.

## Milestone 2 Artifacts

- `docs/project_claim.md`: concise novelty framing against 4DGS, PhysGaussian, and GASP.
- `docs/method_equations.md`: slide-ready equations for object Gaussians, rigid state, pose transforms, bounce dynamics, rendering loss, physics losses, and staged optimization.
- `docs/milestone2_slides.md`: 7-slide outline for the milestone presentation.
- `configs/sphere_bounce_m2.json`: locked first experiment spec for a single PyBullet sphere-ground bounce with known masks and poses.
- `scripts/generate_sphere_bounce_dataset.py`: PyBullet multi-view RGB, segmentation masks, camera JSON, and sphere pose CSV (see `docs/PROGRESS.md`).
- `scripts/export_gs_blender_scene.py`: packages **your** `rgb/cam*/frame*.png` + `cameras.json` into a 3DGS Blender folder (`transforms_train.json` + `train/*.png`).
- `scripts/export_nerf_transforms.py`: writes `transforms_train_static.json` (NeRF-style cameras + paths) for one timestep — input toward 3DGS / NeRF forks.
- `scripts/recover_restitution_from_poses.py`: fits the toy 1D bounce to **PyBullet** `z_m` from `object_poses.csv` (bridges sim logs to optimization; expect bias vs true `e` until the sim matches PyBullet or you use pixels).
- `scripts/warp_gaussians_to_frame.py`: rigidly warps trained 3DGS to a target frame via `object_poses.csv`.
- `scripts/eval_physics_trajectory_split.py`: fit restitution on train frames, MSE on held-out test frames.
- `scripts/eval_warped_mask_coverage.py`: project warped sphere Gaussians into masks (proxy for render alignment).
- `src/phys4d/differentiable_bounce.py`: minimal PyTorch differentiable bounce simulator and restitution recovery helper.
- `scripts/recover_restitution.py`: synthetic restitution recovery demo.
- `tests/test_restitution_recovery.py`: `unittest` checks for gradient flow and recovery accuracy.
- `modal_app.py` + `docs/modal.md`: Modal GPU smoke test and remote unittest (`pip install -r requirements-modal.txt`, `modal setup`).

**Start here:** [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — full pipeline (datagen → Modal 3DGS → SuperSplat).

## Quick Checks

Use one Python that has **both** `torch` and `pybullet` (on macOS, `conda install -c conda-forge pybullet` is often easier than `pip install pybullet`). If `import torch` fails with a NumPy `_ARRAY_API` error, pin **`numpy<2`**.

```bash
conda activate phys4d   # or your env with torch + pybullet
python scripts/recover_restitution.py
python -m unittest tests/test_restitution_recovery.py
```

Dataset export (needs `pybullet`, `numpy`, `imageio` — see `requirements-m2.txt` or your project venv):

```bash
python scripts/generate_sphere_bounce_dataset.py
# or: python scripts/generate_sphere_bounce_dataset.py --dry-run
```

If imports fail: `pip install -r requirements-m2.txt`.

## Vision bridge (after dataset exists)

```bash
python scripts/export_nerf_transforms.py
python scripts/recover_restitution_from_poses.py
```

`export_nerf_transforms.py` writes `outputs/sphere_bounce_m2/transforms_train_static.json` next to `rgb/`. Point your **3DGS / NeRF / Nerfstudio** workflow at that folder (you may still need COLMAP conversion depending on the repo).

**Next external step:** clone [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (or your chosen 4DGS fork), run on a GPU machine, and feed either COLMAP from these views or a fork that reads `transforms*.json`.

## Modal (cloud GPU — team credits)

See **`docs/modal.md`**. Quick start:

```bash
pip install -r requirements-modal.txt
modal setup
python scripts/generate_sphere_bounce_dataset.py
python scripts/export_gs_blender_scene.py
modal run modal_app.py --upload    # uploads your PNGs to Modal Volume
modal run modal_app.py --train     # 3DGS on those images (one timestep, 6 views)
```

**Uses your generated RGB**, not synthetic placeholders. Default training is **static** (frame 0 only); full bounce video needs 4DGS later.