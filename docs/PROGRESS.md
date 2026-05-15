# Project progress log

This file tracks implementation status for the CS231N Phys4D / Milestone 2 direction. Update it when you merge meaningful changes or finish experiments.

## Compute budget (team)

- **Approximate cloud credits available:** USD **350** total — **Modal USD 200**, **AWS USD 100**, **Google Cloud USD 50**. Use these for long 3DGS / 4DGS training runs, GPU baselines, or hosted rendering when local machines are saturated.
- **Suggestion:** spend local CPU first for PyBullet datagen and small PyTorch tests; reserve credits for Gaussian Splatting optimization jobs that need many hours on a single GPU.
- **Rough routing:** Modal is convenient for ephemeral GPU jobs from Python/`modal run`; AWS/GCP suit renting a GPU VM (EC2 / GCE) and `ssh` + training. I (the IDE agent) cannot spend these credits unless you run cloud CLIs or Modal from your machine with your logged-in account.

## 2026-05-14

- **Scaffold (M2):** docs for claim, method equations, 7-slide outline; `configs/sphere_bounce_m2.json` locks first experiment (sphere-ground, restitution 0.72, 6 cameras, train/test frame split).
- **Differentiable toy:** `src/phys4d/differentiable_bounce.py` implements vertical bounce with differentiable restitution; `scripts/recover_restitution.py` and `tests/test_restitution_recovery.py` verify synthetic recovery.
- **Data pipeline (PyBullet):** added `scripts/generate_sphere_bounce_dataset.py` to export multi-view RGB, segmentation masks, camera metadata, and sphere pose CSV under `outputs/sphere_bounce_m2/` per config paths (requires `pybullet`, `numpy`, `imageio`; see `requirements-m2.txt`). Use `python scripts/generate_sphere_bounce_dataset.py --dry-run` to print paths without importing sim deps.
- **Repo hygiene:** `outputs/` added to `.gitignore` so generated RGB trees are not committed by accident.
- **Progress log:** use this file as the running team record (append dated bullets when you ship features or results).

## 2026-05-15

- **Modal + 3DGS:** `export_gs_blender_scene.py` packages PyBullet RGB; `modal run modal_app.py --upload/--train` uploads and runs graphdeco 3DGS on **one timestep** (your PNGs, not full video).
- **First 3DGS train completed:** 7000 iterations, train PSNR ~46.6; checkpoint on Modal volume `phys4d-gs-output` (`gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply`). View in SuperSplat, not raw point-cloud viewers.
- **Runbook:** `docs/WORKFLOW.md` documents end-to-end commands (datagen, upload, train, download, SuperSplat).
- **End-to-end test (local `phys4d` conda env):** `scripts/recover_restitution.py` recovered restitution 0.72 with final loss 0; `python -m unittest tests/test_restitution_recovery.py` passed (2 tests); `scripts/generate_sphere_bounce_dataset.py` wrote RGB, masks, `cameras.json`, `object_poses.csv` (91 lines for 90 steps + header), and `metadata.json` under `outputs/sphere_bounce_m2/`.
- **Environment gotcha:** PyTorch wheels built against NumPy 1.x can error under **NumPy 2.x** (`_ARRAY_API not found`). For `phys4d`, keep **`numpy<2`** (e.g. 1.26.x) alongside conda-installed **PyTorch** and conda-forge **pybullet** + **imageio**. Prefer `conda install` for PyTorch/pybullet on macOS when `pip install pybullet` tries to compile from source.
- **Vision bridge:** `scripts/export_nerf_transforms.py` writes NeRF-style `transforms_train_static.json` (6 cameras, default frame 0) next to `rgb/` for downstream 3DGS / NeRF tooling. `scripts/recover_restitution_from_poses.py` fits the toy 1D bounce to PyBullet `z_m`; expect bias vs config `e` until the sim matches PyBullet or losses use images.

## Next (short list)

1. ~~Run dataset generator once~~ done 2026-05-15; spot-check RGB and masks; commit **metadata only** if you want reproducibility without large binaries (or store samples elsewhere).
2. ~~Wire PyBullet `object_poses.csv` into restitution fitting~~ done via `recover_restitution_from_poses.py` (pose-only; refine with pixel loss next).
3. ~~Stand up vanilla 3DGS baseline~~ static 3DGS done; next: 4DGS or pose-linked Gaussians + physics joint opt.
4. Optional: SAM or motion clustering only after PyBullet-mask path is stable.

## Open risks

- Segmentation and object grouping remain the main integration risk for real video; M2 stays on simulator masks.
- Static 3DGS init quality gates everything downstream; budget time before joint physics+appearance optimization.
