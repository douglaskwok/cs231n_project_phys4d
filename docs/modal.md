# Modal (GPU credits)

Team credits include **Modal USD 200** (see `docs/PROGRESS.md`).

## Does training use our generated video?

**Partially — by design for the first 3DGS run:**

| Data | Used? |
|------|--------|
| `outputs/sphere_bounce_m2/rgb/cam*/frame*.png` | **Yes** — copied into `gs_blender_frame*/train/*.png` |
| `outputs/sphere_bounce_m2/cameras.json` | **Yes** — becomes `transforms_train.json` poses |
| Masks, full 90-frame time series | **No** for vanilla 3DGS (static scene = **one** `frame` index, default `0`) |

So you are training on **your simulator images**, but on a **single time slice across 6 cameras**, not the full bouncing video. That matches **static** 3D Gaussian Splatting.

**4DGS** uses the full train split (cams 0–3, frames 0–59) via `export_4dgs_dataset.py` → `dynerf_sphere_bounce/`.

## One-time setup

```bash
pip install -r requirements-modal.txt
modal setup
```

## Local: generate data + package for 3DGS

```bash
conda activate phys4d
cd cs231n_project_phys4d
python scripts/generate_sphere_bounce_dataset.py
python scripts/export_gs_blender_scene.py   # default frame 0
```

Creates `outputs/sphere_bounce_m2/gs_blender_frame00000/` with `transforms_train.json` and `train/cam00.png` … `cam05.png`.

## Modal workflow

```bash
modal run modal_app.py              # GPU smoke
modal run modal_app.py --tests      # remote unittest
modal run modal_app.py --upload     # export + upload scene to Volume
modal run modal_app.py --train      # 3DGS train (default 7000 iters, A10G)
```

After training, download weights:

```bash
modal volume get phys4d-gs-output gs_sphere_bounce . --force
# trained PLY: gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply
```

Do not create an empty `local_gs_out/` directory before download (Modal may error). See `docs/WORKFLOW.md`.

## 4D Gaussian Splatting (full bounce)

```bash
python scripts/export_4dgs_dataset.py
modal run modal_app.py --upload-4d
modal run modal_app.py --train-4d
modal volume get phys4d-gs-output 4dgs_sphere_bounce . --force
```

### 4DGS video (correct time + multiple cameras)

SuperSplat PLY is a single snapshot; to see the bounce with the trained 4D field, rasterize every `transforms_*` view at its `time` field:

```bash
modal run modal_app.py --render-4d
modal volume get phys4d-gs-output 4dgs_renders/latest . --force
# open trajectory.mp4 (PNGs also in that folder)
```

Optional: `--render-4d-checkpoint chkpnt15000.pth`, `--render-fps 30`, `--render-dry-run-max 48` (first N sorted views only).

### Orbit video (novel camera path)

Horizontal orbit around the mean calibrated camera position; intrinsics match the DyNeRF JSON. Time defaults to a linear sweep across the configured `time_duration` (override with `--orbit-time-start` / `--orbit-time-end` on Modal).

```bash
modal run modal_app.py --render-4d-orbit
modal volume get phys4d-gs-output 4dgs_renders/orbit_latest . --force
```

### Metrics vs ground truth (PSNR / MAE)

```bash
modal run modal_app.py --eval-4d
modal volume get phys4d-gs-output 4dgs_eval/metrics_4dgs.json . --force
```

Quick eval (subsample views): `--eval-4d-dry-run-max 80`.

- Repo: [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)
- Config: `configs/sphere_bounce_4dgs.yaml` (15k iters, `gaussian_dim: 4`, `time_duration` matched to train frames)
- Volume paths: `phys4d-gs-data:/4d_scene`, output `phys4d-gs-output/4dgs_sphere_bounce/`

First `--train-4d` builds a separate image (`/opt/4dgs` + `simple-knn`, `pointops2`, rasterizer); expect several minutes.

## Costs

Billed to Modal credits: GPU type (`A10G`), image build time, and iteration count (`--train` 7k iters; `--train-4d` 15k default).

## Troubleshooting

- First `--train` / `--train-4d` builds a large CUDA image; expect several minutes.
- **No scene on volume:** run `--upload` / `--upload-4d` after local dataset generation.
- **Axis flip / mirrored render:** Blender loader flips Y/Z; if previews look wrong, fix in `colmap_export.py` / `dynerf_export.py`.
- **Static vs 4D:** `--train` = one frame; `--train-4d` = temporal model on train cameras/frames only.
