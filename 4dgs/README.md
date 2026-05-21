# 4D Gaussian Splatting (4DGS)

This folder groups everything for **[fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)** (ICLR 2024). The upstream training code is **not** vendored here; Modal clones it to `/opt/4dgs`, or you clone it locally and set `FOURDGS_ROOT`.

## Layout

```
4dgs/
  README.md                 ← you are here
  configs/                  ← fudan train.py YAML (sphere bounce, quick variants)
  scripts/                  ← export, train wrappers, render, eval, upload
  experiments/
    object_only/            ← masked DyNeRF export for ball-only 4DGS
```

Related code **outside** this folder:

| Path | Role |
|------|------|
| `src/phys4d/dynerf_export.py` | PyBullet RGB → DyNeRF `transforms_*.json` layout |
| `modal_app.py` | GPU train / render / eval on Modal |
| `segmentation/sa4d/` | SA4D-inspired identity table (uses exported 4DGS PLY) |

## Prerequisites

1. **Sim data** (multi-view RGB + cameras), e.g. `outputs/sphere_bounce_m2/` or `dataset/outputs/...`
2. **CUDA machine** for local train/render, *or* Modal account (`pip install -r requirements-modal.txt`, `modal setup`)
3. **Optional local clone** of upstream 4DGS:

```bash
git clone --recursive https://github.com/fudan-zvg/4d-gaussian-splatting.git
export FOURDGS_ROOT=/path/to/4d-gaussian-splatting
```

## Quick start (sphere bounce, Modal)

From the **repository root**:

```bash
# 1) Export full bounce video to DyNeRF layout
python 4dgs/scripts/export_4dgs_dataset.py

# 2) Upload + train on Modal (first run builds CUDA extensions; can take 10–30+ min)
modal run modal_app.py --upload-4d
modal run modal_app.py --train-4d

# 3) Export PLY, render MP4, metrics
modal run modal_app.py --export-4d-ply
modal run modal_app.py --render-4d
modal run modal_app.py --eval-4d

# 4) Download artifacts
modal volume get phys4d-gs-output 4dgs_sphere_bounce . --force
modal volume get phys4d-gs-output 4dgs_renders/latest . --force
```

Upload **without** rebuilding the 4DGS Modal image (data only):

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py outputs/sphere_bounce_m2/dynerf_sphere_bounce
modal run modal_app.py --train-4d
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/export_4dgs_dataset.py` | Full-scene DyNeRF export from sim config (`configs/sphere_bounce_m2.json`) |
| `scripts/upload_4d_scene_to_modal.py` | Push a prepared DyNeRF folder to `phys4d-gs-data:/4d_scene` |
| `scripts/export_4dgs_ply.py` | `chkpnt*.pth` → SuperSplat-friendly PLY |
| `scripts/render_4dgs_trajectory.py` | Rasterize checkpoint (dataset views or orbit) |
| `scripts/eval_4dgs_metrics.py` | PSNR / MAE vs GT PNGs (train + test splits) |
| `scripts/build_4dgs_render_videos.py` | MP4 from downloaded render PNGs |
| `scripts/view_4dgs_time.py` | Interactive HTML viewer — scrub time, multi-cam grid, GT compare |
| `scripts/run_ping_pong_sa4d_pipeline.sh` | Ping-pong dataset → 4DGS → SA4D-style table |

Configs (under `4dgs/configs/`): `sphere_bounce_4dgs.yaml` (15k iters), `sphere_bounce_4dgs_quick.yaml`, `sphere_bounce_4dgs_quick_120fps.yaml`.

## Local train / render (advanced)

Requires `FOURDGS_ROOT`, CUDA, and extensions built per upstream README.

```bash
export FOURDGS_ROOT=/path/to/4d-gaussian-splatting
cd "$FOURDGS_ROOT"
python train.py --config /path/to/repo/4dgs/configs/sphere_bounce_4dgs.yaml
```

Render / metrics from repo root:

```bash
python 4dgs/scripts/render_4dgs_trajectory.py \
  --config 4dgs/configs/sphere_bounce_4dgs.yaml \
  --dataset outputs/sphere_bounce_m2/dynerf_sphere_bounce \
  --checkpoint path/to/chkpnt15000.pth \
  --out-dir /tmp/4dgs_render

python 4dgs/scripts/eval_4dgs_metrics.py \
  --config 4dgs/configs/sphere_bounce_4dgs.yaml \
  --dataset outputs/sphere_bounce_m2/dynerf_sphere_bounce \
  --checkpoint path/to/chkpnt15000.pth \
  --out-json metrics_4dgs.json
```

## Time viewer (scrub through training progression)

After `modal run modal_app.py --render-4d`, download renders and open the viewer:

```bash
modal volume get phys4d-gs-output 4dgs_renders/latest ./4dgs_renders/latest --force

python 4dgs/scripts/view_4dgs_time.py build \
  --render-dir 4dgs_renders/latest \
  --dataset outputs/sphere_bounce_m2/dynerf_sphere_bounce \
  --open

# If images do not load from file://, use the built-in server:
python 4dgs/scripts/view_4dgs_time.py serve --dir 4dgs_renders/latest/viewer --open
```

Controls: **time slider**, **camera** dropdown, **Play** (spacebar), **multi-cam grid**, optional **ground truth** side-by-side (needs `--dataset` with `images/`). Arrow keys step frames.

## Object-only experiment

Train 4DGS on masked ball pixels only:

```bash
python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --rgb-root outputs/sphere_bounce_m2/rgb \
  --masks-root outputs/sphere_bounce_m2/masks \
  --cameras-json outputs/sphere_bounce_m2/cameras.json \
  --output 4dgs/experiments/object_only/runs/dynerf_ball_only

modal run modal_app.py --upload-4d \
  --upload-4d-path 4dgs/experiments/object_only/runs/dynerf_ball_only
modal run modal_app.py --train-4d
```

See `experiments/object_only/README.md` for modes (`ball_only`, `background`, etc.).

## Modal volume paths

| Volume | Path | Contents |
|--------|------|----------|
| `phys4d-gs-data` | `/4d_scene` | DyNeRF dataset (upload) |
| `phys4d-gs-output` | `4dgs_sphere_bounce/` | Checkpoints + exported PLY |
| `phys4d-gs-output` | `4dgs_renders/latest/` | Render PNGs + `trajectory.mp4` |
| `phys4d-gs-output` | `4dgs_eval/metrics_4dgs.json` | PSNR / MAE JSON |

## Backward compatibility

Old entry points under `scripts/` forward to `4dgs/scripts/`:

```bash
python scripts/export_4dgs_dataset.py   # → 4dgs/scripts/export_4dgs_dataset.py
```

More detail: `docs/modal.md`, `docs/WORKFLOW.md`.
