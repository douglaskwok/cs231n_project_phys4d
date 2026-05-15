# Modal (GPU credits)

Team credits include **Modal USD 200** (see `docs/PROGRESS.md`).

## Does training use our generated video?

**Partially — by design for the first 3DGS run:**

| Data | Used? |
|------|--------|
| `outputs/sphere_bounce_m2/rgb/cam*/frame*.png` | **Yes** — copied into `gs_blender_frame*/train/*.png` |
| `outputs/sphere_bounce_m2/cameras.json` | **Yes** — becomes `transforms_train.json` poses |
| Masks, full 90-frame time series | **No** for vanilla 3DGS (static scene = **one** `frame` index, default `0`) |

So you are training on **your simulator images**, but on a **single time slice across 6 cameras**, not the full bouncing video. That matches **static** 3D Gaussian Splatting. **4D / dynamic** splatting is a later step.

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

First `--train` builds a large image (clones `gaussian-splatting` + CUDA submodules); expect several minutes.

## Costs

Billed to Modal credits: GPU type (`A10G` in `train_gs`), image build time, and iteration count (`--train` uses 7000 by default vs 30k full).

## Troubleshooting

- **No scene on volume:** run `--upload` after local `generate_sphere_bounce_dataset.py`.
- **Axis flip / mirrored render:** Blender loader in 3DGS flips Y/Z; if previews look wrong, we may add a fixed axis correction in `colmap_export.py`.
- **Full video:** train separate exports per frame or switch to a 4DGS codebase; not covered by `train_gs` yet.
