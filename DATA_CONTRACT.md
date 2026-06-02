# Data contract — bounce pipeline (Steps 4a–6)

This document specifies the **exact artifacts, schemas, and naming conventions** an
upstream agent must produce (working with [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians))
so the trajectory-extrapolation pipeline (`scripts/bounce/step4a..step6`) can run on a scene.

Everything downstream is path-driven. The code parses these files literally — field
names, filename padding, and the `frame` join key must match exactly.

See also: [`milestone3.md`](milestone3.md) (pipeline spec) and
[`STAGES_4_6_COMMANDS.txt`](STAGES_4_6_COMMANDS.txt) (commands).

---

## Three artifact groups

To run the pipeline on a scene, produce:

- **Group A** — Wu 4DGS training output (canonical Gaussians + deformation + cfg).
- **Group B** — DyNeRF export (the dataset Wu trained on; carries the split + time map).
- **Group C** — PyBullet scene ground truth (validation + render compositing).

---

## Group A — Wu 4DGS training output

Produced by training Wu 4DGaussians on the **object-only, train-frames-only** DyNeRF export.

```
<wu_run>/
  cfg_args                                  # text: repr(argparse.Namespace)
  point_cloud/iteration_<N>/
    point_cloud.ply                         # canonical 3D Gaussians
    deformation.pth                         # deform_network state_dict
    deformation_table.pth                   # loaded by GaussianModel.load_model
    deformation_accum.pth
```

| File | Type / schema | Consumed by | Notes |
|---|---|---|---|
| `point_cloud.ply` | Binary PLY. Vertex props: `x y z`, `opacity` (**logit**, pre-sigmoid), `scale_0..2`, `rot_0..3`, `f_dc_0..2`, `f_rest_0..N` (SH). Loaded via `GaussianModel.load_ply`. | 4a, 5 | `sh_degree=3` ⇒ 48 SH coeffs. Opacity is the raw logit; the getter applies sigmoid. |
| `deformation.pth` | `torch` state dict for `deform_network(args)`. | 4a | Loaded with `gaussians.load_model(iteration_dir)`. |
| `cfg_args` | Text eval'able to `Namespace(...)`. Includes `sh_degree, net_width, defor_depth, posebase_pe, kplanes_config, bounds, …`. | 4a, 5 | **Must NOT contain `all_train=True`** — Step 4a raises if it does. Rebuilds the deform-net architecture. |

GPU: loading + deforming requires the built Wu CUDA extensions (`simple-knn`,
`depth-diff-gaussian-rasterization`).

---

## Group B — DyNeRF export (Wu training dataset)

The bridge between Wu and physics. The train/test split and per-frame time mapping live here.

```
<export>/
  transforms_train.json
  transforms_test.json
  frame_map.json          # authoritative split + time mapping
  export_meta.json
  images/cam{cc}_{fffff}.png
```

### `frame_map.json` — most important file

```json
{
  "fps": 120.0,
  "num_frames": 61,
  "train": [{"kept_index": 0, "original_frame": 0,  "original_time_s": 0.0}],
  "test":  [{"kept_index": 0, "original_frame": 41, "original_time_s": 0.3417}]
}
```

- `train` / `test`: arrays of `{kept_index:int, original_frame:int, original_time_s:float(sec)}`.
- `original_frame` = PyBullet sim frame index (links to `object_poses.csv` and GT `rgb/`).
- `test` **must be non-empty**, else Step 4c aborts (means an all-train export).
- Step 4a derives `t_norm = (t − t_min) / (t_max − t_min)` over **train** `original_time_s`.
  This time normalization is the pipeline's #1 silent-failure point — get it right.

### `transforms_train.json` / `transforms_test.json`

```json
{
  "w": 960, "h": 544,
  "fl_x": 1059.37, "fl_y": 1059.37, "cx": 480.0, "cy": 272.0,
  "frames": [
    {"file_path": "images/cam00_00041", "time": 0.0,
     "transform_matrix": [[/* 4x4 camera-to-world */]]}
  ]
}
```

- `file_path` stem **must** be `cam{cam:02d}_{frame:05d}` — Step 5 parses cam + frame from it.
- `transform_matrix` = 4×4 **camera-to-world** (Blender/NeRF convention; Step 5 inverts it).
- `transforms_test.json` is required for Step 5 (held-out camera poses).

### `export_meta.json`

Informational. `{format, source_config, rgb_root, masks_root, mode:"object", background,
fps, train_cameras[], test_cameras[], train_frame_range, test_frame_range,
train_timestamps[], test_timestamps[], num_*}`. `fps` is a fallback for Step 4c.

---

## Group C — PyBullet scene ground truth

```
<scene>/
  object_poses.csv
  config.json
  rgb/cam{cc}/frame{fffff}.png
  masks/cam{cc}/frame{fffff}.png
  background_3dgs/background.ply
```

| File | Schema | Consumed by |
|---|---|---|
| `object_poses.csv` | Header **must** include `frame,time_s,x_m,y_m,z_m,qx,qy,qz,qw,vx_m_s,vy_m_s,vz_m_s` (extra columns ignored). One row per sim frame, covering **both train and test** frames. | 4a (overlay/MSE), 6 (RMSE/R²) |
| `config.json` | `{fps:float, train_frames:int, test_frames:int, scene:{objects:[{name:"sphere", shape:"sphere", radius_m:float}]}}` | 4c (split check), 5 (sphere radius) |
| `rgb/cam{cc}/frame{fffff}.png` | RGB renders. `cam` zero-padded to 2, `frame` to 5. | 6 (PSNR/SSIM), 5 overlay fallback |
| `masks/cam{cc}/frame{fffff}.png` | Object masks (white = object). | background training, overlay mask-out |
| `background_3dgs/background.ply` | Standard 3DGS PLY (static room/table). | 5 |

---

## Filename conventions (must match exactly)

| Where | Pattern | Example |
|---|---|---|
| Export images / transforms `file_path` | `cam{cam:02d}_{frame:05d}` | `cam00_00041` |
| Scene GT rgb/masks | `cam{cam:02d}/frame{frame:05d}.png` | `cam10/frame00060.png` |
| Step 5 render output | `{frame:04d}_cam{cam}.png` | `0041_cam10.png` |

`frame` everywhere = **PyBullet sim frame index** — the join key across all three groups.

---

## Hard invariants

1. **Train-only Wu training** — no `--all-train`; `frame_map.test` non-empty. Otherwise
   held-out evaluation is invalid (4a/4c abort).
2. **Time mapping** — `frame_map.original_time_s` must reflect the real per-frame times used
   at train time.
3. **Opacity = logit** in the PLY (the Wu getter applies sigmoid).
4. **Consistent `frame` indices** across `frame_map`, `object_poses.csv`, and `rgb/`/`masks/`
   filenames.
5. World-frame offset between the 3DGS frame and the PyBullet frame is acceptable — handled
   via Procrustes in Step 4a and pose-deltas in Step 5.

---

## Per-step input map

| Step | Reads | GPU? |
|---|---|---|
| 4a | A: `point_cloud.ply`, `deformation.pth`, `cfg_args` + B: `frame_map.json` / `transforms_train.json` + C: `object_poses.csv` | yes (deform net) |
| 4b | 4a `trajectory_smoothed.csv` | no |
| 4c | 4b `physics_params.json` + B: `frame_map.json` (+ C: `config.json`) | no |
| 5  | A: `point_cloud.ply`, `cfg_args` + C: `background.ply`, `config.json`, `rgb/` + B: `transforms_test.json` + 4a/4c CSVs | yes (rasterizer) |
| 6  | 4c `trajectory_predicted.csv` + C: `object_poses.csv`, `rgb/` + 5 `renders/` | no |

**Minimum handoff for a brand-new scene:** Group A (Wu train output), Group B
(`frame_map.json` + `transforms_{train,test}.json`), and Group C (`object_poses.csv`,
`config.json`, `rgb/`, `masks/`, `background.ply`).
