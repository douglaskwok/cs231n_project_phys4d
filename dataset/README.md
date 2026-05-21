# 12-View Ping-Pong Dataset

This folder contains the tooling for generating and exporting the synthetic
12-view bouncing-ball dataset. The main entrypoint is:

```bash
dataset/export_ping_pong_12view.py
```

The current recommended dataset is the small room scene:

```text
dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0/
```

It has 12 calibrated cameras, RGB frames, ball masks, object poses, camera
metadata, and preview videos. The room version adds a floor, four walls, and
colored anchor panels so 4DGS has background structure to reconstruct.

Generated outputs are intentionally gitignored.

## Quick Start

Generate the room dataset used for the current 4DGS checks:

```bash
phys_sim/bin/python dataset/export_ping_pong_12view.py \
  --variation-set single \
  --output-dir dataset/outputs/ping_pong_12view_single_fast_room \
  --max-frames 91 \
  --ball-radius-m 0.10 \
  --restitution 0.90 \
  --ball-angle-deg 0.0 \
  --environment room
```

Expected output:

```text
dataset/outputs/ping_pong_12view_single_fast_room/
  dataset_manifest.json
  scene_0000_e0p90_a0p0/
    rgb/cam00/frame00000.png
    masks/cam00/frame00000.png              # ball only
    masks_table/cam00/frame00000.png        # table only
    masks_ball_table/cam00/frame00000.png   # ball + table
    videos/rgb/cam00_front.mp4
    videos/masks/cam00_front.mp4
    videos/masks_table/cam00_front.mp4
    videos/masks_ball_table/cam00_front.mp4
    cameras.json
    object_poses.csv
    physics_params.json
    metadata.json
    config.json
```

View a generated RGB or mask video:

```bash
open dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0/videos/rgb/cam00_front.mp4
open dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0/videos/masks/cam00_front.mp4
open dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0/videos/masks_ball_table/cam00_front.mp4
```

## Exporter Modes

`--variation-set` controls how many generated scenes are made:

```text
single  one scene, using --restitution and --ball-angle-deg
poc     2x2 grid over restitution x ball-angle
full    configured full grid over restitution x ball-angle
none    legacy mode: export existing notebook videos from pybullet_outputs/
```

`--environment` controls the generated background:

```text
room      floor, walls, colored anchor panels; recommended
tabletop  old black-background tabletop scene
```

Useful flags:

```text
--ball-radius-m        ball radius in meters; current default is 0.10
--video-fps            output/render FPS for generated scenes; default 60
--sim-hz               internal PyBullet rate; must divide evenly by video FPS
--duration-sec         generated clip duration before --max-frames is applied
--max-frames           cap frame count for fast tests
--no-videos            write PNGs only, no MP4 previews
--render-camera-names  comma-separated camera names, or all
--video-camera-names   comma-separated camera names for preview MP4s, or all
```

Keep `--render-camera-names all` for 4DGS training because it needs the full
multi-view dataset.

## Higher-Framerate Exports

The default generated dataset is 60 FPS with an internal PyBullet rate of
480 Hz. For denser temporal supervision, generate 120 FPS while keeping the same
rough 1.5 second training/test window:

```bash
phys_sim/bin/python dataset/export_ping_pong_12view.py \
  --variation-set single \
  --output-dir dataset/outputs/ping_pong_12view_single_fast_room_120fps \
  --duration-sec 1.5 \
  --video-fps 120 \
  --sim-hz 480 \
  --ball-radius-m 0.10 \
  --restitution 0.90 \
  --ball-angle-deg 0.0 \
  --environment room
```

This gives:

```text
181 frames total at 120 FPS
train_frames: 0..119
test_frames:  120..179
steps_per_frame: 4
```

For 120 FPS 4DGS tests, use:

```bash
arch -arm64 modal run modal_app.py \
  --train-4d \
  --train-4d-config sphere_bounce_4dgs_quick_120fps.yaml
```

For 240 FPS, `--sim-hz 480` still works but gives only 2 simulation steps per
video frame. If the motion gets numerically noisy, use a higher internal rate
that is still an integer multiple of the video FPS, for example:

```bash
--video-fps 240 --sim-hz 960
```

## Convert To 4DGS Inputs

Use `4dgs/experiments/object_only/export_object_only_dynerf.py` to convert
the generated dataset into the DyNeRF-style folder expected by the Modal 4DGS
pipeline.

Set this once for shorter commands:

```bash
SCENE=dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0
```

Full video, including room, table, and ball:

```bash
phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks" \
  --output 4dgs/experiments/object_only/runs/ping_pong_room_full \
  --mode full
```

Background-only video, with the ball removed:

```bash
phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks" \
  --output 4dgs/experiments/object_only/runs/ping_pong_room_background \
  --mode background \
  --background black
```

Ball-only video, with everything outside the mask blacked out:

```bash
phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks" \
  --output 4dgs/experiments/object_only/runs/ping_pong_room_ball_only \
  --mode object \
  --background black
```

Ball+table foreground video, with the room/background blacked out:

```bash
phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks_ball_table" \
  --output 4dgs/experiments/object_only/runs/ping_pong_room_ball_table_foreground \
  --mode object \
  --background black
```

Room background with ball+table removed:

```bash
phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks_ball_table" \
  --output 4dgs/experiments/object_only/runs/ping_pong_room_without_ball_table \
  --mode background \
  --background black
```

Each output contains:

```text
images/
transforms_train.json
transforms_test.json
export_meta.json
```

## Complete Three-Variant 4DGS Run

For the current experiment, we usually want to inspect three reconstructions:

```text
full        room + table + ball
background  room + table, with the ball removed
ball-only   only the segmented ball, black outside the mask
```

First generate the room dataset and export the three DyNeRF folders using the
commands above. Then run the following blocks one at a time.

### Full Scene

Upload the full-scene DyNeRF folder:

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py \
  4dgs/experiments/object_only/runs/ping_pong_room_full \
  --modal-cmd "arch -arm64 modal"
```

Train quick 4DGS. Use `sphere_bounce_4dgs_quick.yaml` for the default 60 FPS
dataset, or `sphere_bounce_4dgs_quick_120fps.yaml` for the 120 FPS dataset:

```bash
arch -arm64 modal run modal_app.py \
  --train-4d \
  --train-4d-config sphere_bounce_4dgs_quick.yaml
```

Render the trained checkpoint:

```bash
arch -arm64 modal run modal_app.py \
  --render-4d \
  --render-4d-checkpoint chkpnt1000.pth
```

Download the render frames:

```bash
mkdir -p latest_room_full_4dgs_download

arch -arm64 modal volume get phys4d-gs-output \
  4dgs_renders/latest \
  latest_room_full_4dgs_download \
  --force
```

Build local MP4 previews:

```bash
phys_sim/bin/python 4dgs/scripts/build_4dgs_render_videos.py \
  latest_room_full_4dgs_download/latest
```

View the full-scene videos:

```bash
open latest_room_full_4dgs_download/latest/videos/train_cams_grid.mp4
open latest_room_full_4dgs_download/latest/videos/test_cams_grid.mp4
```

### Background-Only

Upload the background-only DyNeRF folder:

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py \
  4dgs/experiments/object_only/runs/ping_pong_room_background \
  --modal-cmd "arch -arm64 modal"
```

Train quick 4DGS. Use `sphere_bounce_4dgs_quick_120fps.yaml` if this was
exported from the 120 FPS dataset:

```bash
arch -arm64 modal run modal_app.py \
  --train-4d \
  --train-4d-config sphere_bounce_4dgs_quick.yaml
```

Render the trained checkpoint:

```bash
arch -arm64 modal run modal_app.py \
  --render-4d \
  --render-4d-checkpoint chkpnt1000.pth
```

Download the render frames:

```bash
mkdir -p latest_room_background_4dgs_download

arch -arm64 modal volume get phys4d-gs-output \
  4dgs_renders/latest \
  latest_room_background_4dgs_download \
  --force
```

Build local MP4 previews:

```bash
phys_sim/bin/python 4dgs/scripts/build_4dgs_render_videos.py \
  latest_room_background_4dgs_download/latest
```

View the background-only videos:

```bash
open latest_room_background_4dgs_download/latest/videos/train_cams_grid.mp4
open latest_room_background_4dgs_download/latest/videos/test_cams_grid.mp4
```

### Ball-Only

Upload the ball-only DyNeRF folder:

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py \
  4dgs/experiments/object_only/runs/ping_pong_room_ball_only \
  --modal-cmd "arch -arm64 modal"
```

Train quick 4DGS. Use `sphere_bounce_4dgs_quick_120fps.yaml` if this was
exported from the 120 FPS dataset:

```bash
arch -arm64 modal run modal_app.py \
  --train-4d \
  --train-4d-config sphere_bounce_4dgs_quick.yaml
```

Render the trained checkpoint:

```bash
arch -arm64 modal run modal_app.py \
  --render-4d \
  --render-4d-checkpoint chkpnt1000.pth
```

Download the render frames:

```bash
mkdir -p latest_room_ball_only_4dgs_download

arch -arm64 modal volume get phys4d-gs-output \
  4dgs_renders/latest \
  latest_room_ball_only_4dgs_download \
  --force
```

Build local MP4 previews:

```bash
phys_sim/bin/python 4dgs/scripts/build_4dgs_render_videos.py \
  latest_room_ball_only_4dgs_download/latest
```

View the ball-only videos:

```bash
open latest_room_ball_only_4dgs_download/latest/videos/train_cams_grid.mp4
open latest_room_ball_only_4dgs_download/latest/videos/test_cams_grid.mp4
```

### View All Three

After all three blocks have completed and been downloaded:

```bash
open latest_room_full_4dgs_download/latest/videos/train_cams_grid.mp4
open latest_room_background_4dgs_download/latest/videos/train_cams_grid.mp4
open latest_room_ball_only_4dgs_download/latest/videos/train_cams_grid.mp4
```

## What The Modal Commands Mean

The Modal 4DGS pipeline always reads from:

```text
phys4d-gs-data:/4d_scene
```

and writes to:

```text
phys4d-gs-output:/4dgs_sphere_bounce
phys4d-gs-output:/4dgs_renders/latest
```

That means every new upload/train/render overwrites the previous remote result.
Download each variant before starting the next one.

The local upload command chooses which variant to train:

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py <local_dynerf_folder> --modal-cmd "arch -arm64 modal"
```

The train command trains whatever was last uploaded:

```bash
arch -arm64 modal run modal_app.py --train-4d --train-4d-config sphere_bounce_4dgs_quick.yaml
```

For 120 FPS exports, replace the config name:

```bash
arch -arm64 modal run modal_app.py --train-4d --train-4d-config sphere_bounce_4dgs_quick_120fps.yaml
```

The render command renders the latest trained checkpoint:

```bash
arch -arm64 modal run modal_app.py --render-4d --render-4d-checkpoint chkpnt1000.pth
```

The download command preserves the current remote render locally. Use a unique
local folder for each variant:

```bash
arch -arm64 modal volume get phys4d-gs-output 4dgs_renders/latest <local_download_folder> --force
```

## Common Confusion

`arch -arm64 modal run modal_app.py --train-4d ...` does not select full,
background, or ball-only by itself. It trains whatever was last uploaded to
`phys4d-gs-data:/4d_scene`.

So the selector is the upload command:

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py <local_dynerf_folder> --modal-cmd "arch -arm64 modal"
```

The render download should be a directory containing `latest/*.png`. If the
download target becomes a single PNG file, create a fresh directory and download
again:

```bash
mkdir -p latest_room_current_download
arch -arm64 modal volume get phys4d-gs-output 4dgs_renders/latest latest_room_current_download --force
```

Then build videos from:

```bash
phys_sim/bin/python 4dgs/scripts/build_4dgs_render_videos.py latest_room_current_download/latest
```

## Legacy Notebook Export

The exporter can still convert notebook MP4s from `pybullet_outputs/`.

The notebook writes:

```text
pybullet_outputs/ping_pong_bounce_12view_<camera>.mp4
pybullet_outputs/ping_pong_bounce_12view_cameras.csv
pybullet_outputs/ping_pong_bounce_12view_trajectory.csv
```

Convert them with:

```bash
phys_sim/bin/python dataset/export_ping_pong_12view.py \
  --variation-set none \
  --input-dir pybullet_outputs \
  --output-dir dataset/outputs/ping_pong_12view
```

In legacy export mode, masks are RGB-derived blue-ball masks rather than
PyBullet instance masks. Generated scenes use PyBullet instance masks directly.

## Teammate Handoff

Copy-paste note:

```text
Use dataset/export_ping_pong_12view.py for the synthetic 12-view data. The main
dataset we are using is:

dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0

Generate it with the Quick Start command in dataset/README.md. It produces RGB,
ball masks, cameras.json, object_poses.csv, config.json, and preview MP4s. For
4DGS, convert it with 4dgs/experiments/object_only/export_object_only_dynerf.py.
Use --mode full for the full room, --mode background to remove the ball, and
--mode object for ball-only. Modal always overwrites its remote 4DGS output, so
download each variant before training/rendering the next one.
```
