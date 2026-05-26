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
The current PyBullet contact settings force restitution to be applied at the
table contact, so the ball actually rebounds instead of settling onto the table.

Generated outputs are intentionally gitignored.

## Final Dataset

The current final PyBullet dataset lives at:

```text
dataset/outputs/phys4d_final/
```

Generate or refresh it with:

```bash
phys_sim/bin/python dataset/generate_phys4d_final.py --video-fps 60
```

The ball-drop split is a compact 3x3 grid centered on the demo-like bounce:

```text
restitution:    0.70, 0.78, 0.86
ball_angle_deg: -8.0, 0.0, 8.0
fps:            60
sim_hz:         480
```

The angle is implemented as an approximate first-impact trajectory angle from
vertical. The exporter computes the required horizontal velocity from the drop
height while leaving gravity to determine the vertical speed.

### First 4DGS Timing Run

Use this wrapper to train one final scene, download the Gaussian artifacts, and
record wall-clock timing:

```bash
bash 4dgs/scripts/train_one_final_scene_4dgs.sh \
  dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_0004_e0p78_a0p0 \
  ball_drop_e0p78_a0p0_object
```

It writes:

```text
4dgs/experiments/object_only/runs/ball_drop_e0p78_a0p0_object/
latest_ball_drop_e0p78_a0p0_object_4dgs/
latest_ball_drop_e0p78_a0p0_object_4dgs/timing.json
```

For a quick smoke test before the quality run, add `--quick`.

For per-object scenes, change `--mask-subdir`. Examples:

```bash
# Collision object A only
bash 4dgs/scripts/train_one_final_scene_4dgs.sh \
  dataset/outputs/phys4d_final/collision_base_60fps/scene_0000_collision_room \
  collision_object_a \
  --mask-subdir masks_object_a

# First stacking block only
bash 4dgs/scripts/train_one_final_scene_4dgs.sh \
  dataset/outputs/phys4d_final/stacking_base_60fps/scene_0000_stacking_room \
  stacking_block_00 \
  --mask-subdir masks_block_00
```

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
    videos/rgb/cam00_front.mp4
    videos/masks/cam00_front.mp4
    videos/masks_table/cam00_front.mp4
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
open dataset/outputs/ping_pong_12view_single_fast_room/scene_0000_e0p90_a0p0/videos/masks_table/cam00_front.mp4
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

Room background with ball removed:

```bash
phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks" \
  --output 4dgs/experiments/object_only/runs/ping_pong_room_without_ball \
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

## Other Room Physics Scenarios

Use this exporter for the additional 12-view room scenarios:

```bash
dataset/export_room_physics_12view.py
```

It uses the same 12-camera room rig as the ping-pong exporter: floor, walls,
colored anchor panels, table, RGB frames, PyBullet instance masks, camera
metadata, object poses, and preview videos. These scenarios are meant to test
whether reconstruction methods can handle physics beyond one bouncing sphere.

### Scenario Catalog

`collision`

Two rigid boxes slide toward each other on the table and collide. The red box is
`object_a`; the blue box is `object_b`. The dynamic foreground mask `masks/` is
the union of both boxes, while `masks_object_a/` and `masks_object_b/` let you
train or evaluate each object separately. The default scene lasts 2.6 seconds:
157 frames at 60 FPS with 480 Hz PyBullet simulation.

The collision boxes are intentionally enlarged for reconstruction experiments:
the default full size is `0.18 m x 0.18 m x 0.08 m`. Earlier small boxes were
easy to segment but occupied too few pixels for stable object-only 4DGS.

This scenario is useful for testing multi-object motion, contact, and occlusion.
It is also intentionally hard for plain object-only 4DGS: if trained on black
background with only 1000 iterations, the reconstruction may keep the easier
persistent object and drop the other one. Use the per-object masks or
`masks_object_table/` if you want a more stable diagnostic.

`stacking`

Three enlarged colored rigid blocks are released sequentially and settle into a
small stack. The blocks are named `block_00` through `block_02`, with corresponding
per-object masks like `masks_block_00/`. The dynamic foreground mask `masks/`
contains all active blocks. The default scene lasts 5.0 seconds: 301 frames at
60 FPS with 480 Hz PyBullet simulation.

The stacking blocks are intentionally larger than the first pass: the default
full size is `0.12 m x 0.12 m x 0.07 m`. Three blocks is the first target because
it gives contact, occlusion, and a stable stack without making object-only 4DGS
fight six tiny moving masks at once.

This scenario tests repeated contacts, resting contact stability, partial
occlusion, and whether a model can represent several objects that become
spatially close over time.

`deformable`

A blue soft torus drops onto the table and deforms. The object is named
`soft_torus`, with a per-object mask in `masks_soft_torus/`. The default scene
lasts 2.6 seconds: 157 frames at 60 FPS with 480 Hz PyBullet simulation. The
exporter records soft-mesh pose statistics in `object_poses.csv`, including mesh
height span and approximate XY radius.

This scenario tests non-rigid motion. It is the least like the rigid ping-pong
case, so use it as a stress test rather than the first debugging target.

### Supported Names

Pass one scenario name, a comma-separated list, or `all`:

```text
collision    two rigid boxes collide on the table
stacking     six blocks are released sequentially into a stack
deformable   soft torus drops and deforms on the table
all          generate all three
```

Generate one instance of all three in the room setting:

```bash
phys_sim/bin/python dataset/export_room_physics_12view.py \
  --scenario all \
  --output-dir dataset/outputs/room_physics_12view_base
```

Expected output:

```text
dataset/outputs/room_physics_12view_base/
  dataset_manifest.json
  scene_0000_collision_room/
  scene_0001_stacking_room/
  scene_0002_deformable_room/
```

Each scene follows the same general structure as the ping-pong room export:

```text
rgb/cam00/frame00000.png
masks/cam00/frame00000.png                # all dynamic objects
masks_table/cam00/frame00000.png          # table only
masks_object_table/cam00/frame00000.png   # dynamic objects + table
masks_<object_name>/cam00/frame00000.png  # per-object masks
videos/rgb/cam00_front.mp4
videos/masks/cam00_front.mp4
videos/masks_table/cam00_front.mp4
videos/masks_object_table/cam00_front.mp4
cameras.json
object_poses.csv
metadata.json
config.json
```

For a quick proof of concept:

```bash
phys_sim/bin/python dataset/export_room_physics_12view.py \
  --scenario all \
  --output-dir dataset/outputs/room_physics_12view_poc \
  --max-frames 91 \
  --video-camera-names front,top_oblique
```

This still writes PNG frames/masks for all 12 cameras, but only writes preview
MP4s for `front` and `top_oblique`.

View examples:

```bash
open dataset/outputs/room_physics_12view_base/scene_0000_collision_room/videos/rgb/cam00_front.mp4
open dataset/outputs/room_physics_12view_base/scene_0001_stacking_room/videos/rgb/cam00_front.mp4
open dataset/outputs/room_physics_12view_base/scene_0002_deformable_room/videos/rgb/cam00_front.mp4
```

Useful masks for 4DGS-style object exports:

```text
masks                 dynamic foreground only
masks_table           table only
masks_object_table    dynamic foreground + table
masks_<object_name>   one named object only, when available
```

For example, export collision dynamic objects only:

```bash
SCENE=dataset/outputs/room_physics_12view_base/scene_0000_collision_room

phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks" \
  --output 4dgs/experiments/object_only/runs/collision_room_objects \
  --mode object \
  --background black
```

Export dynamic objects plus table:

```bash
phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks_object_table" \
  --output 4dgs/experiments/object_only/runs/collision_room_objects_table \
  --mode object \
  --background black
```

Export only one collision object:

```bash
phys_sim/bin/python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --config "$SCENE/config.json" \
  --masks-root "$SCENE/masks_object_a" \
  --output 4dgs/experiments/object_only/runs/collision_room_object_a \
  --mode object \
  --background black
```

For quick 4DGS runs, use the config matching the scene duration:

```text
room_physics_4dgs_quick_2p6s.yaml  collision and deformable
room_physics_4dgs_quick_5p0s.yaml  stacking
```

For quality 4DGS runs, use the 15k-iteration configs:

```text
room_physics_4dgs_2p6s.yaml  collision and deformable
room_physics_4dgs_5p0s.yaml  stacking
```

The Modal 4DGS pipeline still overwrites `/data/4d_scene`,
`/outputs/4dgs_sphere_bounce`, and `/outputs/4dgs_renders/latest`, so train and
download one scenario at a time.

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
