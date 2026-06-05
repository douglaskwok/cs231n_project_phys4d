# Collision pipeline workflow

Multi-body (N-sphere) analogue of `scripts/bounce`. Same 4a→6 structure, but every
stage works on a **list of objects** and the physics step adds object-object
collisions on top of the floor bounce. Per-object CLI flags take comma-separated
values (one per object); scalar flags broadcast to all objects.

Always run modal commands so a progress bar is visible.

In `scripts/collision`:

1. `step4a_extract.py` — run the Wu 4DGS extractor once per object (default savgol
   window 9, polyorder 2). Writes `<out>/obj{k}/trajectory_smoothed.csv`.
2. `refit_metric_gt.py` — per-object Umeyama to the GT metric frame, then a single
   **joint** least-squares fit of all bodies `(p0, v0, e_floor)` + shared
   `restitution_pair`, gravity and ground fixed to GT by default. Reads radii/masses
   from the scene `config.json` (`--config`). Writes `refit_metric.json`.
3. `predict_metric_from_refit.py` — integrate all bodies together from the anchor
   frame through the held-out window. Writes `trajectory_predicted_obj{k}.csv` each.
4. `step5_render.py` on modal with per-object scale + crop radius (estimate good
   values). **Use layered alpha compositing** (`--composite-mode alpha`, default):
   render background and each object in separate 3D passes, then alpha-matte objects
   over the background. Keeps correct placement while avoiding both (a) background
   floaters swallowing objects in a 3D merge and (b) depth-buffer erasure in depth
   mode. Optional: `--object-opacity-boost 3.0`.
5. `step6_eval.py` — per-object trajectory metrics + shared render metrics.

Then create an mp4 for the fused final scene.

## Differences vs bounce

- **Masses + radii** come from `config.json` `scene.objects[k]` (`radius_m`, `mass_kg`).
  Object-object collisions only constrain the **mass ratio**, so masses are fixed by
  default (`--fit-mass-ratio` is not yet exposed).
- **Sub-stepping**: the integrator sub-steps between frames (`--substeps`, default 8)
  so fast spheres can't tunnel through each other at 120 fps.
- **Frictionless / no-spin**: collisions transfer only the normal (line-of-centres)
  component; tangential velocity and spin are ignored.
- The collision refit stores a **per-object** `similarity` list (not a single dict),
  so Step 6's fused extracted-overlay is disabled by default; the test-marker split
  line is still drawn.

## Example (two objects)

```bash
PY=.venv_pipeline/bin/python
RUN=outputs/collision_pipeline/<RUN>
SCENE=dataset/.../scene_xxxx

# 4a — per object
"$PY" scripts/collision/step4a_extract.py \
  --canonical objA/point_cloud.ply,objB/point_cloud.ply \
  --deform    objA/deformation.pth,objB/deformation.pth \
  --cfg-args  objA/cfg_args,objB/cfg_args \
  --dynerf-export runs/objA,runs/objB \
  --gt-poses "$SCENE/object_poses_obj0.csv,$SCENE/object_poses_obj1.csv" \
  --out "$RUN/step4a"

# 4b — joint refit
"$PY" scripts/collision/refit_metric_gt.py \
  --traj "$RUN/step4a/obj0/trajectory_smoothed.csv,$RUN/step4a/obj1/trajectory_smoothed.csv" \
  --gt-poses "$SCENE/object_poses_obj0.csv,$SCENE/object_poses_obj1.csv" \
  --config "$SCENE/config.json" \
  --out "$RUN/step4b_metric" \
  --train-start 0 --train-end 39 --test-start 40 --test-end 60

# 4c — joint predict
"$PY" scripts/collision/predict_metric_from_refit.py \
  --refit "$RUN/step4b_metric/refit_metric.json" \
  --out-dir "$RUN/step4c" \
  --frame-start 40 --frame-end 60 --anchor-frame 0 --fps 120 \
  --gt-poses "$SCENE/object_poses_obj0.csv,$SCENE/object_poses_obj1.csv"

# 5 — render (on modal in practice)
"$PY" scripts/collision/step5_render.py \
  --canonical objA/point_cloud.ply,objB/point_cloud.ply \
  --bg-ply background.ply \
  --predicted "$RUN/step4c/trajectory_predicted_obj0.csv,$RUN/step4c/trajectory_predicted_obj1.csv" \
  --ref-traj "$RUN/step4a/obj0/trajectory_smoothed.csv,$RUN/step4a/obj1/trajectory_smoothed.csv" \
  --dynerf-export runs/objA --cfg-args objA/cfg_args \
  --object-scale 0.05,0.05 --object-crop-radius 0.33,0.33 \
  --object-opacity-boost 3.0 --composite-mode alpha \
  --out "$RUN/step5"

# 6 — eval
"$PY" scripts/collision/step6_eval.py \
  --predicted "$RUN/step4c/trajectory_predicted_obj0.csv,$RUN/step4c/trajectory_predicted_obj1.csv" \
  --gt-poses "$SCENE/object_poses_obj0.csv,$SCENE/object_poses_obj1.csv" \
  --rendered "$RUN/step5/renders" --gt-rgb-root "$SCENE/rgb" \
  --out "$RUN/step6" --fps 120 \
  --extracted-traj "$RUN/step4a/obj0/trajectory_smoothed.csv,$RUN/step4a/obj1/trajectory_smoothed.csv" \
  --refit-metric "$RUN/step4b_metric/refit_metric.json"
```
