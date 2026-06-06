# PyBullet Data Generation Hyperparameters

Last updated: 2026-06-06

This note traces the currently selected Wu 4DGS bounce and collision viewers back to the PyBullet scene-generation settings. It is intended as a compact source table for the paper's data-generation appendix.

Primary viewer trackers:

- Bounce: `4dgs/experiments/wu4dgs/BOUNCE_BEST_VIEWER.md`
- Collision: `4dgs/experiments/wu4dgs/COLLISION_BEST_VIEWERS.md`

Primary PyBullet generators:

- Bounce scenes: `dataset/export_ping_pong_12view.py`
- Collision variant grid: `dataset/generate_collision_variants_final.py`
- Collision simulator implementation: `dataset/export_room_physics_12view.py`
- Older single-collision wrapper: `dataset/generate_collision_final.py`

For the current Wu ball-bounce data, use `dataset/export_ping_pong_12view.py` as the source of truth. The older `scripts/generate_sphere_bounce_dataset.py` is a separate M2/sphere-bounce experiment and is not the generator for the best-viewer 12-view Wu bounce grid. The current grid also does not exactly match the old `FULL_RESTITUTIONS` defaults in `export_ping_pong_12view.py`; it was generated with the Wu-specific settings recorded in the scene metadata and experiment log: `e in {0.89, 0.93, 0.97}` plus an extra `0.85` row, ball radius `0.20 m`, table top `z=0.40 m`, drop height `0.80 m`, `120 fps`, and `2.0 s`.

## Key Hyperparameters

| Category | Bounce scenes | Collision scenes | Produced by / traced to |
| --- | --- | --- | --- |
| Scenario type | Single rigid sphere bouncing on table | Two rigid boxes sliding/colliding on table with low rails | Bounce: `dataset/export_ping_pong_12view.py` `_simulate_variation_scene`; Collision: `dataset/generate_collision_variants_final.py` calling `simulate_scenario` in `dataset/export_room_physics_12view.py` |
| Selected dataset root | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s` | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps`; scene 0000 viewer uses physics-matched older root `collision_scale1p75_v1p4_sidecams_60fps` | Best-viewer docs plus each dataset's `scenario_manifest.json` |
| Parameter grid | Restitution `e in {0.89, 0.93, 0.97}` crossed with ball angle `{-5, 0, +5}` deg; extra row `e=0.85` for scenes 0009-0011 | Object A mass `{0.22, 0.44}` kg crossed with velocity mode `{asym1.4, equal1.05}` and restitution `{0.98, 0.90}` | Bounce manifest and `BOUNCE_BEST_VIEWER.md`; Collision `generate_collision_variants_final.py` defaults |
| Render FPS | `120 fps` | `60 fps` | Dataset manifests and generator CLI defaults/arguments |
| Internal simulation rate | `480 Hz` | `480 Hz` | Bounce config `internal_dt_s=1/480`; Collision manifest `sim_hz=480` |
| Steps per rendered frame | `4` | `8` | `_steps_per_frame(video_fps, sim_hz)` |
| Duration / frames | `2.0 s`, `241` frames | `2.6 s`, `157` frames | Per-scene `config.json`; collision default duration in `SCENARIO_DEFAULTS` |
| Gravity | `[0, 0, -9.80665] m/s^2` | `[0, 0, -9.80665] m/s^2` | Shared `GRAVITY` constant imported from `export_ping_pong_12view.py` |
| PyBullet solver | `numSolverIterations=150`, `numSubSteps=2`, `deterministicOverlappingPairs=1`, `restitutionVelocityThreshold=0.0` | `numSolverIterations=180`, `numSubSteps=2`, `deterministicOverlappingPairs=1`, `restitutionVelocityThreshold=0.0` | Bounce `p.setPhysicsEngineParameter`; Collision `_init_pybullet(... solver_iterations=180)` |
| Contact threshold | `contactProcessingThreshold=0.0` for ball/table | `contactProcessingThreshold=0.0` for table, boxes, rails | `changeDynamics` calls in generators |
| Geometry | Sphere radius `0.20 m`; table `5.20 x 5.20 x 0.06 m`; table top `z=0.40 m` | Base box half-extents `[0.120, 0.120, 0.075] m`, scaled by `1.75` to half-extents `[0.21, 0.21, 0.13125] m`; full box size `0.42 x 0.42 x 0.2625 m`; table `5.20 x 5.20 x 0.06 m`; table top `z=0.25 m`; low rails enabled | Bounce per-scene config and manifest; Collision manifest and `_setup_collision_scene` |
| Collision rail geometry | - | Rail bounds use `wall_half_x=1.05 m`, `wall_half_y=0.35 m`, half-thickness `0.02625 m`, half-height `0.035 m`; full rail height is `0.070 m`, with rail center at `z=0.285 m` and rail top at `z=0.320 m`. | Collision config `scenario_params`; `_add_collision_walls` in `dataset/export_room_physics_12view.py` |
| Mass | Ball `0.0027 kg` | Object A `{0.22, 0.44} kg`; object B `0.160 kg` | Bounce `BALL_MASS`; Collision grid and fixed params |
| Restitution | Ball and table both use scene `e` | Boxes and rails use scene `e`; table uses `0.0` | Bounce `changeDynamics(table_id/ball_id, restitution=restitution)`; Collision `_setup_collision_scene` |
| Friction | Ball lateral `0.03`, rolling `0.0`, spinning `0.0`; table lateral `0.03` | PyBullet object/rail/table lateral friction `0.002`; object rolling/spinning `0.0` | Bounce constants; Collision `changeDynamics` calls. Note: generated collision configs before 2026-06-06 may have stale `scenario_params.object_lateral_friction=0.02`, but the actual PyBullet calls and root manifest use `0.002`. |
| Initial placement | Ball starts at `x=0.35` for `-5 deg`, `x=-0.35` for `+5 deg`, `x=0` for `0 deg`; `y=0.20`; `z=table_top + 0.80 = 1.20` | Object A starts at `x=-0.525`; object B starts at `x=+0.525`; both at `z=0.38125`. Initial face-to-face gap is `0.63 m` because center separation is `1.05 m` and each box half-width is `0.21 m`. | Bounce `_start_xy_for_ball_angle` and config; Collision `start_x=0.30*geometry_scale`, `geometry_scale=1.75` |
| Initial velocity | Horizontal speed derived from first-impact angle: `sqrt(2*g*(drop_height-radius))*tan(angle)`, signed along x; zero angular velocity | `asym1.4`: A `+0.98 m/s`, B `-0.77 m/s`, closing speed `1.75 m/s`; `equal1.05`: A `+1.05 m/s`, B `-1.05 m/s`, closing speed `2.10 m/s` | Bounce `_initial_velocity_for_ball_angle`; Collision `_parse_velocity_modes` and `_setup_collision_scene` |
| Environment | Room with floor/walls and six visual anchor markers; no ceiling | Same room, plus collision rails | `_create_room_geometry`; collision `_add_collision_walls` |
| Camera rig | 12 named views, `960 x 544`; ring radius `1.75`, side/top/low cameras; train cams 0-9, held-out cams 10-11 | Same 12 named views, but left/right side cameras are moved outward by `0.3 m` and upward by `0.1 m` | Bounce `_default_camera_rows`; Collision `_default_camera_rows(side_camera_extra_radius=0.3, side_camera_extra_height=0.1)` |
| Camera FOV | Ring `48 deg`; top `42`; top-oblique `46`; low views `58`; target z `0.95` | Same FOVs; target `[0, 0, 0.60]` | Camera row definitions in both generators |
| Masks / GT outputs | RGB, ball masks, table masks, `cameras.json`, `object_poses.csv`, `physics_params.json`, videos | RGB, union object masks, per-object masks, table/object-table masks, `cameras.json`, `object_poses.csv`, videos | Per-scene `config.json` outputs; render loops use PyBullet segmentation ids |
| Train/test temporal split in config | Train frames `0-119`; test frames `120-179`; full sequence has 241 frames | Full timeline marked train/test `0-156`; short split stored as train `0-59`, test `60-89` | `_frame_split_ranges`; collision `simulate_scenario` stores both full and short split |

## Per-Scene Parameter Grids

### Bounce

| Scene(s) | Restitution | Angle deg | Notes |
| --- | ---: | ---: | --- |
| 0000-0008 | `{0.89, 0.93, 0.97}` | `{-5, 0, +5}` | Main 3x3 grid used in the best-viewer tracker. |
| 0009-0011 | `0.85` | `{-5, 0, +5}` | Extra row, same generation settings, outside the main 3x3 tracker. |

The selected center viewer in `BOUNCE_BEST_VIEWER.md` points to the older folder `wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0`, but it is physics-matched to grid scene `0004_e0p93_a0p0` for the parameters above.

### Collision

| Scene | Object A mass kg | Velocity mode | Object A speed m/s | Object B speed m/s | Closing speed m/s | Restitution |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0000 | 0.22 | asym1.4 | 0.98 | 0.77 | 1.75 | 0.98 |
| 0001 | 0.22 | asym1.4 | 0.98 | 0.77 | 1.75 | 0.90 |
| 0002 | 0.22 | equal1.05 | 1.05 | 1.05 | 2.10 | 0.98 |
| 0003 | 0.22 | equal1.05 | 1.05 | 1.05 | 2.10 | 0.90 |
| 0004 | 0.44 | asym1.4 | 0.98 | 0.77 | 1.75 | 0.98 |
| 0005 | 0.44 | asym1.4 | 0.98 | 0.77 | 1.75 | 0.90 |
| 0006 | 0.44 | equal1.05 | 1.05 | 1.05 | 2.10 | 0.98 |
| 0007 | 0.44 | equal1.05 | 1.05 | 1.05 | 2.10 | 0.90 |

Shared collision fixed parameters for all eight scenes: object geometry scale `1.75`, box half-extents `[0.21, 0.21, 0.13125] m`, box full size `0.42 x 0.42 x 0.2625 m`, initial face-to-face gap `0.63 m`, rail half-extents as above, table top `z=0.25 m`, `60 fps`, `480 Hz`, `2.6 s`, all 12 cameras, left/right side cameras offset by `+0.3 m` radius and `+0.1 m` height.

Scene 0000 in `COLLISION_BEST_VIEWERS.md` uses the older successful `collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room` viewer, but the doc notes it is the physics-matched anchor for `scene_0000_col_m220_asym140_r098` in the 2x2x2 grid.

## Source Trace

- Bounce best doc: `4dgs/experiments/wu4dgs/BOUNCE_BEST_VIEWER.md`
- Bounce per-scene metadata source: `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_am5p0/metadata.json` records `source = dataset/export_ping_pong_12view.py`.
- Bounce center-anchor metadata source: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/metadata.json` also records `source = dataset/export_ping_pong_12view.py`.
- Collision best doc: `4dgs/experiments/wu4dgs/COLLISION_BEST_VIEWERS.md`
- Bounce PyBullet constants: `dataset/export_ping_pong_12view.py` lines 23-68.
- Bounce PyBullet engine setup and dynamics: `dataset/export_ping_pong_12view.py` lines 810-913.
- Bounce config writing: `dataset/export_ping_pong_12view.py` lines 1071-1144.
- Collision grid generation: `dataset/generate_collision_variants_final.py` lines 68-132 and 252-316.
- Older single-collision wrapper: `dataset/generate_collision_final.py` lines 51-151; useful for provenance of the older scene-0000 viewer root, but not the full 8-scene grid.
- Collision engine setup/camera rig: `dataset/export_room_physics_12view.py` lines 57-179.
- Collision object, rail, and initial-velocity setup: `dataset/export_room_physics_12view.py` lines 276-438.
