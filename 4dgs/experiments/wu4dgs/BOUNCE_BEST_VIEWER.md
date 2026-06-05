# Bounce Best 4DGS Viewers

Last updated: 2026-06-05

This file records the current best known Wu 4DGS viewers for the ball-bounce 3x3 grid. The browser links assume the repo root is being served with:

```bash
python -u -m http.server 8950 --directory .
```

Root URL:

```text
http://127.0.0.1:8950/
```

Notes:

- `best` means this is the version we currently prefer for viewing/handoff.
- `rendered` means the 75k Wu checkpoint and native train renders are local, but the HTML viewer wrapper is not built locally yet.
- `training` means the Wu 4DGS run is active or expected, but the local checkpoint/render folder is not present yet.
- `pending` means the PyBullet scene exists, but the Wu 4DGS fit/viewer is not rendered locally yet.
- The core 3x3 grid is restitution `0.89, 0.93, 0.97` crossed with launch/ball angle `-5, 0, +5` degrees.
- The generated dataset also includes an extra `e=0.85` row in scenes `0009` to `0011`; those are not part of the main 9-scenario tracker below.
- The successful center setup was first trained under the older `0601_scene_0000_e0p93_a0p0` folder. It is physics-matched to the grid's `scene_0004_e0p93_a0p0`, so the center row below points to that existing best viewer until we explicitly rerender it under the 3x3 folder.

## Dataset

Main 3x3 dataset:

```text
dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s
```

Shared setup:

- Blue marker ball, radius `0.20m`
- Table top `z=0.40m`
- Drop height above surface `0.80m`
- 120 FPS, 2.0 seconds, all 12 cameras
- Main fitting recipe: `fg20`, `mask1`, `spill1`, `area0p1`, `scaleiso5e-2`, `densify_until=4000`
- Strong known center baseline: 50k scratch, then 75k from 50k; 100k exists but did not clearly improve visually.

## Current Best Grid

| Scene | Restitution | Angle | Status | 12-view viewer | Local path | Notes |
| --- | ---: | ---: | --- | --- | --- | --- |
| 0000 | 0.89 | -5 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_am5p0/4dgs_wu/wu_ball12_2s_blue_e89_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html) | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_am5p0/4dgs_wu/wu_ball12_2s_blue_e89_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html` | 75k checkpoint with local 12-view HTML viewer generated from native train renders. |
| 0001 | 0.89 | 0 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0001_e0p89_a0p0/4dgs_wu/wu_ball12_2s_blue_e89_a0_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html) | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0001_e0p89_a0p0/4dgs_wu/wu_ball12_2s_blue_e89_a0_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html` | 75k checkpoint with local 12-view HTML viewer generated from native train renders. |
| 0002 | 0.89 | 5 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0002_e0p89_a5p0/4dgs_wu/wu_ball12_2s_blue_e89_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html) | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0002_e0p89_a5p0/4dgs_wu/wu_ball12_2s_blue_e89_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html` | 75k checkpoint with local 12-view HTML viewer generated from native train renders. |
| 0003 | 0.93 | -5 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0003_e0p93_am5p0/4dgs_wu/wu_ball12_2s_blue_e93_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html) | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0003_e0p93_am5p0/4dgs_wu/wu_ball12_2s_blue_e93_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html` | 75k fit for the `e=0.93, angle=-5` variant. |
| 0004 | 0.93 | 0 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k_viewer/index.html) | `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k_viewer/index.html` | Physics-matched center anchor. Prefer 75k-from-50k visually; 100k exists but did not clearly improve. |
| 0005 | 0.93 | 5 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0005_e0p93_a5p0/4dgs_wu/wu_ball12_2s_blue_e93_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html) | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0005_e0p93_a5p0/4dgs_wu/wu_ball12_2s_blue_e93_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html` | 75k fit for the `e=0.93, angle=+5` variant. |
| 0006 | 0.97 | -5 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0006_e0p97_am5p0/4dgs_wu/wu_ball12_2s_blue_e97_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html) | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0006_e0p97_am5p0/4dgs_wu/wu_ball12_2s_blue_e97_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html` | 75k checkpoint with local 12-view HTML viewer generated from native train renders. |
| 0007 | 0.97 | 0 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0007_e0p97_a0p0/4dgs_wu/wu_ball12_2s_blue_e97_a0_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html) | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0007_e0p97_a0p0/4dgs_wu/wu_ball12_2s_blue_e97_a0_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html` | 75k checkpoint with local 12-view HTML viewer generated from native train renders. |
| 0008 | 0.97 | 5 | best | [open](http://127.0.0.1:8950/dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0008_e0p97_a5p0/4dgs_wu/wu_ball12_2s_blue_e97_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html) | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0008_e0p97_a5p0/4dgs_wu/wu_ball12_2s_blue_e97_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_viewer/index.html` | 75k checkpoint with local 12-view HTML viewer generated from native train renders. |

## Extra e=0.85 Row

These are outside the main 3x3 tracker, but use the same ball-bounce dataset root and the same 75k recipe.

| Scene | Restitution | Angle | Status | 12-view viewer | Local path | Notes |
| --- | ---: | ---: | --- | --- | --- | --- |
| 0009 | 0.85 | -5 | training | - | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0009_e0p85_am5p0` | Active attached Modal run in screen `wu_e85_0009`: `wu_ball12_2s_blue_e85_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k`, app `ap-gaWWL8nBL3bkffEGxQMhdB`. |
| 0010 | 0.85 | 0 | training | - | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0010_e0p85_a0p0` | Active attached Modal run in screen `wu_e85_0010`: `wu_ball12_2s_blue_e85_a0_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k`, app `ap-cs0RC80OSssVyyBgaZ0cF3`. |
| 0011 | 0.85 | 5 | training | - | `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scene_0011_e0p85_a5p0` | Active attached Modal run in screen `wu_e85_0011`: `wu_ball12_2s_blue_e85_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k`, app `ap-DyQ09VipCyABHhGEXDaYS6`. |

## Center Anchor Alternatives

These are useful for visual comparison on the `e=0.93, angle=0` center bounce:

- 50k scratch: [open](http://127.0.0.1:8950/dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch_viewer/index.html) - `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch_viewer/index.html`
- 75k from 50k: [open](http://127.0.0.1:8950/dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k_viewer/index.html) - `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k_viewer/index.html`
- 100k from 75k: [open](http://127.0.0.1:8950/dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k_viewer/index.html) - `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k_viewer/index.html`

## Pending Updates

- Finish/download/render the active e=0.85 Wu runs for `0009`, `0010`, and `0011`.
- QA the newly generated 12-view HTML wrappers for `0000`, `0002`, `0006`, `0007`, and `0008`.
- Decide whether to rerender the center `0004` under the actual 3x3 dataset folder, or keep using the equivalent `0601_scene_0000_e0p93_a0p0` anchor.
