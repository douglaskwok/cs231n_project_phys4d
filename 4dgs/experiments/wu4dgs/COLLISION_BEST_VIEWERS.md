# Collision Best 4DGS Viewers

Last updated: 2026-06-05

This file records the current best known Wu 4DGS collision viewers for each scene. The browser links assume the repo root is being served with:

```bash
python -u -m http.server 8950 --directory .
```

Root URL:

```text
http://127.0.0.1:8950/
```

Notes:

- `best` means this is the version we currently prefer for viewing/handoff.
- `baseline` means usable but not final; we have known issues or pending retries.
- `pending` means a better run is training or expected, but not yet recorded here.
- `not present` means the separate object viewer is not currently built/downloaded locally.
- The 0000 successful baseline lives under the older `collision_scale1p75_v1p4_sidecams_60fps` root. The later 2x2x2 grid's `scene_0000_col_m220_asym140_r098` is the physics-matched anchor, but the already successful Wu viewers are in the older folder.

## Scene 0000

Source scene:

```text
dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room
```

| Item | Status | 12-view viewer | Local path | Notes |
| --- | --- | --- | --- | --- |
| 0000 object A | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_a_compactA_60k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_a_compactA_60k/index.html` | Object-specific A fit: `masks_object_a`, 50k A base, then compact-A continuation from 50k to 60k. |
| 0000 object B | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_compactA_60k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_compactA_60k/index.html` | Object-specific B fit: `masks_object_b`, 50k B base, then compact-A continuation from 50k to 60k. B also had anchor/compact-B variants; compact-A was selected. |
| 0000 combined | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_ab_compactA_60k_layered_thr48/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_ab_compactA_60k_layered_thr48/index.html` | Layered composite from the separate object-specific 60k continued A/B checkpoints, opacity threshold 0.48. |

0000 object B alternates to re-check visually:

- 50k baseline: [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_50k/index.html) - `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_50k/index.html`
- anchor 0.05, 55k: [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_anchor0p05_55k/index.html) - `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_anchor0p05_55k/index.html`
- anchor 0.5, 60k: [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_anchor0p5_60k/index.html) - `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_anchor0p5_60k/index.html`
- compact-B, 60k: [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_compactB_60k/index.html) - `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_compactB_60k/index.html`

## Scene 0001

Source scene:

```text
dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090
```

| Item | Status | 12-view viewer | Local path | Notes |
| --- | --- | --- | --- | --- |
| 0001 object A | baseline | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090/4dgs_wu/viewer_object_a_staged_50k10k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090/4dgs_wu/viewer_object_a_staged_50k10k/index.html` | Correct object A for the current combined reference: `object_a_compactA_50000_to60000`. Do not confuse with `viewer_object_a_compactA_60k`, which points to the separate `object_a_compactA_iter60000` run. |
| 0001 object B | baseline | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090/4dgs_wu/viewer_object_b_staged_50k10k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090/4dgs_wu/viewer_object_b_staged_50k10k/index.html` | Correct object B for the current combined reference: `object_b_compactA_50000_to60000`. Cam 10 has been weak/missing in some B variants. |
| 0001 combined | baseline | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090/4dgs_wu/viewer_ab_staged_50k10k_layered_thr48/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090/4dgs_wu/viewer_ab_staged_50k10k_layered_thr48/index.html` | Current combined reference; built from the staged `50000_to60000` object A/B renders, threshold 0.48. Scene still needs QA. |

## Scene 0002

Source scene:

```text
dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0002_col_m220_equal105_r098
```

| Item | Status | 12-view viewer | Local path | Notes |
| --- | --- | --- | --- | --- |
| 0002 object A | not present | - | - | Separate viewer is not currently present locally. |
| 0002 object B | not present | - | - | Separate viewer is not currently present locally. |
| 0002 combined | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0002_col_m220_equal105_r098/4dgs_wu/viewer_ab_compactA_60k_layered_thr48/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0002_col_m220_equal105_r098/4dgs_wu/viewer_ab_compactA_60k_layered_thr48/index.html` | This was one of the cleaner combined results. |

## Scene 0003

Source scene:

```text
dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0003_col_m220_equal105_r090
```

| Item | Status | 12-view viewer | Local path | Notes |
| --- | --- | --- | --- | --- |
| 0003 object A | not present | - | - | Separate viewer is not currently present locally. Combined uses object A 60k. |
| 0003 object B | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0003_col_m220_equal105_r090/4dgs_wu/viewer_object_b_anchor005_desaturn_80k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0003_col_m220_equal105_r090/4dgs_wu/viewer_object_b_anchor005_desaturn_80k/index.html` | Best object B after cleanup/desaturn continuation. |
| 0003 combined | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0003_col_m220_equal105_r090/4dgs_wu/viewer_ab_a60k_b80k_desaturn_layered_thr48/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0003_col_m220_equal105_r090/4dgs_wu/viewer_ab_a60k_b80k_desaturn_layered_thr48/index.html` | Combined object A 60k with object B 80k desaturn, layered threshold 0.48. |

## Scene 0004

Source scene:

```text
dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098
```

| Item | Status | 12-view viewer | Local path | Notes |
| --- | --- | --- | --- | --- |
| 0004 object A | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_object_a_cleanup_70k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_object_a_cleanup_70k/index.html` | 70k cleanup/compression continuation, promoted over the 60k redo. |
| 0004 object B | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_object_b_compactA_60k_redo/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_object_b_compactA_60k_redo/index.html` | Object B looked acceptable in separate viewer. |
| 0004 combined | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_ab_a70k_b60k_layered_thr48/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_ab_a70k_b60k_layered_thr48/index.html` | Layered same-coordinate composite from A 70k cleanup + B 60k, threshold 0.48. Do not use `viewer_ab_75k_joint_thr048` as best; it was diagnostic and looked poor. |

0004 object A variants to inspect:

- 60k redo baseline: [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_object_a_compactA_60k_redo/index.html) - `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_object_a_compactA_60k_redo/index.html`
- 70k cleanup continuation: [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_object_a_cleanup_70k/index.html) - `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098/4dgs_wu/viewer_object_a_cleanup_70k/index.html`

## Scene 0005

Source scene:

```text
dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0005_col_m440_asym140_r090
```

| Item | Status | 12-view viewer | Local path | Notes |
| --- | --- | --- | --- | --- |
| 0005 object A | not present | - | - | Separate viewer is not currently present locally. |
| 0005 object B | not present | - | - | Separate viewer is not currently present locally. |
| 0005 combined | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0005_col_m440_asym140_r090/4dgs_wu/viewer_ab_staged_50k10k_layered_thr48/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0005_col_m440_asym140_r090/4dgs_wu/viewer_ab_staged_50k10k_layered_thr48/index.html` | Current combined reference. |

## Scene 0006

Source scene:

```text
dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098
```

| Item | Status | 12-view viewer | Local path | Notes |
| --- | --- | --- | --- | --- |
| 0006 object A | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_object_a_compactA_60k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_object_a_compactA_60k/index.html` | Current object A pick. |
| 0006 object B | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_object_b_all30k_compactA_60k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_object_b_all30k_compactA_60k/index.html` | All-pose 30k-init compact-A 60k retry. Objective QA beat the original: fewer blank frames and lower foreground overgrowth. |
| 0006 combined | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_ab_a60k_bretry_all30k_60k_layered_thr48/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_ab_a60k_bretry_all30k_60k_layered_thr48/index.html` | Layered same-coordinate composite from A 60k + B all30k retry 60k, threshold 0.48. |

0006 object B variants:

- Superseded original compact-A 60k: [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_object_b_compactA_60k/index.html) - `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_object_b_compactA_60k/index.html`
- Selected best, all-pose 30k-init compact-A 60k retry: [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_object_b_all30k_compactA_60k/index.html) - `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098/4dgs_wu/viewer_object_b_all30k_compactA_60k/index.html`

## Scene 0007

Source scene:

```text
dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0007_col_m440_equal105_r090
```

| Item | Status | 12-view viewer | Local path | Notes |
| --- | --- | --- | --- | --- |
| 0007 object A | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0007_col_m440_equal105_r090/4dgs_wu/viewer_object_a_compactA_60k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0007_col_m440_equal105_r090/4dgs_wu/viewer_object_a_compactA_60k/index.html` | Current object A pick. |
| 0007 object B | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0007_col_m440_equal105_r090/4dgs_wu/viewer_object_b_compactA_60k/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0007_col_m440_equal105_r090/4dgs_wu/viewer_object_b_compactA_60k/index.html` | Current object B pick. |
| 0007 combined | best | [open](http://127.0.0.1:8950/dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0007_col_m440_equal105_r090/4dgs_wu/viewer_ab_compactA_60k_layered_thr48/index.html) | `dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0007_col_m440_equal105_r090/4dgs_wu/viewer_ab_compactA_60k_layered_thr48/index.html` | Layered same-coordinate composite from A/B 60k, threshold 0.48. |

## Pending Updates

- 0001 object B: cam-weighted/fresh experiments were tried; current baseline is still the 60k CompactA viewer unless a later retry clearly beats it.
- 0006 object B and combined: promoted to the all-pose 30k-init B retry version after objective QA and composite generation.
