#!/usr/bin/env bash
# Train Wu 4DGS per-object models for the tuned collision 2x2x2 dataset.
#
# Default order follows the requested sequence:
#   scene 1, 2, 4, 3, 5, 6, 7
#
# Each scene trains object_a then object_b with the compact-A collision recipe.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASET_ROOT="${DATASET_ROOT:-dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_2x2x2_tuned_60fps}"
MODAL_PROFILE="${MODAL_PROFILE:-simon}"
ITERATIONS="${ITERATIONS:-60000}"
SCENE_ORDER="${SCENE_ORDER:-1 2 4 3 5 6 7}"
OBJECTS="${OBJECTS:-a b}"

COMMON_ARGS=(
  --mode object
  --background black
  --iterations "$ITERATIONS"
  --coarse-iterations 1000
  --time-resolution 120
  --bounds 1.2
  --foreground-loss-weight 20
  --mask-loss-weight 1
  --bg-spill-loss-weight 2
  --area-loss-weight 0.2
  --scale-isotropy-loss-weight 0.05
  --wu-densify-until-iter 1000
  --wu-opacity-reset-interval 300000
  --drop-invisible-frames
  --min-visible-cameras 6
  --min-mask-pixels 50
  --init-points 20000
  --init-center-mode first
  --init-surface-ratio 0.8
  --render
)

for scene_idx in $SCENE_ORDER; do
  scene_dir="$(find "$DATASET_ROOT" -maxdepth 1 -type d -name "$(printf 'scene_%04d_*' "$scene_idx")" | sort | head -n 1)"
  if [[ -z "$scene_dir" ]]; then
    echo "Missing scene index $scene_idx under $DATASET_ROOT" >&2
    exit 1
  fi

  scene_tag="$(basename "$scene_dir" | sed 's/^scene_//')"
  for object in $OBJECTS; do
    case "$object" in
      a|object_a)
        mask_subdir="masks_object_a"
        object_tag="object_a"
        ;;
      b|object_b)
        mask_subdir="masks_object_b"
        object_tag="object_b"
        ;;
      *)
        echo "Unknown object '$object'; use a/b or object_a/object_b." >&2
        exit 1
        ;;
    esac

    run_name="wu_collision_tuned_s${scene_idx}_${object_tag}_compactA_iter${ITERATIONS}"
    echo
    echo "==> Training $run_name"
    echo "Scene: $scene_dir"
    MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$scene_dir" \
      "$run_name" \
      --mask-subdir "$mask_subdir" \
      "${COMMON_ARGS[@]}"
  done
done
