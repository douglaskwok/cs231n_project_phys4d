#!/usr/bin/env bash
# Train Wu 4DGS per-object models for selected elastic collision scenes.
#
# Each selected scene runs in its own background pipeline. Within a scene,
# object_a is trained before object_b so logs and downloads stay organized.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASET_ROOT="${DATASET_ROOT:-dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps}"
MODAL_PROFILE="${MODAL_PROFILE:-simon}"
ITERATIONS="${ITERATIONS:-60000}"
SCENE_ORDER="${SCENE_ORDER:-1 2 4}"
OBJECTS="${OBJECTS:-a b}"
LOG_DIR="${LOG_DIR:-/tmp/wu4dgs_collision_logs}"

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

mkdir -p "$LOG_DIR"

train_scene() {
  local scene_idx="$1"
  local scene_dir
  scene_dir="$(find "$DATASET_ROOT" -maxdepth 1 -type d -name "$(printf 'scene_%04d_*' "$scene_idx")" | sort | head -n 1)"
  if [[ -z "$scene_dir" ]]; then
    echo "Missing scene index $scene_idx under $DATASET_ROOT" >&2
    return 1
  fi

  echo "==> Scene $scene_idx: $scene_dir"
  for object in $OBJECTS; do
    local mask_subdir object_tag
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
        return 1
        ;;
    esac

    local run_name="wu_collision_elastic_s$(printf '%04d' "$scene_idx")_${object_tag}_compactA_iter${ITERATIONS}"
    echo "==> Training $run_name"
    MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$scene_dir" \
      "$run_name" \
      --mask-subdir "$mask_subdir" \
      "${COMMON_ARGS[@]}"
  done
}

pids=()
for scene_idx in $SCENE_ORDER; do
  log_path="$LOG_DIR/collision_elastic_s$(printf '%04d' "$scene_idx")_iter${ITERATIONS}.log"
  echo "Launching scene $scene_idx -> $log_path"
  train_scene "$scene_idx" >"$log_path" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
