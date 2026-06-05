#!/usr/bin/env bash
# Resume scenes 0006/0007 after their object-A 50k base fits completed but the
# staged script stopped before compact continuation because of a wrapper bug.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASET_ROOT="${DATASET_ROOT:-dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps}"
MODAL_PROFILE="${MODAL_PROFILE:-simon}"
SCENE_ORDER="${SCENE_ORDER:-6 7}"
LOG_DIR="${LOG_DIR:-/tmp/wu4dgs_collision_logs}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --mode object
  --background black
  --coarse-iterations 1000
  --time-resolution 120
  --bounds 1.2
  --foreground-loss-weight 20
  --mask-loss-weight 1
  --wu-densify-until-iter 1000
  --wu-opacity-reset-interval 300000
  --drop-invisible-frames
  --min-visible-cameras 6
  --min-mask-pixels 50
  --init-points 20000
  --init-center-mode first
  --init-surface-ratio 0.8
)

BASE_ARGS=(
  --bg-spill-loss-weight 1
  --area-loss-weight 0.1
  --scale-isotropy-loss-weight 0.05
)

COMPACT_ARGS=(
  --bg-spill-loss-weight 2
  --area-loss-weight 0.2
  --scale-isotropy-loss-weight 0.05
)

scene_dir_for() {
  local scene_idx="$1"
  find "$DATASET_ROOT" -maxdepth 1 -type d -name "$(printf 'scene_%04d_*' "$scene_idx")" | sort | head -n 1
}

train_base() {
  local scene_idx="$1"
  local scene_dir="$2"
  local object_tag="$3"
  local mask_subdir="$4"
  local run_name="wu_collision_elastic_s$(printf '%04d' "$scene_idx")_${object_tag}_base50000"
  MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
    "$scene_dir" \
    "$run_name" \
    --mask-subdir "$mask_subdir" \
    --iterations 50000 \
    "${COMMON_ARGS[@]}" \
    "${BASE_ARGS[@]}"
}

train_compact() {
  local scene_idx="$1"
  local scene_dir="$2"
  local object_tag="$3"
  local mask_subdir="$4"
  local base_run="wu_collision_elastic_s$(printf '%04d' "$scene_idx")_${object_tag}_base50000"
  local compact_run="wu_collision_elastic_s$(printf '%04d' "$scene_idx")_${object_tag}_compactA_50000_to60000"
  MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
    "$scene_dir" \
    "$compact_run" \
    --mask-subdir "$mask_subdir" \
    --iterations 60000 \
    "${COMMON_ARGS[@]}" \
    "${COMPACT_ARGS[@]}" \
    --wu-start-checkpoint "wu4dgs_${base_run}/chkpnt_fine_50000.pth" \
    --render
}

train_scene() {
  local scene_idx="$1"
  local scene_dir
  scene_dir="$(scene_dir_for "$scene_idx")"
  if [[ -z "$scene_dir" ]]; then
    echo "Missing scene index $scene_idx under $DATASET_ROOT" >&2
    return 1
  fi

  echo "==> Scene $scene_idx: $scene_dir"
  echo "==> Continue object A from existing base 50k"
  train_compact "$scene_idx" "$scene_dir" object_a masks_object_a

  echo "==> Train object B base 50k"
  train_base "$scene_idx" "$scene_dir" object_b masks_object_b

  echo "==> Continue object B compact 60k"
  train_compact "$scene_idx" "$scene_dir" object_b masks_object_b
}

pids=()
status=0
for scene_idx in $SCENE_ORDER; do
  log_path="$LOG_DIR/collision_elastic_s$(printf '%04d' "$scene_idx")_resume_after_a_base.log"
  echo "Launching resumed scene $scene_idx -> $log_path"
  train_scene "$scene_idx" >"$log_path" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
