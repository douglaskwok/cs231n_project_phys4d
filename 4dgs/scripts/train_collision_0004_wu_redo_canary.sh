#!/usr/bin/env bash
# Redo scene_0004 object A/B using the canary recipe that fixed scene_0003 B:
# full-trajectory initialization, 30k init points, early densification only,
# and the same spatial + temporal segmentation gates.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODAL_PROFILE="${MODAL_PROFILE:-simon}"
SCENE_DIR="${SCENE_DIR:-dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0004_col_m440_asym140_r098}"
LOG_DIR="${LOG_DIR:-/tmp/wu4dgs_collision_logs}"
BASE_ITERATIONS="${BASE_ITERATIONS:-50000}"
COMPACT_ITERATIONS="${COMPACT_ITERATIONS:-60000}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --mode object
  --background black
  --coarse-iterations 1000
  --time-resolution 120
  --bounds 1.2
  --foreground-loss-weight 25
  --mask-loss-weight 1
  --bg-spill-loss-weight 1
  --area-loss-weight 0.1
  --scale-isotropy-loss-weight 0.02
  --wu-densify-until-iter 1000
  --wu-opacity-reset-interval 300000
  --drop-invisible-frames
  --min-visible-cameras 6
  --min-mask-pixels 50
  --init-points 30000
  --init-center-mode all
  --init-surface-ratio 0.8
)

run_object() {
  local object_tag="$1"
  local mask_subdir="$2"
  local base_run="wu_collision_elastic_s0004_${object_tag}_all30k_iso02_den1k_base${BASE_ITERATIONS}"
  local compact_run="wu_collision_elastic_s0004_${object_tag}_all30k_iso02_den1k_${BASE_ITERATIONS}_to${COMPACT_ITERATIONS}"
  local base_ckpt="wu4dgs_${base_run}/chkpnt_fine_${BASE_ITERATIONS}.pth"

  {
    echo "==> Redo scene_0004 ${object_tag}"
    echo "base_run=${base_run}"
    echo "compact_run=${compact_run}"

    MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$SCENE_DIR" \
      "$base_run" \
      --mask-subdir "$mask_subdir" \
      --iterations "$BASE_ITERATIONS" \
      "${COMMON_ARGS[@]}"

    MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$SCENE_DIR" \
      "$compact_run" \
      --mask-subdir "$mask_subdir" \
      --iterations "$COMPACT_ITERATIONS" \
      "${COMMON_ARGS[@]}" \
      --bg-spill-loss-weight 2 \
      --area-loss-weight 0.2 \
      --wu-start-checkpoint "$base_ckpt" \
      --render
  } >"$LOG_DIR/collision_s0004_${object_tag}_all30k_iso02_den1k_${BASE_ITERATIONS}_to${COMPACT_ITERATIONS}.log" 2>&1
}

run_object object_a masks_object_a &
pid_a="$!"

run_object object_b masks_object_b &
pid_b="$!"

status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
exit "$status"
