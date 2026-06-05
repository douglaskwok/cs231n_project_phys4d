#!/usr/bin/env bash
# Continue the staged scene_0001 collision Wu 4DGS fits from 60k to 75k.
# Runs object A and object B in parallel, each resuming from the staged 60k
# checkpoint and rendering after training.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODAL_PROFILE="${MODAL_PROFILE:-simon}"
SCENE_DIR="${SCENE_DIR:-dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090}"
LOG_DIR="${LOG_DIR:-/tmp/wu4dgs_collision_logs}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --mode object
  --background black
  --iterations 75000
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

train_object() {
  local object_tag="$1"
  local mask_subdir="$2"
  local start_checkpoint="$3"
  local run_name="wu_collision_elastic_s0001_${object_tag}_compactA_60000_to75000"

  echo "==> Continuing $object_tag to 75k"
  MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
    "$SCENE_DIR" \
    "$run_name" \
    --mask-subdir "$mask_subdir" \
    "${COMMON_ARGS[@]}" \
    --wu-start-checkpoint "$start_checkpoint"
}

train_object \
  object_a \
  masks_object_a \
  wu4dgs_wu_collision_elastic_s0001_object_a_compactA_50000_to60000/chkpnt_fine_60000.pth \
  >"$LOG_DIR/collision_elastic_s0001_object_a_60000_to75000.log" 2>&1 &
pid_a="$!"

train_object \
  object_b \
  masks_object_b \
  wu4dgs_wu_collision_elastic_s0001_object_b_compactA_50000_to60000/chkpnt_fine_60000.pth \
  >"$LOG_DIR/collision_elastic_s0001_object_b_60000_to75000.log" 2>&1 &
pid_b="$!"

status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
exit "$status"
