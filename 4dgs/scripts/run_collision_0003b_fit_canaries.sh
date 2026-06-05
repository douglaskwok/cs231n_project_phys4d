#!/usr/bin/env bash
# Run a small parallel canary search for scene 0003 object B.
# Goal: reduce object-B temporal black-frame collapse without touching object A.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENE_DIR="${SCENE_DIR:-dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0003_col_m220_equal105_r090}"
MODAL_PROFILE="${MODAL_PROFILE:-simon}"
LOG_DIR="${LOG_DIR:-/tmp/wu4dgs_collision_logs}"
BASE_ITERATIONS="${BASE_ITERATIONS:-50000}"
COMPACT_ITERATIONS="${COMPACT_ITERATIONS:-60000}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --mask-subdir masks_object_b
  --mode object
  --background black
  --coarse-iterations 1000
  --time-resolution 120
  --bounds 1.2
  --foreground-loss-weight 25
  --mask-loss-weight 1
  --bg-spill-loss-weight 1
  --area-loss-weight 0.1
  --drop-invisible-frames
  --min-visible-cameras 6
  --min-mask-pixels 50
  --init-points 30000
  --init-center-mode all
  --init-surface-ratio 0.8
)

run_variant() {
  local label="$1"
  local densify="$2"
  local scale_iso="$3"
  local anchor="$4"

  local base_run="wu_collision_elastic_s0003_object_b_${label}_base${BASE_ITERATIONS}"
  local compact_run="wu_collision_elastic_s0003_object_b_${label}_${BASE_ITERATIONS}_to${COMPACT_ITERATIONS}"
  local base_ckpt="wu4dgs_${base_run}/chkpnt_fine_${BASE_ITERATIONS}.pth"
  local log="$LOG_DIR/collision_s0003_object_b_${label}_${BASE_ITERATIONS}_to${COMPACT_ITERATIONS}.log"

  {
    echo "==> Variant ${label}"
    echo "base_run=${base_run}"
    echo "compact_run=${compact_run}"

    MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$SCENE_DIR" \
      "$base_run" \
      --iterations "$BASE_ITERATIONS" \
      "${COMMON_ARGS[@]}" \
      --wu-densify-until-iter "$densify" \
      --wu-opacity-reset-interval 300000 \
      --scale-isotropy-loss-weight "$scale_iso" \
      ${anchor:+--trajectory-anchor-loss-weight "$anchor"}

    MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$SCENE_DIR" \
      "$compact_run" \
      --iterations "$COMPACT_ITERATIONS" \
      "${COMMON_ARGS[@]}" \
      --wu-densify-until-iter "$densify" \
      --wu-opacity-reset-interval 300000 \
      --bg-spill-loss-weight 2 \
      --area-loss-weight 0.2 \
      --scale-isotropy-loss-weight "$scale_iso" \
      ${anchor:+--trajectory-anchor-loss-weight "$anchor"} \
      --wu-start-checkpoint "$base_ckpt" \
      --render
  } >"$log" 2>&1
}

pids=()
labels=()

run_variant "all30k_iso02_den1k" 1000 0.02 "" &
pids+=("$!")
labels+=("all30k_iso02_den1k")

run_variant "all30k_iso02_den8k" 8000 0.02 "" &
pids+=("$!")
labels+=("all30k_iso02_den8k")

run_variant "all30k_iso02_den1k_anchor005" 1000 0.02 0.05 &
pids+=("$!")
labels+=("all30k_iso02_den1k_anchor005")

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Variant ${labels[$i]} failed; see $LOG_DIR/collision_s0003_object_b_${labels[$i]}_${BASE_ITERATIONS}_to${COMPACT_ITERATIONS}.log" >&2
    status=1
  fi
done

exit "$status"
