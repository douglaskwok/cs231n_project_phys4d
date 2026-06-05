#!/usr/bin/env bash
# Continue the two successful scene_0003 object-B canaries from 60k to 70k.
# This is a cleanup pass aimed at reducing halo while preserving the fixed
# non-blinking behavior from the full-trajectory 30k-point init canaries.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODAL_PROFILE="${MODAL_PROFILE:-simon}"
SCENE_DIR="${SCENE_DIR:-dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0003_col_m220_equal105_r090}"
LOG_DIR="${LOG_DIR:-/tmp/wu4dgs_collision_logs}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --mask-subdir masks_object_b
  --mode object
  --background black
  --iterations 70000
  --coarse-iterations 1000
  --time-resolution 120
  --bounds 1.2
  --foreground-loss-weight 25
  --mask-loss-weight 1
  --bg-spill-loss-weight 3
  --area-loss-weight 0.3
  --scale-isotropy-loss-weight 0.05
  --wu-densify-until-iter 0
  --wu-opacity-reset-interval 300000
  --drop-invisible-frames
  --min-visible-cameras 6
  --min-mask-pixels 50
  --init-points 30000
  --init-center-mode all
  --init-surface-ratio 0.8
  --render
)

run_cleanup() {
  local label="$1"
  local anchor="$2"
  local start_checkpoint="wu4dgs_wu_collision_elastic_s0003_object_b_${label}_50000_to60000/chkpnt_fine_60000.pth"
  local run_name="wu_collision_elastic_s0003_object_b_${label}_cleanup_60000_to70000"

  echo "==> Cleaning up ${label}"
  MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
    "$SCENE_DIR" \
    "$run_name" \
    "${COMMON_ARGS[@]}" \
    ${anchor:+--trajectory-anchor-loss-weight "$anchor"} \
    --wu-start-checkpoint "$start_checkpoint"
}

run_cleanup \
  all30k_iso02_den1k \
  "" \
  >"$LOG_DIR/collision_s0003_object_b_all30k_iso02_den1k_cleanup_60000_to70000.log" 2>&1 &
pid_plain="$!"

run_cleanup \
  all30k_iso02_den1k_anchor005 \
  0.05 \
  >"$LOG_DIR/collision_s0003_object_b_all30k_iso02_den1k_anchor005_cleanup_60000_to70000.log" 2>&1 &
pid_anchor="$!"

status=0
wait "$pid_plain" || status=1
wait "$pid_anchor" || status=1
exit "$status"
