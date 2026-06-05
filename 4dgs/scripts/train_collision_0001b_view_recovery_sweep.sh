#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENE="dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0001_col_m220_asym140_r090"
LOG_DIR="${LOG_DIR:-/tmp/wu4dgs_collision_logs}"
MODAL_PROFILE="${MODAL_PROFILE:-simon}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  "$SCENE"
  "__RUN_NAME__"
  --mask-subdir masks_object_b
  --mode object
  --background black
  --coarse-iterations 1000
  --time-resolution 120
  --bounds 1.2
  --drop-invisible-frames
  --min-visible-cameras 6
  --min-mask-pixels 50
  --init-points 30000
  --init-center-mode all
  --init-surface-ratio 0.8
  --wu-opacity-reset-interval 300000
  --trajectory-anchor-loss-weight 0.05
  --render
)

run_case() {
  local run_name="$1"
  shift
  local log="$LOG_DIR/${run_name}.log"
  echo "Launching $run_name -> $log"
  (
    export MODAL_PROFILE
    args=("${COMMON_ARGS[@]}")
    args[1]="$run_name"
    bash 4dgs/scripts/train_one_scene_wu4dgs.sh "${args[@]}" "$@"
  ) >"$log" 2>&1 &
}

# Continue the old 60k model, but explicitly give weak cameras more gradient
# and avoid the harsh cleanup that made cam06/cam10 worse.
run_case \
  wu_collision_elastic_s0001_object_b_vieww_c6x3_c10x5_60000_to70000 \
  --iterations 70000 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 0.5 \
  --area-loss-weight 0.05 \
  --scale-isotropy-loss-weight 0.03 \
  --camera-loss-weights 6:3,10:5 \
  --wu-densify-until-iter 0 \
  --wu-start-checkpoint wu4dgs_wu_collision_elastic_s0001_object_b_compactA_50000_to60000/chkpnt_fine_60000.pth

# Same starting point, more aggressive camera balancing, minimal pruning.
run_case \
  wu_collision_elastic_s0001_object_b_vieww_c6x4_c10x8_noprune_60000_to70000 \
  --iterations 70000 \
  --foreground-loss-weight 25 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 0 \
  --area-loss-weight 0 \
  --scale-isotropy-loss-weight 0.02 \
  --camera-loss-weights 6:4,10:8 \
  --wu-densify-until-iter 0 \
  --wu-start-checkpoint wu4dgs_wu_collision_elastic_s0001_object_b_compactA_50000_to60000/chkpnt_fine_60000.pth

# Redo the compact pass from the 50k base checkpoint, because cam10 may already
# be unrecoverable by 60k in the old fit.
run_case \
  wu_collision_elastic_s0001_object_b_vieww_from50k_c6x3_c10x6_50000_to60000 \
  --iterations 60000 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 0.5 \
  --area-loss-weight 0.05 \
  --scale-isotropy-loss-weight 0.03 \
  --camera-loss-weights 6:3,10:6 \
  --wu-densify-until-iter 1000 \
  --wu-start-checkpoint wu4dgs_wu_collision_elastic_s0001_object_b_base50000/chkpnt_fine_50000.pth

wait
