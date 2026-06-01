#!/usr/bin/env bash
# Run a small controlled Wu 4DGS ablation/variant batch for the promising
# 12-camera blue-marker ball setup. This is intentionally conservative:
# no frame stride, spatial masks, temporal visibility filtering, native aspect.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-phys_sim/bin/python}"
BASE_SCENE="dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0"
BASE_FRAME_LIST_2S="$BASE_SCENE/frame_list_first2s.txt"
BASE_FRAME_LIST_1P5S="$BASE_SCENE/frame_list_first1p5s.txt"
LOG_DIR="4dgs/experiments/wu4dgs/batch_logs"
BATCH_LOG="$LOG_DIR/wu_ball_promising_ablation_batch.log"
TRAIN_ATTEMPTS="${TRAIN_ATTEMPTS:-3}"

mkdir -p "$LOG_DIR"
touch "$BATCH_LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$BATCH_LOG"
}

ensure_frame_lists() {
  if [[ ! -f "$BASE_FRAME_LIST_2S" ]]; then
    seq 0 240 > "$BASE_FRAME_LIST_2S"
  fi
  if [[ ! -f "$BASE_FRAME_LIST_1P5S" ]]; then
    seq 0 180 > "$BASE_FRAME_LIST_1P5S"
  fi
}

qa_and_viewer() {
  local scene_dir="$1"
  local run_name="$2"
  local iter="$3"
  local dataset_dir="4dgs/experiments/wu4dgs/runs/$run_name"
  local model_dir="$scene_dir/4dgs_wu/$run_name/wu4dgs_$run_name"
  local render_dir="$model_dir/train/ours_${iter}/renders"
  local gt_dir="$model_dir/train/ours_${iter}/gt"
  local viewer_dir="$scene_dir/4dgs_wu/${run_name}_viewer"
  local qa_out="$scene_dir/4dgs_wu/${run_name}_qa.png"

  if [[ ! -d "$render_dir" ]]; then
    log "WARN missing render dir for $run_name: $render_dir"
    return 1
  fi

  log "QA $run_name"
  "$PYTHON_BIN" 4dgs/scripts/qa_wu_render.py \
    --renders "$render_dir" \
    --gt "$gt_dir" \
    --out "$qa_out" \
    --threshold 8 \
    --samples 12 | tee -a "$BATCH_LOG"

  log "Build viewer $run_name"
  "$PYTHON_BIN" 4dgs/scripts/view_4dgs_time.py build \
    --render-dir "$render_dir" \
    --dataset "$dataset_dir" \
    --out "$viewer_dir" | tee -a "$BATCH_LOG"
}

train_base_like() {
  local run_name="$1"
  local iterations="$2"
  local densify_until="$3"
  local frame_list="$4"
  shift 4

  local status=1
  local attempt
  for attempt in $(seq 1 "$TRAIN_ATTEMPTS"); do
    log "START train $run_name attempt=$attempt/$TRAIN_ATTEMPTS"
    bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$BASE_SCENE" \
      "$run_name" \
      --iterations "$iterations" \
      --coarse-iterations 1000 \
      --time-resolution 120 \
      --bounds 0.8 \
      --foreground-loss-weight 20 \
      --mask-loss-weight 1.0 \
      --bg-spill-loss-weight 1.0 \
      --area-loss-weight 0.10 \
      --scale-isotropy-loss-weight 0.05 \
      --wu-densify-until-iter "$densify_until" \
      --wu-opacity-reset-interval 300000 \
      --drop-invisible-frames \
      --min-visible-cameras 6 \
      --min-mask-pixels 50 \
      --frame-list "$frame_list" \
      --init-points 10000 \
      --init-center-mode first \
      --init-surface-ratio 0.85 \
      --render \
      "$@" 2>&1 | tee -a "$BATCH_LOG"
    status=${PIPESTATUS[0]}
    if [[ "$status" -eq 0 ]]; then
      break
    fi
    log "FAIL train $run_name attempt=$attempt status=$status"
    sleep 60
  done
  if [[ "$status" -ne 0 ]]; then
    log "GIVE UP train $run_name status=$status"
    return "$status"
  fi
  qa_and_viewer "$BASE_SCENE" "$run_name" "$iterations" || true
  log "DONE train $run_name"
}

generate_variant() {
  local out_dir="$1"
  local restitution="$2"
  local angle="$3"
  log "START generate variant out=$out_dir e=$restitution angle=$angle"
  "$PYTHON_BIN" dataset/export_ping_pong_12view.py \
    --variation-set single \
    --output-dir "$out_dir" \
    --duration-sec 2.0 \
    --video-fps 120 \
    --sim-hz 480 \
    --ball-radius-m 0.20 \
    --table-top-z 0.40 \
    --drop-height-above-surface-m 0.80 \
    --restitution "$restitution" \
    --ball-angle-deg "$angle" \
    --environment room \
    --ball-visual-style marker \
    --render-camera-names all \
    --video-camera-names all 2>&1 | tee -a "$BATCH_LOG"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    log "FAIL generate variant out=$out_dir status=$status"
    return "$status"
  fi
  log "DONE generate variant out=$out_dir"
}

train_variant() {
  local scene_dir="$1"
  local run_name="$2"
  local status=1
  local attempt
  for attempt in $(seq 1 "$TRAIN_ATTEMPTS"); do
    log "START train variant $run_name attempt=$attempt/$TRAIN_ATTEMPTS"
    bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$scene_dir" \
      "$run_name" \
      --iterations 30000 \
      --coarse-iterations 1000 \
      --time-resolution 120 \
      --bounds 0.8 \
      --foreground-loss-weight 20 \
      --mask-loss-weight 1.0 \
      --bg-spill-loss-weight 1.0 \
      --area-loss-weight 0.10 \
      --scale-isotropy-loss-weight 0.05 \
      --wu-densify-until-iter 4000 \
      --wu-opacity-reset-interval 300000 \
      --drop-invisible-frames \
      --min-visible-cameras 6 \
      --min-mask-pixels 50 \
      --init-points 10000 \
      --init-center-mode first \
      --init-surface-ratio 0.85 \
      --render 2>&1 | tee -a "$BATCH_LOG"
    status=${PIPESTATUS[0]}
    if [[ "$status" -eq 0 ]]; then
      break
    fi
    log "FAIL train variant $run_name attempt=$attempt status=$status"
    sleep 60
  done
  if [[ "$status" -ne 0 ]]; then
    log "GIVE UP train variant $run_name status=$status"
    return "$status"
  fi
  qa_and_viewer "$scene_dir" "$run_name" 30000 || true
  log "DONE train variant $run_name"
}

main() {
  ensure_frame_lists

  log "=== Wu ball promising ablation batch started ==="

  # 1. Same setup as current best, but 50k from scratch instead of resumed.
  train_base_like \
    "wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch" \
    50000 \
    4000 \
    "$BASE_FRAME_LIST_2S" || true

  # 2. Densification ablation: keep loss/data fixed, allow longer densification.
  train_base_like \
    "wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k" \
    30000 \
    6000 \
    "$BASE_FRAME_LIST_2S" || true

  # 3. Temporal difficulty ablation: same fit recipe, first 1.5 seconds only.
  train_base_like \
    "wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k" \
    30000 \
    4000 \
    "$BASE_FRAME_LIST_1P5S" || true

  # 4. Small physics variant: restitution -0.04 from current e=0.93.
  local var_e89_dir="dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s"
  generate_variant "$var_e89_dir" 0.89 0.0 || true
  train_variant \
    "$var_e89_dir/scene_0000_e0p89_a0p0" \
    "wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k" || true

  # 5. Small physics variant: +5 degree slanted trajectory at current e=0.93.
  local var_a5_dir="dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s"
  generate_variant "$var_a5_dir" 0.93 5.0 || true
  train_variant \
    "$var_a5_dir/scene_0000_e0p93_a5p0" \
    "wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k" || true

  log "=== Wu ball promising ablation batch finished ==="
}

main "$@"
