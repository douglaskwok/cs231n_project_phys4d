#!/usr/bin/env bash
# Render, download, and build 12-view Wu 4DGS HTML viewers for the extra
# e=0.85 ball-bounce row.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv_pipeline/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

MODAL_PROFILE="${MODAL_PROFILE:-simon}"
MODAL_BIN="${MODAL_BIN:-arch -arm64 modal}"
VOLUME="${VOLUME:-phys4d-gs-output}"
ITERATION="${ITERATION:-75000}"
TIME_RESOLUTION="${TIME_RESOLUTION:-120}"
BOUNDS="${BOUNDS:-0.8}"

DATA_ROOT="dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s"
RUNS_ROOT="4dgs/experiments/wu4dgs/runs"

run_one() {
  local scene_id="$1"
  local scene_dir="$2"
  local run_name="$3"

  local model_rel="wu4dgs_${run_name}"
  local wu_scene_rel="4d_scene_${run_name}"
  local local_run_dir="${scene_dir}/4dgs_wu/${run_name}"
  local model_dir="${local_run_dir}/${model_rel}"
  local render_dir="${model_dir}/train/ours_${ITERATION}/renders"
  local viewer_dir="${scene_dir}/4dgs_wu/${run_name}_viewer"
  local dataset_dir="${RUNS_ROOT}/${run_name}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] render scene ${scene_id}: ${run_name}"
  MODAL_PROFILE="$MODAL_PROFILE" $MODAL_BIN run modal_app.py \
    --render-wu-4d \
    --render-wu-4d-model "$model_rel" \
    --wu-scene-rel "$wu_scene_rel" \
    --render-wu-4d-iteration "$ITERATION" \
    --wu-time-resolution "$TIME_RESOLUTION" \
    --wu-bounds "$BOUNDS"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] download scene ${scene_id}: ${model_rel}/train/ours_${ITERATION}"
  mkdir -p "${model_dir}/train"
  MODAL_PROFILE="$MODAL_PROFILE" $MODAL_BIN volume get \
    "$VOLUME" \
    "${model_rel}/train/ours_${ITERATION}" \
    "${model_dir}/train" \
    --force

  if [[ ! -d "$render_dir" ]]; then
    echo "ERROR missing render dir for ${scene_id}: ${render_dir}" >&2
    return 1
  fi
  if [[ ! -d "$dataset_dir" ]]; then
    echo "ERROR missing dataset dir for ${scene_id}: ${dataset_dir}" >&2
    return 1
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] build viewer scene ${scene_id}: ${viewer_dir}"
  "$PYTHON_BIN" 4dgs/scripts/view_4dgs_time.py build \
    --render-dir "$render_dir" \
    --dataset "$dataset_dir" \
    --out "$viewer_dir" \
    --copy-renders

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] done scene ${scene_id}: ${viewer_dir}/index.html"
}

run_one "0009" \
  "${DATA_ROOT}/scene_0009_e0p85_am5p0" \
  "wu_ball12_2s_blue_e85_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"

run_one "0010" \
  "${DATA_ROOT}/scene_0010_e0p85_a0p0" \
  "wu_ball12_2s_blue_e85_a0_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"

run_one "0011" \
  "${DATA_ROOT}/scene_0011_e0p85_a5p0" \
  "wu_ball12_2s_blue_e85_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"
