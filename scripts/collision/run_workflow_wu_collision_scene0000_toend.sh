#!/usr/bin/env bash
# Long-horizon collision 0000 prediction variant.
# Baseline event split is 112..131; this keeps the same start/cameras and
# extends prediction/rendering through the last frame, 156.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MODAL_PROFILE="${MODAL_PROFILE:-simon}"
export MODAL="${MODAL:-arch -arm64 modal}"
export RUN="${RUN:-outputs/collision_pipeline/scene_0000_collision_room_toend}"
export MBASE="${MBASE:-wu_collision_scene0000_toend_20260605}"
export TRAIN_START="${TRAIN_START:-0}"
export TRAIN_END="${TRAIN_END:-111}"
export TEST_START="${TEST_START:-112}"
export TEST_END="${TEST_END:-156}"
export MODEL_LAST="${MODEL_LAST:-156}"

echo "START $(date)"
echo "RUN=$RUN"
echo "MBASE=$MBASE"
echo "split train=${TRAIN_START}-${TRAIN_END} test=${TEST_START}-${TEST_END} model_last=${MODEL_LAST}"

bash scripts/collision/run_workflow_wu_collision_scene0000.sh
