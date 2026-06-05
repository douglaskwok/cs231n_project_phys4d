#!/usr/bin/env bash
# Launch the e=0.97 ball-bounce prediction workflows in parallel.
#
# Logs:
#   outputs/logs/bounce_pred_0006.log
#   outputs/logs/bounce_pred_0007.log
#   outputs/logs/bounce_pred_0008.log
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p outputs/logs
MODAL_FOR_CHILDREN="${MODAL:-modal}"
PY_FOR_CHILDREN="${PY:-.venv_pipeline/bin/python}"

for scene_id in 0006 0007 0008; do
  screen_name="bounce_pred_${scene_id}"
  log_path="outputs/logs/${screen_name}.log"
  if screen -list | rg -q "[.]${screen_name}[[:space:]]"; then
    echo "screen $screen_name already exists; skipping"
    continue
  fi
  echo "starting $screen_name -> $log_path"
  screen -dmS "$screen_name" bash -lc "
    cd '$PWD'
    export PY='$PY_FOR_CHILDREN'
    export MODAL='$MODAL_FOR_CHILDREN'
    bash scripts/bounce/run_workflow_wu75k_grid_scene.sh '$scene_id' 2>&1 | tee '$log_path'
  "
done

screen -ls || true
echo "Attach with: screen -r bounce_pred_0006   (or 0007 / 0008)"
