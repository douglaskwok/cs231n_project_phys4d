#!/usr/bin/env bash
# Serial rerun for e=0.97 prediction workflows.
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p outputs/logs
LOG="${LOG:-outputs/logs/bounce_pred_serial_0006_0008.log}"
if [[ "${NO_TEE:-0}" != "1" && -z "${BOUNCE_SERIAL_TEE:-}" ]]; then
  export BOUNCE_SERIAL_TEE=1
  exec > >(tee -a "$LOG") 2>&1
fi

echo "===== SERIAL LOG $(date) ====="
export MODAL_PROFILE="${MODAL_PROFILE:-simon}"
export MODAL="${MODAL:-arch -arm64 modal}"
SCENES="${SCENES:-0006 0008}"
for scene_id in $SCENES; do
  echo "===== SERIAL START $scene_id ====="
  bash scripts/bounce/run_workflow_wu75k_grid_scene.sh "$scene_id"
  echo "===== SERIAL DONE $scene_id ====="
done
