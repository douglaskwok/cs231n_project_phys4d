#!/usr/bin/env bash
# Serial prediction workflow for the extra e=0.85 ball-bounce row.
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p outputs/logs
LOG="${LOG:-outputs/logs/bounce_pred_serial_0009_0011.log}"
if [[ "${NO_TEE:-0}" != "1" && -z "${BOUNCE_E85_SERIAL_TEE:-}" ]]; then
  export BOUNCE_E85_SERIAL_TEE=1
  exec > >(tee -a "$LOG") 2>&1
fi

echo "===== SERIAL E85 LOG $(date) ====="
export MODAL_PROFILE="${MODAL_PROFILE:-simon}"
export MODAL="${MODAL:-arch -arm64 modal}"

SCENES="${SCENES:-0009 0010 0011}"
for scene_id in $SCENES; do
  echo "===== SERIAL START $scene_id $(date) ====="
  bash scripts/bounce/run_workflow_wu75k_grid_scene.sh "$scene_id"
  echo "===== SERIAL DONE $scene_id $(date) ====="
done
