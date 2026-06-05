#!/usr/bin/env bash
# Wait for an active screen to finish, then run a serial bounce prediction queue.
set -euo pipefail
cd "$(dirname "$0")/../.."

WAIT_SCREEN="${WAIT_SCREEN:-bounce_pred_serial_0006_0008}"
SCENES="${SCENES:-0004 0005 0000 0002}"
LOG="${LOG:-outputs/logs/bounce_pred_queue_after_0008.log}"
POLL_SECS="${POLL_SECS:-60}"

mkdir -p outputs/logs
echo "===== QUEUE WAITER $(date) =====" | tee -a "$LOG"
echo "waiting for screen: $WAIT_SCREEN" | tee -a "$LOG"
while true; do
  screen -ls > /tmp/phys4d_screen_list.txt 2>/dev/null || true
  grep -q "[.]$WAIT_SCREEN" /tmp/phys4d_screen_list.txt || break
  sleep "$POLL_SECS"
done

echo "===== QUEUE START $(date): $SCENES =====" | tee -a "$LOG"
SCENES="$SCENES" LOG="$LOG" bash scripts/bounce/run_workflow_wu75k_0006_0008_serial.sh
echo "===== QUEUE DONE $(date) =====" | tee -a "$LOG"
