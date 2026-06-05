#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$DIR/run_workflow_r0p20_nativeaspect.sh"
bash "$DIR/run_workflow_ball12.sh" v1
bash "$DIR/run_workflow_ball12.sh" v2
bash "$DIR/ping_user.sh"
echo "All three workflows complete."
