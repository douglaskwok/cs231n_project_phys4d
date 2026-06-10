#!/usr/bin/env bash
# Backup: play ping on agent stop if the agent set .ping_needed
cat >/dev/null
ROOT="${CURSOR_PROJECT_DIR:-$(pwd)}"
FLAG="$ROOT/outputs/bounce_pipeline/.ping_needed"
if [[ -f "$FLAG" ]]; then
  rm -f "$FLAG"
  if [[ -x "$ROOT/scripts/bounce/ping_user.sh" ]]; then
    bash "$ROOT/scripts/bounce/ping_user.sh"
  fi
fi
exit 0
