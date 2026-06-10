#!/usr/bin/env bash
# Ping in foreground before Modal approval dialog (background afplay was getting killed).
cat >/dev/null
ROOT="${CURSOR_PROJECT_DIR:-$(pwd)}"
if [[ -x "$ROOT/scripts/bounce/ping_user.sh" ]]; then
  bash "$ROOT/scripts/bounce/ping_user.sh"
else
  /usr/bin/osascript -e 'display notification "Modal command pending" with title "Cursor"' 2>/dev/null || true
  for _ in 1 2 3 4 5; do /usr/bin/afplay /System/Library/Sounds/Ping.aiff; done
fi
printf '%s\n' '{"permission":"allow"}'
exit 0
