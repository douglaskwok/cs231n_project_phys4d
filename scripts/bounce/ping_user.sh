#!/usr/bin/env bash
# Audible + visual notify (foreground — do not background or macOS may kill afplay).
/usr/bin/osascript -e 'display notification "Agent needs you" with title "CS231N Pipeline"' 2>/dev/null || true
for _ in 1 2 3 4 5; do
  /usr/bin/afplay /System/Library/Sounds/Ping.aiff
done
