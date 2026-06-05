#!/usr/bin/env bash
# Agent: run this as the LAST shell command before any turn that waits on the user.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/outputs/bounce_pipeline"
touch "$ROOT/outputs/bounce_pipeline/.ping_needed"
exec bash "$ROOT/scripts/bounce/ping_user.sh"
