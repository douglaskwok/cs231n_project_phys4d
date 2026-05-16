#!/usr/bin/env bash
# Copy batch to a non-iCloud path so every byte is on disk (then upload from there).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/outputs/sphere_bounce_batch"
DEST="${1:-${HOME}/phys4d_data/sphere_bounce_batch}"

if [[ ! -d "${SRC}" ]]; then
  echo "Missing ${SRC}" >&2
  exit 1
fi

echo "Copying ${SRC} -> ${DEST}"
echo "Use a path NOT under ~/Desktop or ~/Documents if iCloud sync is on."
mkdir -p "$(dirname "${DEST}")"
rm -rf "${DEST}"
# ditto preserves metadata; forces read from source (downloads iCloud files during copy)
ditto "${SRC}" "${DEST}"

echo "Verifying..."
PYTHONPATH="${REPO_ROOT}/src" python "${REPO_ROOT}/scripts/verify_batch_local.py" --batch-root "${DEST}"

echo ""
echo "Upload from local copy:"
echo "  modal volume rm phys4d-gs-data sphere_bounce_batch -r 2>/dev/null || true"
echo "  modal volume put phys4d-gs-data \"${DEST}\" sphere_bounce_batch"
echo ""
echo "Optional symlink (repo still on Desktop):"
echo "  mv \"${SRC}\" \"${SRC}.icloud_backup\""
echo "  ln -s \"${DEST}\" \"${SRC}\""
