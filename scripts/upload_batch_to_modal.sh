#!/usr/bin/env bash
# Upload sphere_bounce_batch without `modal run` (avoids 10–30 min image build on first deploy).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BATCH_REL="${1:-sphere_bounce_batch}"
BATCH_SRC="${REPO_ROOT}/outputs/${BATCH_REL}"

if [[ ! -d "${BATCH_SRC}" ]]; then
  echo "Missing ${BATCH_SRC}" >&2
  echo "Generate first: python scripts/generate_param_id_batch.py" >&2
  exit 1
fi

SIZE="$(du -sh "${BATCH_SRC}" | cut -f1)"
echo "Uploading ${BATCH_SRC} (${SIZE}) -> phys4d-gs-data:/${BATCH_REL}"
echo "(This uses modal volume put only — no GPU image build.)"

modal volume rm phys4d-gs-data "${BATCH_REL}" -r 2>/dev/null || true
modal volume put phys4d-gs-data "${BATCH_SRC}" "${BATCH_REL}"

MANIFEST="${BATCH_SRC}/dataset_manifest.json"
if [[ -f "${MANIFEST}" ]]; then
  modal volume put phys4d-gs-data "${MANIFEST}" "${BATCH_REL}/dataset_manifest.json"
fi

echo "Done. Next:"
echo "  modal run modal_app.py --extract-perception"
