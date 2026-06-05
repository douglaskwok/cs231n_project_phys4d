#!/usr/bin/env bash
# Build cam10/cam11 mp4s from existing step5 render PNGs.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-.venv_pipeline/bin/python}"
RUN="${1:?usage: scripts/bounce/make_step5_mp4s.sh RUN_DIR [test_start] [test_end] [fps]}"
TEST_START="${2:-157}"
TEST_END="${3:-240}"
FPS="${4:-120}"

FFMPEG_BIN="${FFMPEG_BIN:-}"
if [[ -z "$FFMPEG_BIN" ]] && command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BIN="$(command -v ffmpeg)"
fi
if [[ -z "$FFMPEG_BIN" ]]; then
  set +e
  FFMPEG_BIN="$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null)"
  FFMPEG_STATUS=$?
  set -e
  if [[ "$FFMPEG_STATUS" -ne 0 ]]; then
    FFMPEG_BIN=""
  fi
fi

if [[ -z "$FFMPEG_BIN" ]]; then
  echo "ffmpeg not found. Install ffmpeg or imageio-ffmpeg first." >&2
  exit 1
fi

for cam in cam10 cam11; do
  mkdir -p "$RUN/step5/_mp4_frames"
  i=0
  for fr in $(seq "$TEST_START" "$TEST_END"); do
    src=$(printf '%04d_%s.png' "$fr" "$cam")
    dst=$(printf "$RUN/step5/_mp4_frames/frame%05d.png" "$i")
    if cp "$RUN/step5/renders/$src" "$dst" 2>/dev/null; then
      i=$((i+1))
    fi
  done

  if [[ "$i" -gt 0 ]]; then
    "$FFMPEG_BIN" -y -loglevel error -framerate "$FPS" -i "$RUN/step5/_mp4_frames/frame%05d.png" \
      -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
      "$RUN/step5/fused_scene_$cam.mp4"
    echo "wrote $RUN/step5/fused_scene_$cam.mp4 ($i frames)"
  else
    echo "mp4 skip $cam (no frames)"
  fi

  rm -rf "$RUN/step5/_mp4_frames"
done
