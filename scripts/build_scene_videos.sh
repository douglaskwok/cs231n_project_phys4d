#!/usr/bin/env bash
# Build high-quality MP4 previews from per-camera PNG frame sequences.
set -euo pipefail

SCENE_ROOT="${1:?usage: build_scene_videos.sh <scene_root>}"
FPS="${2:-60}"
CRF="${3:-18}"

SCENE_ROOT="$(cd "$SCENE_ROOT" && pwd)"
OUT_RGB="$SCENE_ROOT/videos/hq/rgb"
OUT_GRIDS="$SCENE_ROOT/videos/hq/grids"
mkdir -p "$OUT_RGB" "$OUT_GRIDS"

CAM_NAMES=(
  front front_right right back_right back
  back_left left front_left top top_oblique
  low_front_left low_back_right
)

encode_camera() {
  local cam_idx="$1"
  local cam_name="$2"
  local in_dir="$SCENE_ROOT/rgb/cam$(printf '%02d' "$cam_idx")"
  local out_path="$OUT_RGB/cam$(printf '%02d' "$cam_idx")_${cam_name}.mp4"
  if [[ ! -d "$in_dir" ]]; then
    echo "skip missing $in_dir" >&2
    return 0
  fi
  ffmpeg -y -loglevel error \
    -framerate "$FPS" \
    -start_number 0 \
    -i "$in_dir/frame%05d.png" \
    -frames:v "$(find "$in_dir" -name 'frame*.png' | wc -l | tr -d ' ')" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -crf "$CRF" \
    -preset slow \
    -movflags +faststart \
    "$out_path"
  echo "wrote $out_path"
}

build_grid() {
  local out_name="$1"
  shift
  local -a cams=("$@")
  local -a inputs=()
  local -a filter_parts=()
  local idx=0
  for cam in "${cams[@]}"; do
    local in_dir="$SCENE_ROOT/rgb/cam$(printf '%02d' "$cam")"
    inputs+=(-framerate "$FPS" -start_number 0 -i "$in_dir/frame%05d.png")
    filter_parts+=("[${idx}:v]")
    idx=$((idx + 1))
  done
  local n="${#cams[@]}"
  local cols=5
  if (( n <= 2 )); then
    cols="$n"
  elif (( n <= 4 )); then
    cols=2
  fi
  local rows=$(( (n + cols - 1) / cols ))
  local filter=""
  for part in "${filter_parts[@]}"; do
    filter+="${part}"
  done
  filter+="xstack=inputs=${n}:grid=${cols}x${rows},scale=trunc(iw/2)*2:trunc(ih/2)*2[v]"
  ffmpeg -y -loglevel error \
    "${inputs[@]}" \
    -filter_complex "$filter" \
    -map "[v]" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -crf "$CRF" \
    -preset slow \
    -movflags +faststart \
    "$OUT_GRIDS/${out_name}.mp4"
  echo "wrote $OUT_GRIDS/${out_name}.mp4"
}

for cam_idx in "${!CAM_NAMES[@]}"; do
  encode_camera "$cam_idx" "${CAM_NAMES[$cam_idx]}"
done

build_grid train_cams_grid 0 1 2 3 4 5 6 7 8 9
build_grid test_cams_grid 10 11
build_grid hero_cams_grid 0 4 8 9
