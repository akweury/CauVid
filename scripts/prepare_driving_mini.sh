#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
VIDEO_COUNT="${VIDEO_COUNT:-30}"
TARGET_FPS="${TARGET_FPS:-5}"
GPU_ID="${GPU_ID:-0}"
VIDEO_MODE="${VIDEO_MODE:-copy}"

RAW_ROOT="${RAW_ROOT:-$REPO_ROOT/dataset/bdd100k}"
VIDEO_DIR="${VIDEO_DIR:-$RAW_ROOT/bdd100k_videos_train_00/bdd100k/videos/train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/dataset/driving_mini}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Preparing driving_mini:"
echo "  raw dataset: $RAW_ROOT"
echo "  videos:      $VIDEO_DIR"
echo "  output:      $OUTPUT_ROOT"
echo "  video count: $VIDEO_COUNT"
echo "  target FPS:  $TARGET_FPS"
echo "  GPU:         $GPU_ID"

"$PYTHON_BIN" -m src.exp_driving_videos.legacy.prepare_driving_dataset \
    --raw-root "$RAW_ROOT" \
    --video-dir "$VIDEO_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --limit "$VIDEO_COUNT" \
    --target-fps "$TARGET_FPS" \
    --video-mode "$VIDEO_MODE" \
    --generate-depth \
    --depth-device cuda \
    "$@"
