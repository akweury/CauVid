#!/bin/bash
# Build (if needed) and run the exp_roadpp pipeline in Docker on this machine.
#
# This targets the `args.machine == "dgx"` branch in src/exp_roadpp/config.py,
# which hardcodes the dataset_path/output_dir for the dgx docker setup. To keep
# that code path working unchanged, the dataset is bind-mounted at the exact
# same absolute path inside the container as on the dgx host.
#
# Usage:
#   ./run_roadpp_docker.sh [--gpu 0|all|none] [-- <extra args for main.py>]
#
# Examples:
#   ./run_roadpp_docker.sh
#   ./run_roadpp_docker.sh --gpu 0 -- --exp medium --learning True
#   ./run_roadpp_docker.sh --gpu none -- --device cpu --machine jst
#
# Environment variables (set before running, or export in your shell profile):
#   CAUVID_ROADPP_DATA_HOST   Host path to the ROAD-Waymo dataset on dgx
#                             (default: /home/sha/mnt/remote/dgx-g/storage-01/CauVid_Data/roadpp,
#                             matching config.py's "dgx" dataset_path)
#   CAUVID_TORCH_CACHE_HOST   Host path to cache downloaded model weights
#                             (default: /storage-02/ml-jsha/.cache/torch)
#   CAUVID_ROADPP_DEVICE      Device string passed to main.py
#                             (default: cuda:0, overridden by --gpu none -> cpu)
#   CAUVID_ROADPP_MACHINE     --machine value passed to main.py (default: dgx)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="cauvid-pipeline"
GPU_ARG="0"
EXTRA_ARGS=()

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ARG="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

ROADPP_DATA_HOST="${CAUVID_ROADPP_DATA_HOST:-/storage-01/ml-jsha/CauVid_Data/roadpp}"
TORCH_CACHE_HOST="${CAUVID_TORCH_CACHE_HOST:-/storage-02/ml-jsha/.cache/torch}"
MACHINE="${CAUVID_ROADPP_MACHINE:-dgx}"

mkdir -p "$TORCH_CACHE_HOST"

if [[ "$GPU_ARG" == "none" ]]; then
    GPU_DOCKER_ARGS=()
    DEVICE="${CAUVID_ROADPP_DEVICE:-cpu}"
else
    GPU_DOCKER_ARGS=(--gpus "$GPU_ARG")
    DEVICE="${CAUVID_ROADPP_DEVICE:-$GPU_ARG}"
fi

echo "[run_roadpp_docker] Image:           $IMAGE_NAME"
echo "[run_roadpp_docker] Dataset path:    $ROADPP_DATA_HOST"
echo "[run_roadpp_docker] Torch cache:     $TORCH_CACHE_HOST"
echo "[run_roadpp_docker] Machine:         $MACHINE"
echo "[run_roadpp_docker] GPU:             $GPU_ARG"
echo "[run_roadpp_docker] Device arg:      $DEVICE"
echo "[run_roadpp_docker] Extra args:      ${EXTRA_ARGS[*]:-<none>}"

# --- Build image if it doesn't exist yet ---
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[run_roadpp_docker] Image '$IMAGE_NAME' not found, building..."
    docker build -t "$IMAGE_NAME" -f Dockerfile .
fi

# --- Run the pipeline ---
# The dataset is mounted at the *same* absolute path as on the dgx host so
# config.py's hardcoded "dgx" dataset_path resolves correctly inside the
# container without any code changes. Output is written under
# src/exp_roadpp/output/roadpp/<exp>, which lives inside the bind-mounted
# $SCRIPT_DIR/src, so results land back on the host automatically.
docker run --rm -it \
    "${GPU_DOCKER_ARGS[@]}" \
    -v "$SCRIPT_DIR/src:/app/src" \
    -v "$SCRIPT_DIR/configs:/app/configs" \
    -v "$ROADPP_DATA_HOST:$ROADPP_DATA_HOST" \
    -v "$TORCH_CACHE_HOST:/.cache/torch" \
    -e PYTHONPATH=/app \
    -e MPLBACKEND=Agg \
    -e TORCH_HOME=/.cache/torch \
    "$IMAGE_NAME" \
    python -m src.exp_roadpp.main --device "$DEVICE" --machine "$MACHINE" "${EXTRA_ARGS[@]}"
