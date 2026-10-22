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
#                             (default: /storage-01/ml-jsha/CauVid_Data/roadpp,
#                             matching config.py's "dgx" dataset_path)
#   CAUVID_ROADPP_OUTPUT_HOST Host path for pipeline outputs on dgx
#                             (default: /storage-01/ml-jsha/CauVid_output,
#                             matching config.py's "dgx" output_dir)
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
ROADPP_OUTPUT_HOST="${CAUVID_ROADPP_OUTPUT_HOST:-/storage-01/ml-jsha/CauVid_output}"
TORCH_CACHE_HOST="${CAUVID_TORCH_CACHE_HOST:-/storage-02/ml-jsha/.cache/torch}"
MACHINE="${CAUVID_ROADPP_MACHINE:-dgx-jing}"

mkdir -p "$TORCH_CACHE_HOST" "$ROADPP_OUTPUT_HOST"

if [[ "$GPU_ARG" == "none" ]]; then
    GPU_DOCKER_ARGS=()
    DEVICE="${CAUVID_ROADPP_DEVICE:-cpu}"
elif [[ "$GPU_ARG" == "all" ]]; then
    GPU_DOCKER_ARGS=(--gpus all)
    DEVICE="${CAUVID_ROADPP_DEVICE:-cuda:0}"
else
    # Pin to a specific physical GPU index. Docker then exposes only that
    # GPU inside the container, where it always shows up as cuda:0 (NOT
    # cuda:$GPU_ARG) -- so the --device passed to main.py must be cuda:0.
    GPU_DOCKER_ARGS=(--gpus "device=$GPU_ARG")
    DEVICE="${CAUVID_ROADPP_DEVICE:-cuda:0}"
fi

echo "[run_roadpp_docker] Image:           $IMAGE_NAME"
echo "[run_roadpp_docker] Dataset path:    $ROADPP_DATA_HOST"
echo "[run_roadpp_docker] Output host path:$ROADPP_OUTPUT_HOST"
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
# The dataset and output dirs are mounted at the *same* absolute paths as on
# the dgx host so config.py's hardcoded "dgx" dataset_path/output_dir resolve
# correctly inside the container without any code changes.
docker run --rm -it \
    "${GPU_DOCKER_ARGS[@]}" \
    -v "$SCRIPT_DIR/src:/app/src" \
    -v "$SCRIPT_DIR/configs:/app/configs" \
    -v "$ROADPP_DATA_HOST:$ROADPP_DATA_HOST" \
    -v "$ROADPP_OUTPUT_HOST:$ROADPP_OUTPUT_HOST" \
    -v "$TORCH_CACHE_HOST:/.cache/torch" \
    -e PYTHONPATH=/app \
    -e MPLBACKEND=Agg \
    -e TORCH_HOME=/.cache/torch \
    "$IMAGE_NAME" \
    python -m src.exp_roadpp.main --device "$DEVICE" --machine "$MACHINE" "${EXTRA_ARGS[@]}"

# --- Report where the outputs were written ---
EXP_NAME="debug"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "--exp" ]]; then
        EXP_NAME="${EXTRA_ARGS[$((i + 1))]}"
        break
    fi
done
echo "[run_roadpp_docker] Output folder:   $ROADPP_OUTPUT_HOST/roadpp/$EXP_NAME"
