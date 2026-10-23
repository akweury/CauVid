#!/bin/bash
# Build (if needed) and run the exp_roadpp pipeline in Docker on this machine.
#
# This targets the `args.machine == "dgx"` branch in src/exp_roadpp/config.py,
# which hardcodes the dataset_path/output_dir for the dgx docker setup. To keep
# that code path working unchanged, the dataset is bind-mounted at the exact
# same absolute path inside the container as on the dgx host.
#
# Usage:
#   ./run_roadpp_docker.sh [--gpu 0|all|none] [--rebuild] [-- <extra args for main.py>]
#   Add --tune_05 to the extra args to run step05_tune.py (step 05
#   hyperparameter sweep, needs the step 04 outputs) instead of main.py.
#
# The image is rebuilt automatically whenever requirements.txt or the
# Dockerfile changes (tracked via a content-hash label on the image).
# Use --rebuild to force a rebuild regardless.
#
# Examples:
#   ./run_roadpp_docker.sh
#   ./run_roadpp_docker.sh --gpu 0 -- --exp medium --learning True
#   ./run_roadpp_docker.sh --gpu none -- --device cpu --machine jst
#   ./run_roadpp_docker.sh --gpu 4 -- --exp medium --tune_05
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
FORCE_REBUILD=0
EXTRA_ARGS=()

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ARG="$2"
            shift 2
            ;;
        --rebuild)
            FORCE_REBUILD=1
            shift
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

# --tune_05 (allowed before or after --) runs the step 05 hyperparameter sweep
# instead of the pipeline; it is not an argument of main.py, so strip it.
ENTRY_MODULE="src.exp_roadpp.main"
PY_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--tune_05" ]]; then
        ENTRY_MODULE="src.exp_roadpp.step05_tune"
    else
        PY_ARGS+=("$arg")
    fi
done
EXTRA_ARGS=("${PY_ARGS[@]}")

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
echo "[run_roadpp_docker] Entry module:    $ENTRY_MODULE"
echo "[run_roadpp_docker] Extra args:      ${EXTRA_ARGS[*]:-<none>}"

# --- Build image if missing or if its dependencies changed ---
# Hash requirements.txt + Dockerfile and store it as an image label; rebuild
# when the hash differs so newly added packages get installed. Docker's layer
# cache keeps the apt layer, so only the pip install and later layers rerun.
DEPS_HASH="$(cat requirements.txt Dockerfile | sha256sum | cut -d' ' -f1)"
IMAGE_DEPS_HASH="$(docker image inspect -f '{{ index .Config.Labels "cauvid.deps_hash" }}' "$IMAGE_NAME" 2>/dev/null || true)"

if [[ "$FORCE_REBUILD" == "1" ]]; then
    echo "[run_roadpp_docker] --rebuild given, rebuilding image..."
    NEED_BUILD=1
elif [[ -z "$IMAGE_DEPS_HASH" ]]; then
    echo "[run_roadpp_docker] Image '$IMAGE_NAME' not found (or unlabeled), building..."
    NEED_BUILD=1
elif [[ "$IMAGE_DEPS_HASH" != "$DEPS_HASH" ]]; then
    echo "[run_roadpp_docker] requirements.txt/Dockerfile changed, rebuilding image..."
    NEED_BUILD=1
else
    NEED_BUILD=0
fi

if [[ "$NEED_BUILD" == "1" ]]; then
    docker build -t "$IMAGE_NAME" --label "cauvid.deps_hash=$DEPS_HASH" -f Dockerfile .
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
    python -m "$ENTRY_MODULE" --device "$DEVICE" --machine "$MACHINE" "${EXTRA_ARGS[@]}"

# --- Report where the outputs were written ---
EXP_NAME="debug"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "--exp" ]]; then
        EXP_NAME="${EXTRA_ARGS[$((i + 1))]}"
        break
    fi
done
echo "[run_roadpp_docker] Output folder:   $ROADPP_OUTPUT_HOST/roadpp/$EXP_NAME"
