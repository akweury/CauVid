#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CAUVID_IMAGE_NAME:-cauvid:latest}"
CONTAINER_NAME="${CAUVID_CONTAINER_NAME:-cauvid-exp-july}"
STORAGE_ROOT="${CAUVID_STORAGE_ROOT:-/storage-02/ml-jsha}"
OUTPUT_ROOT="${CAUVID_OUTPUT_ROOT:-/storage-01/ml-jsha/storage/CauVid_output}"
RAW_DATASET="${CAUVID_RAW_DRIVING_DATASET:-$STORAGE_ROOT/driving-video-with-object-tracking}"
DRIVING_MINI="${CAUVID_DRIVING_MINI_HOST:-$STORAGE_ROOT/driving_mini}"
NUSCENES="${CAUVID_NUSCENES_HOST:-$STORAGE_ROOT/nuScenes}"
PIPELINE_OUTPUT="${CAUVID_OUTPUT_JULY_HOST:-$OUTPUT_ROOT/pipeline_july}"
OUTPUT_DIR="${CAUVID_OUTPUT_HOST:-$OUTPUT_ROOT/output}"
LOGS_DIR="${CAUVID_LOGS_HOST:-$OUTPUT_ROOT/logs}"
TORCH_CACHE="${CAUVID_TORCH_CACHE_HOST:-$STORAGE_ROOT/.cache/torch}"

GPU_ARGS=()
GPU_ID="${CAUVID_GPU_ID:-}"
if [[ -n "$GPU_ID" ]]; then
  if [[ "$GPU_ID" == "all" ]]; then
    GPU_ARGS=(--gpus all)
  else
    GPU_ARGS=(--gpus "device=$GPU_ID")
  fi
elif [[ -n "${CAUVID_DOCKER_GPU_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  GPU_ARGS=(${CAUVID_DOCKER_GPU_ARGS})
else
  GPU_ARGS=(--gpus all)
fi

usage() {
  echo "Usage:"
  echo "  ./d2.sh run --gpu 0 --step 2 --data 10 --rounds 3"
  echo "  ./d2.sh                 # run exp_july with defaults"
  echo "  ./d2.sh build           # build docker image"
  echo "  ./d2.sh shell           # open interactive shell in container"
}

ensure_image() {
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    docker build -t "$IMAGE_NAME" "$ROOT_DIR"
  fi
}

prepare_dirs() {
  mkdir -p "$DRIVING_MINI" "$NUSCENES" "$PIPELINE_OUTPUT" "$OUTPUT_DIR" "$LOGS_DIR" "$TORCH_CACHE"
}

run_container() {
  local video_count="${1:-}"
  local rounds="${2:-3}"
  local max_step="${3:-18}"
  local model_mounts=()
  local runtime_env_args=()

  # Forward runtime service configuration from the host. Using `-e NAME`
  # keeps secrets out of this script and lets Docker read the current value.
  [[ -n "${OPENAI_API_KEY:-}" ]] && runtime_env_args+=(-e OPENAI_API_KEY)
  [[ -n "${OPENAI_BASE_URL:-}" ]] && runtime_env_args+=(-e OPENAI_BASE_URL)
  [[ -n "${OPENAI_MODEL:-}" ]] && runtime_env_args+=(-e OPENAI_MODEL)
  [[ -n "${CAUVID_STEP8_PATTERN_LLM_MODEL:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_PATTERN_LLM_MODEL)
  [[ -n "${CAUVID_STEP8C_LLM_TIMEOUT_SECONDS:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8C_LLM_TIMEOUT_SECONDS)
  [[ -n "${CAUVID_STEP8C_LLM_MAX_ATTEMPTS:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8C_LLM_MAX_ATTEMPTS)
  [[ -n "${CAUVID_STEP8C_LLM_RETRY_BACKOFF_SECONDS:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8C_LLM_RETRY_BACKOFF_SECONDS)
  [[ -n "${CAUVID_STEP8C_REVIEW_INTERVAL_TRACKS:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8C_REVIEW_INTERVAL_TRACKS)
  [[ -n "${CAUVID_STEP8_THRESHOLD_MIN_CONFLICTS:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_MIN_CONFLICTS)
  [[ -n "${CAUVID_STEP8_THRESHOLD_TARGET_QUANTILE:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_TARGET_QUANTILE)
  [[ -n "${CAUVID_STEP8_THRESHOLD_MAX_RELATIVE_CHANGE:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_MAX_RELATIVE_CHANGE)
  [[ -n "${CAUVID_STEP8_THRESHOLD_MAX_UNPROTECTED_FLIP_RATE:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_MAX_UNPROTECTED_FLIP_RATE)
  [[ -n "${CAUVID_STEP8_THRESHOLD_VALIDATION_FRACTION:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_VALIDATION_FRACTION)
  [[ -n "${WANDB_API_KEY:-}" ]] && runtime_env_args+=(-e WANDB_API_KEY)
  [[ -n "${WANDB_PROJECT:-}" ]] && runtime_env_args+=(-e WANDB_PROJECT)
  [[ -n "${WANDB_ENTITY:-}" ]] && runtime_env_args+=(-e WANDB_ENTITY)
  [[ -n "${WANDB_MODE:-}" ]] && runtime_env_args+=(-e WANDB_MODE)
  runtime_env_args+=(
    -e "CAUVID_WANDB_BASE_URL=${CAUVID_WANDB_BASE_URL:-https://api.wandb.ai}"
  )
  [[ -n "${WANDB_DIR:-}" ]] && runtime_env_args+=(-e WANDB_DIR)
  [[ -n "${WANDB_INIT_TIMEOUT:-}" ]] && runtime_env_args+=(-e WANDB_INIT_TIMEOUT)
  [[ -n "${CAUVID_WANDB_ENABLED:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_ENABLED)
  [[ -n "${CAUVID_WANDB_PROJECT:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_PROJECT)
  [[ -n "${CAUVID_WANDB_ENTITY:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_ENTITY)
  [[ -n "${CAUVID_WANDB_RUN_NAME:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_RUN_NAME)
  [[ -n "${CAUVID_WANDB_GROUP:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_GROUP)
  [[ -n "${CAUVID_WANDB_TAGS:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_TAGS)
  [[ -n "${CAUVID_WANDB_MODE:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_MODE)
  [[ -n "${CAUVID_WANDB_DIR:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_DIR)
  [[ -n "${CAUVID_WANDB_INIT_TIMEOUT_SECONDS:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_INIT_TIMEOUT_SECONDS)
  [[ -n "${CAUVID_WANDB_MAX_VIDEOS:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_MAX_VIDEOS)
  [[ -n "${CAUVID_WANDB_MAX_ARTIFACT_FILES:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_MAX_ARTIFACT_FILES)

  [[ -f "$ROOT_DIR/yolov8l-worldv2.pt" ]] && model_mounts+=(-v "$ROOT_DIR/yolov8l-worldv2.pt:/app/yolov8l-worldv2.pt:ro")
  [[ -f "$ROOT_DIR/yolov8s-worldv2.pt" ]] && model_mounts+=(-v "$ROOT_DIR/yolov8s-worldv2.pt:/app/yolov8s-worldv2.pt:ro")

  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  docker run --rm \
    "${GPU_ARGS[@]}" \
    -v "$ROOT_DIR/src:/app/src" \
    -v "$ROOT_DIR/configs:/app/configs" \
    -v "$ROOT_DIR/config.py:/app/config.py:ro" \
    -v "$RAW_DATASET:/raw_driving_data:ro" \
    -v "$DRIVING_MINI:/dataset/driving_mini" \
    -v "$NUSCENES:/dataset/nuScenes" \
    -v "$PIPELINE_OUTPUT:/output/output_july" \
    -v "$OUTPUT_DIR:/output/output" \
    -v "$LOGS_DIR:/logs" \
    -v "$TORCH_CACHE:/.cache/torch" \
    "${model_mounts[@]}" \
    "${runtime_env_args[@]}" \
    -e PYTHONPATH=/app:/app/external/Depth-Anything-3/src \
    -e MPLBACKEND=Agg \
    -e TORCH_HOME=/.cache/torch \
    -e CAUVID_RAW_DRIVING_DATASET=/raw_driving_data \
    -e CAUVID_DRIVING_MINI_PATH=/dataset/driving_mini \
    -e CAUVID_NUSCENES_PATH=/dataset/nuScenes \
    -e CAUVID_PIPELINE_OUTPUT_PATH=/output/output_july \
    -e CAUVID_OUTPUT_PATH=/output/output \
    -e EXP_JULY_VIDEO_COUNT="$video_count" \
    -e EXP_JULY_ROUNDS="$rounds" \
    -e EXP_JULY_MAX_STEP="$max_step" \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" \
    sh -lc 'python -c "import os; from src.exp_july.pipeline import main; count=os.getenv(\"EXP_JULY_VIDEO_COUNT\", \"\"); rounds=int(os.getenv(\"EXP_JULY_ROUNDS\", \"3\")); max_step=int(os.getenv(\"EXP_JULY_MAX_STEP\", \"18\")); main(video_count=int(count) if count else None, rounds=rounds, max_step=max_step)"'
}

shell_container() {
  local model_mounts=()
  local runtime_env_args=()
  [[ -n "${OPENAI_API_KEY:-}" ]] && runtime_env_args+=(-e OPENAI_API_KEY)
  [[ -n "${OPENAI_BASE_URL:-}" ]] && runtime_env_args+=(-e OPENAI_BASE_URL)
  [[ -n "${OPENAI_MODEL:-}" ]] && runtime_env_args+=(-e OPENAI_MODEL)
  [[ -n "${CAUVID_STEP8_PATTERN_LLM_MODEL:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_PATTERN_LLM_MODEL)
  [[ -n "${CAUVID_STEP8_THRESHOLD_MIN_CONFLICTS:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_MIN_CONFLICTS)
  [[ -n "${CAUVID_STEP8_THRESHOLD_TARGET_QUANTILE:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_TARGET_QUANTILE)
  [[ -n "${CAUVID_STEP8_THRESHOLD_MAX_RELATIVE_CHANGE:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_MAX_RELATIVE_CHANGE)
  [[ -n "${CAUVID_STEP8_THRESHOLD_MAX_UNPROTECTED_FLIP_RATE:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_MAX_UNPROTECTED_FLIP_RATE)
  [[ -n "${CAUVID_STEP8_THRESHOLD_VALIDATION_FRACTION:-}" ]] && runtime_env_args+=(-e CAUVID_STEP8_THRESHOLD_VALIDATION_FRACTION)
  [[ -n "${WANDB_API_KEY:-}" ]] && runtime_env_args+=(-e WANDB_API_KEY)
  [[ -n "${WANDB_PROJECT:-}" ]] && runtime_env_args+=(-e WANDB_PROJECT)
  [[ -n "${WANDB_ENTITY:-}" ]] && runtime_env_args+=(-e WANDB_ENTITY)
  [[ -n "${WANDB_MODE:-}" ]] && runtime_env_args+=(-e WANDB_MODE)
  runtime_env_args+=(
    -e "CAUVID_WANDB_BASE_URL=${CAUVID_WANDB_BASE_URL:-https://api.wandb.ai}"
  )
  [[ -n "${WANDB_DIR:-}" ]] && runtime_env_args+=(-e WANDB_DIR)
  [[ -n "${WANDB_INIT_TIMEOUT:-}" ]] && runtime_env_args+=(-e WANDB_INIT_TIMEOUT)
  [[ -n "${CAUVID_WANDB_ENABLED:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_ENABLED)
  [[ -n "${CAUVID_WANDB_PROJECT:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_PROJECT)
  [[ -n "${CAUVID_WANDB_ENTITY:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_ENTITY)
  [[ -n "${CAUVID_WANDB_RUN_NAME:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_RUN_NAME)
  [[ -n "${CAUVID_WANDB_GROUP:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_GROUP)
  [[ -n "${CAUVID_WANDB_TAGS:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_TAGS)
  [[ -n "${CAUVID_WANDB_MODE:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_MODE)
  [[ -n "${CAUVID_WANDB_DIR:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_DIR)
  [[ -n "${CAUVID_WANDB_INIT_TIMEOUT_SECONDS:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_INIT_TIMEOUT_SECONDS)
  [[ -n "${CAUVID_WANDB_MAX_VIDEOS:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_MAX_VIDEOS)
  [[ -n "${CAUVID_WANDB_MAX_ARTIFACT_FILES:-}" ]] && runtime_env_args+=(-e CAUVID_WANDB_MAX_ARTIFACT_FILES)
  [[ -f "$ROOT_DIR/yolov8l-worldv2.pt" ]] && model_mounts+=(-v "$ROOT_DIR/yolov8l-worldv2.pt:/app/yolov8l-worldv2.pt:ro")
  [[ -f "$ROOT_DIR/yolov8s-worldv2.pt" ]] && model_mounts+=(-v "$ROOT_DIR/yolov8s-worldv2.pt:/app/yolov8s-worldv2.pt:ro")

  docker rm -f "${CONTAINER_NAME}-shell" 2>/dev/null || true
  docker run -it --rm \
    "${GPU_ARGS[@]}" \
    -v "$ROOT_DIR/src:/app/src" \
    -v "$ROOT_DIR/configs:/app/configs" \
    -v "$ROOT_DIR/config.py:/app/config.py:ro" \
    -v "$RAW_DATASET:/raw_driving_data:ro" \
    -v "$DRIVING_MINI:/dataset/driving_mini" \
    -v "$NUSCENES:/dataset/nuScenes" \
    -v "$PIPELINE_OUTPUT:/output/output_july" \
    -v "$OUTPUT_DIR:/output/output" \
    -v "$LOGS_DIR:/logs" \
    -v "$TORCH_CACHE:/.cache/torch" \
    "${model_mounts[@]}" \
    "${runtime_env_args[@]}" \
    -e PYTHONPATH=/app:/app/external/Depth-Anything-3/src \
    -e MPLBACKEND=Agg \
    -e TORCH_HOME=/.cache/torch \
    -e CAUVID_RAW_DRIVING_DATASET=/raw_driving_data \
    -e CAUVID_DRIVING_MINI_PATH=/dataset/driving_mini \
    -e CAUVID_NUSCENES_PATH=/dataset/nuScenes \
    -e CAUVID_PIPELINE_OUTPUT_PATH=/output/output_july \
    -e CAUVID_OUTPUT_PATH=/output/output \
    --name "${CONTAINER_NAME}-shell" \
    "$IMAGE_NAME" \
    /bin/bash
}

main() {
  local cmd="${1:-run}"
  local video_count=""
  local rounds="3"
  local max_step="18"
  prepare_dirs

  if [[ "${cmd:-run}" == --* ]]; then
    cmd="run"
  fi

  case "$cmd" in
    build)
      docker build -t "$IMAGE_NAME" "$ROOT_DIR"
      ;;
    run)
      if [[ "${1:-}" == "run" ]]; then
        shift || true
      fi
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --gpu)
            GPU_ID="${2:?missing gpu id}"
            if [[ "$GPU_ID" == "all" ]]; then
              GPU_ARGS=(--gpus all)
            else
              GPU_ARGS=(--gpus "device=$GPU_ID")
            fi
            shift 2
            ;;
          --step)
            max_step="${2:?missing step id}"
            shift 2
            ;;
          --data|--video-count)
            video_count="${2:?missing video count}"
            shift 2
            ;;
          --rounds)
            rounds="${2:?missing rounds}"
            shift 2
            ;;
          *)
            echo "Unknown run option: $1" >&2
            usage
            exit 1
            ;;
        esac
      done
      ensure_image
      run_container "$video_count" "$rounds" "$max_step"
      ;;
    shell)
      shift || true
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --gpu)
            GPU_ID="${2:?missing gpu id}"
            if [[ "$GPU_ID" == "all" ]]; then
              GPU_ARGS=(--gpus all)
            else
              GPU_ARGS=(--gpus "device=$GPU_ID")
            fi
            shift 2
            ;;
          *)
            echo "Unknown shell option: $1" >&2
            usage
            exit 1
            ;;
        esac
      done
      ensure_image
      shell_container
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      ensure_image
      run_container "" "3" "18"
      ;;
  esac
}

main "$@"
