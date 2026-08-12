#!/usr/bin/env bash
set -euo pipefail

# Docker launcher for the Paper-1-only exp_august pipeline.  Its interface is
# intentionally parallel to d2.sh, while its container and output paths are
# isolated from exp_july.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CAUVID_IMAGE_NAME:-cauvid:latest}"
CONTAINER_NAME="${CAUVID_CONTAINER_NAME:-cauvid-exp-august}"
STORAGE_ROOT="${CAUVID_STORAGE_ROOT:-/storage-02/ml-jsha}"
OUTPUT_ROOT="${CAUVID_OUTPUT_ROOT:-/storage-01/ml-jsha/storage/CauVid_output}"
RAW_DATASET="${CAUVID_RAW_DRIVING_DATASET:-$STORAGE_ROOT/driving-video-with-object-tracking}"
DRIVING_MINI="${CAUVID_DRIVING_MINI_HOST:-$STORAGE_ROOT/driving_mini}"
NUSCENES="${CAUVID_NUSCENES_HOST:-$STORAGE_ROOT/nuScenes}"
PIPELINE_OUTPUT_BASE="${CAUVID_OUTPUT_AUGUST_HOST:-$OUTPUT_ROOT/pipeline_august}"
EVALUATION_OUTPUT_BASE="${CAUVID_AUGUST_EVALUATION_HOST:-}"
PIPELINE_OUTPUT="$PIPELINE_OUTPUT_BASE"
EVALUATION_OUTPUT="$PIPELINE_OUTPUT/evaluation"
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
  echo "  ./d3.sh run --gpu 0 --step 8 --evaluate --split test"
  echo "  ./d3.sh evaluate --split test    # evaluate existing predictions only"
  echo "  ./d3.sh                 # run exp_august with defaults"
  echo "  ./d3.sh build           # build docker image"
  echo "  ./d3.sh shell --gpu 0   # open interactive shell in container"
  echo ""
  echo "Run options:"
  echo "  --gpu ID                GPU device ID or 'all'"
  echo "  --step N                Last August step to run (1-11, default: 11)"
  echo "  --scale NAME            debug=10, small=100, full=961 (default: debug)"
  echo "  --data N                Custom video count (alias: --video-count)"
  echo "  --seed N                Seed value or index 1, 2, 3 (default: 1)"
  echo "  --diagnostics           Enable optional visualization/dashboard audits"
  echo "  --render-candidate-filter-comparisons"
  echo "  --evaluate              Evaluate predictions after the pipeline finishes"
  echo ""
  echo "Evaluation options:"
  echo "  --scale NAME            Select the matching run output"
  echo "  --split NAME            train, eval, test, or all (default: test)"
  echo "  Seeds:                  1=726381, 2=184957, 3=930241"
  echo "  --test-ratio R          Test fraction (default: 0.2)"
  echo "  --tolerances LIST       Comma-separated frame tolerances (default: 1,3,5,10)"
  echo "  CAUVID_AUGUST_EVALUATION_HOST overrides the host evaluation output root"
}

resolve_seed() {
  case "$1" in
    1) echo "726381" ;;
    2) echo "184957" ;;
    3) echo "930241" ;;
    726381|184957|930241) echo "$1" ;;
    *)
      echo "[d3][error] --seed must be 1, 2, 3, or one of 726381, 184957, 930241" >&2
      return 1
      ;;
  esac
}

configure_run_output() {
  local scale="$1"
  local seed="$2"
  PIPELINE_OUTPUT="$PIPELINE_OUTPUT_BASE/$scale/seed_$seed"
  if [[ -n "$EVALUATION_OUTPUT_BASE" ]]; then
    EVALUATION_OUTPUT="$EVALUATION_OUTPUT_BASE/$scale/seed_$seed"
  else
    EVALUATION_OUTPUT="$PIPELINE_OUTPUT/evaluation"
  fi
}

ensure_image() {
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    docker build -t "$IMAGE_NAME" "$ROOT_DIR"
  fi
}

prepare_dirs() {
  mkdir -p \
    "$DRIVING_MINI" \
    "$NUSCENES" \
    "$PIPELINE_OUTPUT" \
    "$EVALUATION_OUTPUT" \
    "$OUTPUT_DIR" \
    "$LOGS_DIR" \
    "$TORCH_CACHE"
}

prepare_evaluation_dirs() {
  mkdir -p "$PIPELINE_OUTPUT" "$EVALUATION_OUTPUT"
}

validate_annotations() {
  if [[ ! -d "$ROOT_DIR/annotations/video_segmentation" ]]; then
    echo "[d3][error] Annotation directory not found: $ROOT_DIR/annotations/video_segmentation" >&2
    exit 1
  fi
}

validate_driving_mini() {
  local has_frames=""
  local has_videos=""
  if [[ -d "$DRIVING_MINI/frames" ]]; then
    has_frames="$(find "$DRIVING_MINI/frames" -mindepth 2 -maxdepth 2 -type f -name 'frame_*.jpg' -print -quit 2>/dev/null)"
  fi
  if [[ -d "$DRIVING_MINI/videos" ]]; then
    has_videos="$(find "$DRIVING_MINI/videos" -maxdepth 1 -type f \( -name '*.mov' -o -name '*.mp4' -o -name '*.avi' -o -name '*.mkv' \) -print -quit 2>/dev/null)"
  fi
  if [[ -z "$has_frames" && -z "$has_videos" ]]; then
    echo "[d3][error] No prepared driving_mini data found at: $DRIVING_MINI" >&2
    echo "[d3][error] Set CAUVID_DRIVING_MINI_HOST to the prepared dataset directory." >&2
    exit 1
  fi
}

runtime_env_args() {
  local name
  RUNTIME_ENV_ARGS=()
  # Forward service credentials/settings by name so secret values are not
  # embedded in this script or printed in the docker command.
  for name in \
    OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    CAUVID_STEP8_PATTERN_LLM_MODEL \
    CAUVID_STEP8C_LLM_TIMEOUT_SECONDS \
    CAUVID_STEP8C_LLM_MAX_ATTEMPTS \
    CAUVID_STEP8C_LLM_RETRY_BACKOFF_SECONDS \
    CAUVID_STEP8C_REVIEW_INTERVAL_TRACKS \
    WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_MODE WANDB_DIR \
    CAUVID_WANDB_ENABLED CAUVID_WANDB_PROJECT CAUVID_WANDB_ENTITY \
    CAUVID_WANDB_RUN_NAME CAUVID_WANDB_GROUP CAUVID_WANDB_TAGS \
    CAUVID_WANDB_MODE CAUVID_WANDB_DIR CAUVID_WANDB_INIT_TIMEOUT_SECONDS \
    CAUVID_WANDB_MAX_VIDEOS CAUVID_WANDB_MAX_ARTIFACT_FILES
  do
    [[ -n "${!name:-}" ]] && RUNTIME_ENV_ARGS+=(-e "$name")
  done
  RUNTIME_ENV_ARGS+=(
    -e "CAUVID_WANDB_BASE_URL=${CAUVID_WANDB_BASE_URL:-https://api.wandb.ai}"
  )
}

docker_mount_args() {
  MODEL_MOUNTS=()
  [[ -d "$ROOT_DIR/weights" ]] && MODEL_MOUNTS+=(-v "$ROOT_DIR/weights:/app/weights:ro")
}

run_container() {
  local video_count="${1:-}"
  local max_step="${2:-11}"
  local diagnostics="${3:-0}"
  local render_comparisons="${4:-0}"
  local seed="${5:-726381}"
  runtime_env_args
  docker_mount_args

  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  docker run --rm \
    "${GPU_ARGS[@]}" \
    -v "$ROOT_DIR/src:/app/src" \
    -v "$ROOT_DIR/configs:/app/configs" \
    -v "$ROOT_DIR/config.py:/app/config.py:ro" \
    -v "$ROOT_DIR/annotations:/app/annotations:ro" \
    -v "$RAW_DATASET:/raw_driving_data:ro" \
    -v "$DRIVING_MINI:/dataset/driving_mini" \
    -v "$NUSCENES:/dataset/nuScenes" \
    -v "$PIPELINE_OUTPUT:/output/output_august" \
    -v "$OUTPUT_DIR:/output/output" \
    -v "$LOGS_DIR:/logs" \
    -v "$TORCH_CACHE:/.cache/torch" \
    "${MODEL_MOUNTS[@]}" \
    "${RUNTIME_ENV_ARGS[@]}" \
    -e PYTHONPATH=/app:/app/external/Depth-Anything-3/src \
    -e MPLBACKEND=Agg \
    -e TORCH_HOME=/.cache/torch \
    -e CAUVID_RAW_DRIVING_DATASET=/raw_driving_data \
    -e CAUVID_DRIVING_MINI_PATH=/dataset/driving_mini \
    -e CAUVID_NUSCENES_PATH=/dataset/nuScenes \
    -e CAUVID_PIPELINE_OUTPUT_PATH=/output/output_august \
    -e CAUVID_AUGUST_OUTPUT_PATH=/output/output_august \
    -e CAUVID_OUTPUT_PATH=/output/output \
    -e EXP_AUGUST_VIDEO_COUNT="$video_count" \
    -e EXP_AUGUST_MAX_STEP="$max_step" \
    -e EXP_AUGUST_DIAGNOSTICS="$diagnostics" \
    -e EXP_AUGUST_RENDER_COMPARISONS="$render_comparisons" \
    -e EXP_AUGUST_DATA_SEED="$seed" \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" \
    sh -lc 'python -c "import os; from src.exp_august.pipeline import main; count=os.getenv(\"EXP_AUGUST_VIDEO_COUNT\", \"\"); main(video_count=int(count) if count else None, seed=int(os.getenv(\"EXP_AUGUST_DATA_SEED\", \"726381\")), max_step=int(os.getenv(\"EXP_AUGUST_MAX_STEP\", \"11\")), diagnostics=os.getenv(\"EXP_AUGUST_DIAGNOSTICS\", \"0\") == \"1\", render_candidate_filter_comparisons=os.getenv(\"EXP_AUGUST_RENDER_COMPARISONS\", \"0\") == \"1\")"'
}

evaluate_container() {
  local split="${1:-test}"
  local seed="${2:-20260809}"
  local test_ratio="${3:-0.2}"
  local tolerances="${4:-1,3,5,10}"

  docker rm -f "${CONTAINER_NAME}-evaluation" 2>/dev/null || true
  docker run --rm \
    -v "$ROOT_DIR/src:/app/src:ro" \
    -v "$ROOT_DIR/annotations:/app/annotations:ro" \
    -v "$PIPELINE_OUTPUT:/output/output_august:ro" \
    -v "$EVALUATION_OUTPUT:/output/evaluation_august" \
    -e PYTHONPATH=/app \
    -e EXP_AUGUST_EVAL_SPLIT="$split" \
    -e EXP_AUGUST_EVAL_SEED="$seed" \
    -e EXP_AUGUST_EVAL_TEST_RATIO="$test_ratio" \
    -e EXP_AUGUST_EVAL_TOLERANCES="$tolerances" \
    --name "${CONTAINER_NAME}-evaluation" \
    "$IMAGE_NAME" \
    sh -lc 'python -c "import os; from pathlib import Path; from src.exp_august.evaluation import compact_summary, evaluate_dataset; split=os.environ[\"EXP_AUGUST_EVAL_SPLIT\"]; tolerances=tuple(int(value) for value in os.environ[\"EXP_AUGUST_EVAL_TOLERANCES\"].split(\",\") if value); output=Path(\"/output/evaluation_august\") / split; result=evaluate_dataset(\"/output/output_august\", \"/app/annotations/video_segmentation\", output, split=split, seed=int(os.environ[\"EXP_AUGUST_EVAL_SEED\"]), test_ratio=float(os.environ[\"EXP_AUGUST_EVAL_TEST_RATIO\"]), tolerances=tolerances); print(compact_summary(result)); print(\"results=\" + str(output / \"evaluation_results.json\"))"'
}

shell_container() {
  runtime_env_args
  docker_mount_args
  docker rm -f "${CONTAINER_NAME}-shell" 2>/dev/null || true
  docker run -it --rm \
    "${GPU_ARGS[@]}" \
    -v "$ROOT_DIR/src:/app/src" \
    -v "$ROOT_DIR/configs:/app/configs" \
    -v "$ROOT_DIR/config.py:/app/config.py:ro" \
    -v "$ROOT_DIR/annotations:/app/annotations:ro" \
    -v "$RAW_DATASET:/raw_driving_data:ro" \
    -v "$DRIVING_MINI:/dataset/driving_mini" \
    -v "$NUSCENES:/dataset/nuScenes" \
    -v "$PIPELINE_OUTPUT:/output/output_august" \
    -v "$OUTPUT_DIR:/output/output" \
    -v "$LOGS_DIR:/logs" \
    -v "$TORCH_CACHE:/.cache/torch" \
    "${MODEL_MOUNTS[@]}" \
    "${RUNTIME_ENV_ARGS[@]}" \
    -e PYTHONPATH=/app:/app/external/Depth-Anything-3/src \
    -e MPLBACKEND=Agg \
    -e TORCH_HOME=/.cache/torch \
    -e CAUVID_RAW_DRIVING_DATASET=/raw_driving_data \
    -e CAUVID_DRIVING_MINI_PATH=/dataset/driving_mini \
    -e CAUVID_NUSCENES_PATH=/dataset/nuScenes \
    -e CAUVID_PIPELINE_OUTPUT_PATH=/output/output_august \
    -e CAUVID_AUGUST_OUTPUT_PATH=/output/output_august \
    -e CAUVID_OUTPUT_PATH=/output/output \
    --name "${CONTAINER_NAME}-shell" \
    "$IMAGE_NAME" \
    /bin/bash
}

set_gpu() {
  GPU_ID="${1:?missing gpu id}"
  if [[ "$GPU_ID" == "all" ]]; then
    GPU_ARGS=(--gpus all)
  else
    GPU_ARGS=(--gpus "device=$GPU_ID")
  fi
}

main() {
  local cmd="${1:-run}"
  local data_scale="debug"
  local video_count="10"
  local custom_data="0"
  local max_step="11"
  local diagnostics="0"
  local render_comparisons="0"
  local evaluate_after_run="0"
  local evaluation_split="test"
  local evaluation_seed="726381"
  local evaluation_test_ratio="0.2"
  local evaluation_tolerances="1,3,5,10"

  [[ "$cmd" == --* && "$cmd" != "--help" ]] && cmd="run"
  case "$cmd" in
    build)
      docker build -t "$IMAGE_NAME" "$ROOT_DIR"
      ;;
    run)
      [[ "${1:-}" == "run" ]] && shift
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --gpu)
            set_gpu "${2:?missing gpu id}"
            shift 2
            ;;
          --step)
            max_step="${2:?missing step id}"
            shift 2
            ;;
          --data|--video-count)
            video_count="${2:?missing video count}"
            custom_data="1"
            shift 2
            ;;
          --scale)
            data_scale="${2:?missing data scale}"
            custom_data="0"
            shift 2
            ;;
          --diagnostics)
            diagnostics="1"
            shift
            ;;
          --render-candidate-filter-comparisons)
            render_comparisons="1"
            shift
            ;;
          --evaluate)
            evaluate_after_run="1"
            shift
            ;;
          --split)
            evaluation_split="${2:?missing evaluation split}"
            shift 2
            ;;
          --seed)
            evaluation_seed="${2:?missing evaluation seed}"
            shift 2
            ;;
          --test-ratio)
            evaluation_test_ratio="${2:?missing test ratio}"
            shift 2
            ;;
          --tolerances)
            evaluation_tolerances="${2:?missing comma-separated tolerances}"
            shift 2
            ;;
          *)
            echo "Unknown run option: $1" >&2
            usage
            exit 1
            ;;
        esac
      done
      if ! [[ "$max_step" =~ ^([1-9]|10|11)$ ]]; then
        echo "[d3][error] --step must be an integer from 1 through 11" >&2
        exit 1
      fi
      case "$data_scale" in
        debug) [[ "$custom_data" == "1" ]] || video_count="10" ;;
        small) [[ "$custom_data" == "1" ]] || video_count="100" ;;
        full) [[ "$custom_data" == "1" ]] || video_count="961" ;;
        *) echo "[d3][error] --scale must be debug, small, or full" >&2; exit 1 ;;
      esac
      if ! [[ "$video_count" =~ ^[1-9][0-9]*$ ]]; then
        echo "[d3][error] --data must be a positive integer" >&2
        exit 1
      fi
      [[ "$custom_data" == "1" ]] && data_scale="custom_${video_count}"
      evaluation_seed="$(resolve_seed "$evaluation_seed")"
      configure_run_output "$data_scale" "$evaluation_seed"
      if [[ "$data_scale" == "full" ]]; then
        export CAUVID_WANDB_ENABLED=1
        export CAUVID_WANDB_PROJECT="${CAUVID_WANDB_PROJECT:-cauvid-exp-august}"
        export CAUVID_WANDB_GROUP="${CAUVID_WANDB_GROUP:-full}"
        export CAUVID_WANDB_RUN_NAME="${CAUVID_WANDB_RUN_NAME:-full-seed-${evaluation_seed}}"
      fi
      echo "[d3] run scale=$data_scale videos=$video_count seed=$evaluation_seed output=$PIPELINE_OUTPUT"
      [[ "$data_scale" == "full" ]] && echo "[d3] wandb enabled project=$CAUVID_WANDB_PROJECT run=$CAUVID_WANDB_RUN_NAME"
      if [[ "$evaluate_after_run" == "1" && "$max_step" -lt 8 ]]; then
        echo "[d3][error] --evaluate requires --step 8 or later" >&2
        exit 1
      fi
      if [[ ! "$evaluation_split" =~ ^(train|eval|dev|test|all)$ ]]; then
        echo "[d3][error] --split must be train, eval, test, or all" >&2
        exit 1
      fi
      if [[ ! "$evaluation_tolerances" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "[d3][error] --tolerances must be comma-separated non-negative integers" >&2
        exit 1
      fi
      prepare_dirs
      validate_driving_mini
      [[ "$evaluate_after_run" == "1" ]] && validate_annotations
      ensure_image
      run_container "$video_count" "$max_step" "$diagnostics" "$render_comparisons" "$evaluation_seed"
      if [[ "$evaluate_after_run" == "1" ]]; then
        evaluate_container "$evaluation_split" "$evaluation_seed" "$evaluation_test_ratio" "$evaluation_tolerances"
      fi
      ;;
    evaluate)
      shift || true
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --scale)
            data_scale="${2:?missing data scale}"
            shift 2
            ;;
          --split)
            evaluation_split="${2:?missing evaluation split}"
            shift 2
            ;;
          --seed)
            evaluation_seed="${2:?missing evaluation seed}"
            shift 2
            ;;
          --test-ratio)
            evaluation_test_ratio="${2:?missing test ratio}"
            shift 2
            ;;
          --tolerances)
            evaluation_tolerances="${2:?missing comma-separated tolerances}"
            shift 2
            ;;
          *)
            echo "Unknown evaluate option: $1" >&2
            usage
            exit 1
            ;;
        esac
      done
      if [[ ! "$evaluation_split" =~ ^(train|eval|dev|test|all)$ ]]; then
        echo "[d3][error] --split must be train, eval, test, or all" >&2
        exit 1
      fi
      if ! [[ "$data_scale" =~ ^(debug|small|full|custom_[1-9][0-9]*)$ ]]; then
        echo "[d3][error] --scale must be debug, small, full, or custom_N" >&2
        exit 1
      fi
      evaluation_seed="$(resolve_seed "$evaluation_seed")"
      configure_run_output "$data_scale" "$evaluation_seed"
      echo "[d3] evaluate scale=$data_scale seed=$evaluation_seed input=$PIPELINE_OUTPUT"
      if [[ ! "$evaluation_tolerances" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "[d3][error] --tolerances must be comma-separated non-negative integers" >&2
        exit 1
      fi
      prepare_evaluation_dirs
      validate_annotations
      ensure_image
      evaluate_container "$evaluation_split" "$evaluation_seed" "$evaluation_test_ratio" "$evaluation_tolerances"
      ;;
    shell)
      shift || true
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --gpu)
            set_gpu "${2:?missing gpu id}"
            shift 2
            ;;
          *)
            echo "Unknown shell option: $1" >&2
            usage
            exit 1
            ;;
        esac
      done
      prepare_dirs
      ensure_image
      shell_container
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
