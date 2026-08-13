#!/usr/bin/env bash
set -euo pipefail

# Docker launcher for the annotation-free exp_august target pipeline.
# The CLI follows d3.sh where the new pipeline has an equivalent operation,
# while the container name and output tree are deliberately isolated from d3.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CAUVID_IMAGE_NAME:-cauvid:latest}"
CONTAINER_NAME="${CAUVID_D4_CONTAINER_NAME:-cauvid-exp-august-target}"
STORAGE_ROOT="${CAUVID_STORAGE_ROOT:-/storage-02/ml-jsha}"
OUTPUT_ROOT="${CAUVID_OUTPUT_ROOT:-/storage-01/ml-jsha/storage/CauVid_output}"
RAW_DATASET="${CAUVID_RAW_DRIVING_DATASET:-$STORAGE_ROOT/driving-video-with-object-tracking}"
DRIVING_MINI="${CAUVID_DRIVING_MINI_HOST:-$STORAGE_ROOT/driving_mini}"
NUSCENES="${CAUVID_NUSCENES_HOST:-$STORAGE_ROOT/nuScenes}"

# Never share the d3 output tree. Override D4 with CAUVID_OUTPUT_D4_HOST only.
D3_OUTPUT_BASE="${CAUVID_OUTPUT_AUGUST_HOST:-$OUTPUT_ROOT/pipeline_august}"
PIPELINE_OUTPUT_BASE="${CAUVID_OUTPUT_D4_HOST:-$OUTPUT_ROOT/pipeline_august_target}"
PIPELINE_OUTPUT="$PIPELINE_OUTPUT_BASE"
OUTPUT_DIR="${CAUVID_OUTPUT_HOST:-$OUTPUT_ROOT/output}"
LOGS_DIR="${CAUVID_LOGS_HOST:-$OUTPUT_ROOT/logs}"
TORCH_CACHE="${CAUVID_TORCH_CACHE_HOST:-$STORAGE_ROOT/.cache/torch}"
HF_CACHE="${CAUVID_HF_CACHE_HOST:-$STORAGE_ROOT/.cache/huggingface}"
YOLO_MODEL="${CAUVID_D4_YOLO_MODEL:-weights/yolo/yolov8s-worldv2.pt}"
SAM2_MODEL="${CAUVID_D4_SAM2_MODEL:-weights/sam2/sam2_t.pt}"

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
  echo "  ./d4.sh run --gpu 0 --step 3 --scale debug --seed 1 --diagnostics"
  echo "  ./d4.sh                 # run the new pipeline with debug defaults"
  echo "  ./d4.sh build           # build the Docker image"
  echo "  ./d4.sh shell --gpu 0   # open an interactive container shell"
  echo ""
  echo "Run options (d3-compatible names):"
  echo "  --gpu ID                GPU device ID or 'all'"
  echo "  --step N                Last target-pipeline step (1-3, default: 3)"
  echo "  --scale NAME            debug=10, small=100, full=961 (default: debug)"
  echo "  --data N                Custom video count (alias: --video-count)"
  echo "  --seed N                Seed value or index 1, 2, 3 (default: 1)"
  echo "  --diagnostics           Render Step 3 example frames and video"
  echo "  --render-candidate-filter-comparisons"
  echo "                           Compatibility alias enabling diagnostics"
  echo ""
  echo "Target-pipeline options:"
  echo "  --canonical-fps FPS     Normalized timeline rate (default: 0.2)"
  echo "  --batch-size N          Neural evidence batch size (default: 4)"
  echo "  --depth-resolution N    DA3 processing resolution (default: 224)"
  echo "  --no-model-download     Require every model to exist in mounted caches"
  echo "  --no-step3-video        Render example frames but not the summary video"
  echo ""
  echo "Seeds: 1=726381, 2=184957, 3=930241"
  echo "Output: CAUVID_OUTPUT_D4_HOST (default: .../pipeline_august_target)"
  echo "Models: CAUVID_D4_YOLO_MODEL and CAUVID_D4_SAM2_MODEL may override defaults"
  echo "Note: --evaluate and the evaluate command are not yet available for the"
  echo "      new typed contracts; d4 never invokes the legacy pipeline/evaluator."
}

resolve_seed() {
  case "$1" in
    1) echo "726381" ;;
    2) echo "184957" ;;
    3) echo "930241" ;;
    726381|184957|930241) echo "$1" ;;
    *)
      echo "[d4][error] --seed must be 1, 2, 3, or one of 726381, 184957, 930241" >&2
      return 1
      ;;
  esac
}

set_gpu() {
  GPU_ID="${1:?missing gpu id}"
  if [[ "$GPU_ID" == "all" ]]; then
    GPU_ARGS=(--gpus all)
  else
    GPU_ARGS=(--gpus "device=$GPU_ID")
  fi
}

configure_run_output() {
  local scale="$1"
  local seed="$2"
  PIPELINE_OUTPUT="$PIPELINE_OUTPUT_BASE/$scale/seed_$seed"
}

validate_output_isolation() {
  case "$PIPELINE_OUTPUT_BASE/" in
    "$D3_OUTPUT_BASE/"*)
      echo "[d4][error] D4 output overlaps the d3 output tree: $D3_OUTPUT_BASE" >&2
      echo "[d4][error] Set CAUVID_OUTPUT_D4_HOST to an independent directory." >&2
      exit 1
      ;;
  esac
  case "$D3_OUTPUT_BASE/" in
    "$PIPELINE_OUTPUT_BASE/"*)
      echo "[d4][error] d3 output overlaps the D4 output tree: $PIPELINE_OUTPUT_BASE" >&2
      echo "[d4][error] Set CAUVID_OUTPUT_D4_HOST to an independent directory." >&2
      exit 1
      ;;
  esac
}

ensure_image() {
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    docker build -t "$IMAGE_NAME" "$ROOT_DIR"
  fi
}

prepare_dirs() {
  validate_output_isolation

  # Docker bind mounts do not reliably create a missing host directory with
  # the intended ownership. Create and validate the complete D4 output tree
  # before starting the container.
  if ! mkdir -p "$PIPELINE_OUTPUT_BASE" "$PIPELINE_OUTPUT"; then
    echo "[d4][error] Could not create the D4 output directory: $PIPELINE_OUTPUT" >&2
    exit 1
  fi
  if [[ ! -d "$PIPELINE_OUTPUT" || ! -w "$PIPELINE_OUTPUT" ]]; then
    echo "[d4][error] D4 output directory is not writable: $PIPELINE_OUTPUT" >&2
    exit 1
  fi
  echo "[d4] output directory ready: $PIPELINE_OUTPUT"

  mkdir -p \
    "$DRIVING_MINI" \
    "$NUSCENES" \
    "$OUTPUT_DIR" \
    "$LOGS_DIR" \
    "$TORCH_CACHE" \
    "$HF_CACHE"
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
    echo "[d4][error] No prepared driving_mini data found at: $DRIVING_MINI" >&2
    echo "[d4][error] Set CAUVID_DRIVING_MINI_HOST to the prepared dataset directory." >&2
    exit 1
  fi
}

runtime_env_args() {
  local name
  RUNTIME_ENV_ARGS=()
  for name in \
    HF_TOKEN HUGGING_FACE_HUB_TOKEN \
    WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_MODE WANDB_DIR \
    CAUVID_WANDB_ENABLED CAUVID_WANDB_PROJECT CAUVID_WANDB_ENTITY \
    CAUVID_WANDB_RUN_NAME CAUVID_WANDB_GROUP CAUVID_WANDB_TAGS \
    CAUVID_WANDB_MODE CAUVID_WANDB_DIR CAUVID_WANDB_INIT_TIMEOUT_SECONDS
  do
    [[ -n "${!name:-}" ]] && RUNTIME_ENV_ARGS+=(-e "$name")
  done
}

docker_mount_args() {
  MODEL_MOUNTS=()
  [[ -d "$ROOT_DIR/weights" ]] && MODEL_MOUNTS+=(-v "$ROOT_DIR/weights:/app/weights:ro")
}

run_container() {
  local video_count="$1"
  local max_step="$2"
  local diagnostics="$3"
  local seed="$4"
  local canonical_fps="$5"
  local batch_size="$6"
  local depth_resolution="$7"
  local allow_model_download="$8"
  local no_step3_video="$9"
  local runner_args=()
  local yolo_model="$YOLO_MODEL"
  local sam2_model="$SAM2_MODEL"

  runtime_env_args
  docker_mount_args

  # Ultralytics downloads known model basenames. Keep repository-relative paths
  # when local weights exist, otherwise use basenames for a portable first run.
  if [[ ! -f "$ROOT_DIR/$yolo_model" && "$allow_model_download" == "1" ]]; then
    yolo_model="$(basename "$yolo_model")"
  fi
  if [[ ! -f "$ROOT_DIR/$sam2_model" && "$allow_model_download" == "1" ]]; then
    sam2_model="$(basename "$sam2_model")"
  fi

  runner_args=(
    --dataset-root /dataset/driving_mini
    --dataset-name driving_mini
    --video-count "$video_count"
    --output-root /output/pipeline_august_target
    --seed "$seed"
    --evidence-policy-seed "$seed"
    --max-step "$max_step"
    --canonical-fps "$canonical_fps"
    --decode-validation sample
    --decode-sample-count 3
    --objects-backend yolo_world
    --yolo-model "$yolo_model"
    --masks-backend sam2
    --sam2-model "$sam2_model"
    --flow-backend raft
    --depth-backend da3
    --depth-process-resolution "$depth_resolution"
    --batch-size "$batch_size"
    --tracking-max-age-frames 2
    --tracking-min-score 0.30
    --device cuda:0
  )
  [[ "$diagnostics" == "1" ]] && runner_args+=(--visualize-step3)
  [[ "$allow_model_download" == "1" ]] && runner_args+=(--allow-model-download)
  [[ "$no_step3_video" == "1" ]] && runner_args+=(--no-step3-video)

  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  docker run --rm \
    "${GPU_ARGS[@]}" \
    -v "$ROOT_DIR/src:/app/src:ro" \
    -v "$ROOT_DIR/configs:/app/configs:ro" \
    -v "$ROOT_DIR/config.py:/app/config.py:ro" \
    -v "$RAW_DATASET:/raw_driving_data:ro" \
    -v "$DRIVING_MINI:/dataset/driving_mini:ro" \
    -v "$NUSCENES:/dataset/nuScenes:ro" \
    -v "$PIPELINE_OUTPUT:/output/pipeline_august_target" \
    -v "$OUTPUT_DIR:/output/output" \
    -v "$LOGS_DIR:/logs" \
    -v "$TORCH_CACHE:/cache/torch" \
    -v "$HF_CACHE:/cache/huggingface" \
    "${MODEL_MOUNTS[@]}" \
    "${RUNTIME_ENV_ARGS[@]}" \
    -e PYTHONPATH=/app:/app/external/Depth-Anything-3/src \
    -e MPLBACKEND=Agg \
    -e TORCH_HOME=/cache/torch \
    -e HF_HOME=/cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    -e XDG_CACHE_HOME=/cache \
    -e CAUVID_RAW_DRIVING_DATASET=/raw_driving_data \
    -e CAUVID_DRIVING_MINI_PATH=/dataset/driving_mini \
    -e CAUVID_NUSCENES_PATH=/dataset/nuScenes \
    -e CAUVID_PIPELINE_OUTPUT_PATH=/output/pipeline_august_target \
    -e CAUVID_AUGUST_TARGET_OUTPUT_PATH=/output/pipeline_august_target \
    -e CAUVID_OUTPUT_PATH=/output/output \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" \
    python -m src.exp_august.inference.runner "${runner_args[@]}"
}

shell_container() {
  runtime_env_args
  docker_mount_args
  docker rm -f "${CONTAINER_NAME}-shell" 2>/dev/null || true
  docker run -it --rm \
    "${GPU_ARGS[@]}" \
    -v "$ROOT_DIR/src:/app/src" \
    -v "$ROOT_DIR/configs:/app/configs:ro" \
    -v "$ROOT_DIR/config.py:/app/config.py:ro" \
    -v "$RAW_DATASET:/raw_driving_data:ro" \
    -v "$DRIVING_MINI:/dataset/driving_mini:ro" \
    -v "$NUSCENES:/dataset/nuScenes:ro" \
    -v "$PIPELINE_OUTPUT:/output/pipeline_august_target" \
    -v "$OUTPUT_DIR:/output/output" \
    -v "$LOGS_DIR:/logs" \
    -v "$TORCH_CACHE:/cache/torch" \
    -v "$HF_CACHE:/cache/huggingface" \
    "${MODEL_MOUNTS[@]}" \
    "${RUNTIME_ENV_ARGS[@]}" \
    -e PYTHONPATH=/app:/app/external/Depth-Anything-3/src \
    -e MPLBACKEND=Agg \
    -e TORCH_HOME=/cache/torch \
    -e HF_HOME=/cache/huggingface \
    -e XDG_CACHE_HOME=/cache \
    -e CAUVID_DRIVING_MINI_PATH=/dataset/driving_mini \
    -e CAUVID_PIPELINE_OUTPUT_PATH=/output/pipeline_august_target \
    --name "${CONTAINER_NAME}-shell" \
    "$IMAGE_NAME" \
    /bin/bash
}

main() {
  local cmd="${1:-run}"
  local data_scale="debug"
  local video_count="10"
  local custom_data="0"
  local max_step="3"
  local diagnostics="0"
  local seed="1"
  local canonical_fps="0.2"
  local batch_size="4"
  local depth_resolution="224"
  local allow_model_download="1"
  local no_step3_video="0"
  local evaluate_requested="0"

  [[ "$cmd" == --* && "$cmd" != "--help" ]] && cmd="run"
  case "$cmd" in
    build)
      docker build -t "$IMAGE_NAME" "$ROOT_DIR"
      ;;
    run)
      [[ "${1:-}" == "run" ]] && shift
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --gpu) set_gpu "${2:?missing gpu id}"; shift 2 ;;
          --step) max_step="${2:?missing step id}"; shift 2 ;;
          --data|--video-count)
            video_count="${2:?missing video count}"
            custom_data="1"
            shift 2
            ;;
          --scale) data_scale="${2:?missing data scale}"; custom_data="0"; shift 2 ;;
          --seed) seed="${2:?missing seed}"; shift 2 ;;
          --diagnostics) diagnostics="1"; shift ;;
          --render-candidate-filter-comparisons) diagnostics="1"; shift ;;
          --canonical-fps) canonical_fps="${2:?missing FPS}"; shift 2 ;;
          --batch-size) batch_size="${2:?missing batch size}"; shift 2 ;;
          --depth-resolution) depth_resolution="${2:?missing resolution}"; shift 2 ;;
          --no-model-download) allow_model_download="0"; shift ;;
          --no-step3-video) no_step3_video="1"; shift ;;
          --evaluate) evaluate_requested="1"; shift ;;
          --split|--test-ratio|--tolerances)
            # Parse d3 evaluation options so the final error explains the real issue.
            [[ $# -ge 2 ]] || { echo "[d4][error] $1 requires a value" >&2; exit 1; }
            evaluate_requested="1"
            shift 2
            ;;
          *) echo "[d4][error] Unknown run option: $1" >&2; usage; exit 1 ;;
        esac
      done
      if [[ "$evaluate_requested" == "1" ]]; then
        echo "[d4][error] Evaluation for the new typed pipeline is not implemented." >&2
        echo "[d4][error] d4 will not invoke the legacy d3 evaluator." >&2
        exit 1
      fi
      [[ "$max_step" =~ ^[1-3]$ ]] || { echo "[d4][error] --step must be 1, 2, or 3" >&2; exit 1; }
      case "$data_scale" in
        debug) [[ "$custom_data" == "1" ]] || video_count="10" ;;
        small) [[ "$custom_data" == "1" ]] || video_count="100" ;;
        full) [[ "$custom_data" == "1" ]] || video_count="961" ;;
        *) echo "[d4][error] --scale must be debug, small, or full" >&2; exit 1 ;;
      esac
      [[ "$video_count" =~ ^[1-9][0-9]*$ ]] || { echo "[d4][error] --data must be a positive integer" >&2; exit 1; }
      [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo "[d4][error] --batch-size must be a positive integer" >&2; exit 1; }
      [[ "$depth_resolution" =~ ^[1-9][0-9]*$ ]] || { echo "[d4][error] --depth-resolution must be a positive integer" >&2; exit 1; }
      [[ "$custom_data" == "1" ]] && data_scale="custom_$video_count"
      seed="$(resolve_seed "$seed")"
      configure_run_output "$data_scale" "$seed"
      echo "[d4] run target pipeline scale=$data_scale videos=$video_count seed=$seed step=$max_step"
      echo "[d4] output=$PIPELINE_OUTPUT"
      prepare_dirs
      validate_driving_mini
      ensure_image
      run_container "$video_count" "$max_step" "$diagnostics" "$seed" \
        "$canonical_fps" "$batch_size" "$depth_resolution" \
        "$allow_model_download" "$no_step3_video"
      ;;
    evaluate)
      echo "[d4][error] Evaluation for the new typed pipeline is not implemented." >&2
      echo "[d4][error] d4 never invokes src.exp_august.pipeline or its evaluator." >&2
      exit 1
      ;;
    shell)
      shift || true
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --gpu) set_gpu "${2:?missing gpu id}"; shift 2 ;;
          *) echo "[d4][error] Unknown shell option: $1" >&2; usage; exit 1 ;;
        esac
      done
      configure_run_output "shell" "manual"
      prepare_dirs
      ensure_image
      shell_container
      ;;
    help|-h|--help) usage ;;
    *) echo "[d4][error] Unknown command: $cmd" >&2; usage; exit 1 ;;
  esac
}

main "$@"
