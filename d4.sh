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
  echo "  ./d4.sh run --gpu 0 --step 8 --scale debug --seed 1 --diagnostics"
  echo "  ./d4.sh                 # run the new pipeline through Step 8 with debug defaults"
  echo "  ./d4.sh build           # build the Docker image"
  echo "  ./d4.sh shell --gpu 0   # open an interactive container shell"
  echo ""
  echo "Run options (d3-compatible names):"
  echo "  --gpu ID                GPU device ID or 'all'"
  echo "  --step N                Last target-pipeline step (1-8, default: 8)"
  echo "  --scale NAME            debug=10, small=100, full=961 (default: debug)"
  echo "  --data N                Custom video count (alias: --video-count)"
  echo "  --seed N                Seed value or index 1, 2, 3 (default: 1)"
  echo "  --diagnostics           Render available Step 3 through Step 8 diagnostics"
  echo "  --render-candidate-filter-comparisons"
  echo "                           Compatibility alias enabling diagnostics"
  echo ""
  echo "Target-pipeline options:"
  echo "  --canonical-fps FPS     Normalized timeline rate (default: 0.2)"
  echo "  --batch-size N          Neural evidence batch size (default: 4)"
  echo "  --depth-resolution N    DA3 processing resolution (default: 224)"
  echo "  --no-model-download     Require every model to exist in mounted caches"
  echo "  --no-step3-video        Render example frames but not the summary video"
  echo "  --no-step4-video        Render Step 4 plots/frames but not its summary video"
  echo "  --horizontal-fov DEG    Frozen Step 4 camera FOV prior (default: 90)"
  echo "  --world-top-k N         Maximum initial Step 5 hypotheses (default: 5)"
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
  return 0
}

ensure_image() {
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    docker build -t "$IMAGE_NAME" "$ROOT_DIR"
  fi
}

prepare_dirs() {
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
  # Resolve symlinks and relative components only after the directories exist.
  # The canonical paths make overlap checks and Docker's bind source explicit.
  PIPELINE_OUTPUT_BASE="$(cd "$PIPELINE_OUTPUT_BASE" && pwd -P)"
  PIPELINE_OUTPUT="$(cd "$PIPELINE_OUTPUT" && pwd -P)"
  if [[ -d "$D3_OUTPUT_BASE" ]]; then
    D3_OUTPUT_BASE="$(cd "$D3_OUTPUT_BASE" && pwd -P)"
  fi
  validate_output_isolation
  echo "[d4] host output directory ready: $PIPELINE_OUTPUT"
  echo "[d4] bind mapping: $PIPELINE_OUTPUT -> /output/pipeline_august_target"

  mkdir -p \
    "$DRIVING_MINI" \
    "$NUSCENES" \
    "$OUTPUT_DIR" \
    "$LOGS_DIR" \
    "$TORCH_CACHE" \
    "$HF_CACHE"
}

verify_output_bind_mount() {
  local probe_name=".d4_host_mount_probe_$$"
  local probe_path="$PIPELINE_OUTPUT/$probe_name"
  if ! printf '%s\n' "$PIPELINE_OUTPUT" >"$probe_path"; then
    echo "[d4][error] Could not create host mount probe: $probe_path" >&2
    exit 1
  fi
  if docker run --rm \
    --mount "type=bind,src=$PIPELINE_OUTPUT,dst=/output/pipeline_august_target" \
    "$IMAGE_NAME" \
    sh -c "test -f /output/pipeline_august_target/$probe_name && test -w /output/pipeline_august_target"
  then
    :
  else
    rm -f -- "$probe_path"
    echo "[d4][error] Docker did not receive the intended writable host bind mount." >&2
    echo "[d4][error] Host: $PIPELINE_OUTPUT" >&2
    echo "[d4][error] Container: /output/pipeline_august_target" >&2
    exit 1
  fi
  rm -f -- "$probe_path"
  echo "[d4] writable output bind verified"
}

verify_persisted_run_output() {
  local max_step="$1"
  local diagnostics="$2"
  local marker_path="$3"
  local expected_name="init_bundle.json"
  local artifact_path=""
  case "$max_step" in
    2) expected_name="neural_evidence_store.json" ;;
    3) expected_name="tracking_store.json" ;;
    4) expected_name="geometry_store.json" ;;
    5) expected_name="world_state_store.json" ;;
    6) expected_name="residual_store.json" ;;
    7) expected_name="repair_proposal_store.json" ;;
    8) expected_name="local_reestimation_store.json" ;;
  esac
  artifact_path="$(find "$PIPELINE_OUTPUT" -type f -newer "$marker_path" -name "$expected_name" -print -quit 2>/dev/null)"
  if [[ -z "$artifact_path" ]]; then
    echo "[d4][error] Container exited successfully, but no new $expected_name exists on the host." >&2
    echo "[d4][error] Checked host directory: $PIPELINE_OUTPUT" >&2
    echo "[d4][error] The run is not considered persisted." >&2
    return 1
  fi
  echo "[d4] persisted host artifact: $artifact_path"

  if [[ "$diagnostics" == "1" && "$max_step" -ge 3 ]]; then
    local visualization_name="step3_visualization_manifest.json"
    [[ "$max_step" -ge 4 ]] && visualization_name="step4_visualization_manifest.json"
    [[ "$max_step" -ge 5 ]] && visualization_name="step5_visualization_manifest.json"
    [[ "$max_step" -ge 6 ]] && visualization_name="step6_visualization_manifest.json"
    [[ "$max_step" -ge 7 ]] && visualization_name="step7_visualization_manifest.json"
    [[ "$max_step" -ge 8 ]] && visualization_name="step8_visualization_manifest.json"
    local visualization_path=""
    visualization_path="$(find "$PIPELINE_OUTPUT" -type f -newer "$marker_path" -name "$visualization_name" -print -quit 2>/dev/null)"
    if [[ -z "$visualization_path" ]]; then
      echo "[d4][error] Expected host visualization is missing: $visualization_name" >&2
      echo "[d4][error] Checked host directory: $PIPELINE_OUTPUT" >&2
      return 1
    fi
    echo "[d4] persisted host visualization: $visualization_path"
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
    echo "[d4][error] No prepared driving_mini data found at: $DRIVING_MINI" >&2
    echo "[d4][error] Set CAUVID_DRIVING_MINI_HOST to the prepared dataset directory." >&2
    exit 1
  fi
  return 0
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
  # With `set -e`, the final false optional-variable test must not become the
  # function status and silently terminate the launcher before `docker run`.
  return 0
}

docker_mount_args() {
  MODEL_MOUNTS=()
  [[ -d "$ROOT_DIR/weights" ]] && MODEL_MOUNTS+=(-v "$ROOT_DIR/weights:/app/weights:ro")
  return 0
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
  local horizontal_fov="${10}"
  local no_step4_video="${11}"
  local world_top_k="${12}"
  local runner_args=()
  local yolo_model="$YOLO_MODEL"
  local sam2_model="$SAM2_MODEL"
  local run_marker="$PIPELINE_OUTPUT/.d4_run_started_$$"

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
    --horizontal-fov-degrees "$horizontal_fov"
    --world-top-k "$world_top_k"
  )
  if [[ "$diagnostics" == "1" ]]; then
    runner_args+=(--visualize-step3 --visualize-step4 --visualize-step5 --visualize-step6 --visualize-step7 --visualize-step8)
  fi
  [[ "$allow_model_download" == "1" ]] && runner_args+=(--allow-model-download)
  [[ "$no_step3_video" == "1" ]] && runner_args+=(--no-step3-video)
  [[ "$no_step4_video" == "1" ]] && runner_args+=(--no-step4-video)

  if ! touch -- "$run_marker"; then
    echo "[d4][error] Could not create run marker in host output: $run_marker" >&2
    return 1
  fi
  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  if docker run --rm \
    "${GPU_ARGS[@]}" \
    -v "$ROOT_DIR/src:/app/src:ro" \
    -v "$ROOT_DIR/configs:/app/configs:ro" \
    -v "$ROOT_DIR/config.py:/app/config.py:ro" \
    -v "$RAW_DATASET:/raw_driving_data:ro" \
    -v "$DRIVING_MINI:/dataset/driving_mini:ro" \
    -v "$NUSCENES:/dataset/nuScenes:ro" \
    --mount "type=bind,src=$PIPELINE_OUTPUT,dst=/output/pipeline_august_target" \
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
  then
    :
  else
    local docker_status=$?
    rm -f -- "$run_marker"
    return "$docker_status"
  fi
  if ! verify_persisted_run_output "$max_step" "$diagnostics" "$run_marker"; then
    rm -f -- "$run_marker"
    return 1
  fi
  rm -f -- "$run_marker"
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
    --mount "type=bind,src=$PIPELINE_OUTPUT,dst=/output/pipeline_august_target" \
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
  local max_step="8"
  local diagnostics="0"
  local seed="1"
  local canonical_fps="0.2"
  local batch_size="4"
  local depth_resolution="224"
  local allow_model_download="1"
  local no_step3_video="0"
  local no_step4_video="0"
  local evaluate_requested="0"
  local horizontal_fov="90"
  local world_top_k="5"

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
          --no-step4-video) no_step4_video="1"; shift ;;
          --horizontal-fov) horizontal_fov="${2:?missing horizontal FOV}"; shift 2 ;;
          --world-top-k) world_top_k="${2:?missing Top-K value}"; shift 2 ;;
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
      [[ "$max_step" =~ ^[1-8]$ ]] || { echo "[d4][error] --step must be 1, 2, 3, 4, 5, 6, 7, or 8" >&2; exit 1; }
      case "$data_scale" in
        debug) [[ "$custom_data" == "1" ]] || video_count="10" ;;
        small) [[ "$custom_data" == "1" ]] || video_count="100" ;;
        full) [[ "$custom_data" == "1" ]] || video_count="961" ;;
        *) echo "[d4][error] --scale must be debug, small, or full" >&2; exit 1 ;;
      esac
      [[ "$video_count" =~ ^[1-9][0-9]*$ ]] || { echo "[d4][error] --data must be a positive integer" >&2; exit 1; }
      [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo "[d4][error] --batch-size must be a positive integer" >&2; exit 1; }
      [[ "$depth_resolution" =~ ^[1-9][0-9]*$ ]] || { echo "[d4][error] --depth-resolution must be a positive integer" >&2; exit 1; }
      [[ "$world_top_k" =~ ^[1-9][0-9]*$ ]] || { echo "[d4][error] --world-top-k must be a positive integer" >&2; exit 1; }
      (( world_top_k <= 64 )) || { echo "[d4][error] --world-top-k must be at most 64" >&2; exit 1; }
      [[ "$custom_data" == "1" ]] && data_scale="custom_$video_count"
      seed="$(resolve_seed "$seed")"
      configure_run_output "$data_scale" "$seed"
      echo "[d4] run target pipeline scale=$data_scale videos=$video_count seed=$seed step=$max_step"
      echo "[d4] output=$PIPELINE_OUTPUT"
      prepare_dirs
      validate_driving_mini
      ensure_image
      verify_output_bind_mount
      run_container "$video_count" "$max_step" "$diagnostics" "$seed" \
        "$canonical_fps" "$batch_size" "$depth_resolution" \
        "$allow_model_download" "$no_step3_video" "$horizontal_fov" \
        "$no_step4_video" "$world_top_k"
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
      verify_output_bind_mount
      shell_container
      ;;
    help|-h|--help) usage ;;
    *) echo "[d4][error] Unknown command: $cmd" >&2; usage; exit 1 ;;
  esac
}

main "$@"
