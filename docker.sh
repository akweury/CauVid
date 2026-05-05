#!/bin/bash

# CauVid Docker Management Script
# This script helps build and run the CauVid video processing pipeline in Docker

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="cauvid-pipeline"
CONTAINER_NAME="cauvid-app"
VERSION="latest"
REMOTE_STORAGE_ROOT="/home/ml-jsha/storage/CauVid_Data"

# Functions
print_help() {
    echo -e "${BLUE}CauVid Docker Management Script${NC}"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Docker image"
    echo "  prepare       Prepare raw .mov + CSV dataset for percept2matrix"
    echo "  run           Run driving perception-to-matrix preprocessing"
    echo "  preprocess    Run driving perception-to-matrix preprocessing"
    echo "  download-nuscenes  Download nuScenes archives from config"
    echo "  demo          Run advanced features demo"
    echo "  bonds         Run bond type classification demo"
    echo "  dev           Start development container with interactive shell"
    echo "  stop          Stop and remove containers"
    echo "  clean         Remove containers and images"
    echo "  logs          Show container logs"
    echo "  shell         Open shell in running container"
    echo "  help          Show this help message"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Verbose output"
    echo "  -f, --force      Force rebuild/restart"
    echo "  CAUVID_STORAGE_ROOT defaults outputs to /home/ml-jsha/storage/CauVid_Data when present"
    echo "  CAUVID_DOCKER_GPU_ARGS=\"--gpus all\" enables GPU access for docker run"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build the Docker image"
    echo "  CAUVID_DOCKER_GPU_ARGS=\"--gpus all\" CAUVID_RAW_DRIVING_DATASET=/home/ml-jsha/storage/CauVid_Data/driving-video-with-object-tracking $0 prepare --limit 1000 --target-fps 5 --generate-depth"
    echo "  $0 run                      # Run driving preprocessing over dataset/driving_mini/videos"
    echo "  $0 preprocess               # Same as run"
    echo "  $0 demo                     # Run advanced features demo"
    echo "  $0 dev                      # Start development environment"
    echo "  $0 clean                    # Clean up all containers and images"
}

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        warn "Docker Compose is not installed. Some features may not work."
    fi
    
    local storage_root
    storage_root="$(get_storage_root)"
    local raw_dataset="${CAUVID_RAW_DRIVING_DATASET:-$storage_root/driving-video-with-object-tracking}"
    local prepared_dataset="${CAUVID_DRIVING_MINI_HOST:-$storage_root/driving_mini}"

    if [ ! -d "$raw_dataset" ]; then
        warn "Raw dataset directory not found: $raw_dataset"
    fi

    if [ ! -d "$prepared_dataset" ]; then
        warn "Prepared dataset directory not found yet: $prepared_dataset"
    fi
    
    log "Dependencies check complete"
}

build_image() {
    local force_rebuild=${1:-false}
    
    log "Building CauVid Docker image..."
    
    if [ "$force_rebuild" = true ]; then
        docker build --no-cache -t ${IMAGE_NAME}:${VERSION} .
    else
        docker build -t ${IMAGE_NAME}:${VERSION} .
    fi
    
    log "Image built successfully: ${IMAGE_NAME}:${VERSION}"
}

get_storage_root() {
    if [ -n "${CAUVID_STORAGE_ROOT:-}" ]; then
        echo "$CAUVID_STORAGE_ROOT"
    elif [ -d "$REMOTE_STORAGE_ROOT" ]; then
        echo "$REMOTE_STORAGE_ROOT"
    else
        echo "$(pwd)"
    fi
}

get_raw_dataset() {
    local storage_root
    storage_root="$(get_storage_root)"
    echo "${CAUVID_RAW_DRIVING_DATASET:-$storage_root/driving-video-with-object-tracking}"
}

get_prepared_dataset() {
    local storage_root
    storage_root="$(get_storage_root)"
    echo "${CAUVID_DRIVING_MINI_HOST:-$storage_root/driving_mini}"
}

get_nuscenes_dataset_root() {
    local storage_root
    storage_root="$(get_storage_root)"
    echo "${CAUVID_NUSCENES_HOST:-$storage_root/nuScenes}"
}

get_pipeline_output_dir() {
    local storage_root
    storage_root="$(get_storage_root)"
    echo "${CAUVID_PIPELINE_OUTPUT_HOST:-$storage_root/pipeline_output}"
}

get_output_dir() {
    local storage_root
    storage_root="$(get_storage_root)"
    echo "${CAUVID_OUTPUT_HOST:-$storage_root/output}"
}

get_logs_dir() {
    local storage_root
    storage_root="$(get_storage_root)"
    echo "${CAUVID_LOGS_HOST:-$storage_root/logs}"
}

get_torch_cache_dir() {
    local storage_root
    storage_root="$(get_storage_root)"
    echo "${CAUVID_TORCH_CACHE_HOST:-$storage_root/.cache/torch}"
}

run_pipeline() {
    log "Running driving perception-to-matrix preprocessing..."
    
    # Create writable host directories for outputs, logs, and model cache.
    local raw_dataset prepared_dataset nuscenes_dataset_root pipeline_output_dir output_dir logs_dir torch_cache_dir
    raw_dataset="$(get_raw_dataset)"
    prepared_dataset="$(get_prepared_dataset)"
    nuscenes_dataset_root="$(get_nuscenes_dataset_root)"
    pipeline_output_dir="$(get_pipeline_output_dir)"
    output_dir="$(get_output_dir)"
    logs_dir="$(get_logs_dir)"
    torch_cache_dir="$(get_torch_cache_dir)"
    mkdir -p "$prepared_dataset" "$nuscenes_dataset_root" "$pipeline_output_dir" "$output_dir" "$logs_dir" "$torch_cache_dir"
    
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        ${CAUVID_DOCKER_GPU_ARGS:-} \
        -v "$raw_dataset:/app/raw_driving_data:ro" \
        -v "$prepared_dataset:/app/dataset/driving_mini" \
        -v "$nuscenes_dataset_root:/app/dataset/nuScenes" \
        -v "$pipeline_output_dir:/app/pipeline_output" \
        -v "$output_dir:/app/output" \
        -v "$logs_dir:/app/logs" \
        -v "$torch_cache_dir:/app/.cache/torch" \
        -e PYTHONPATH=/app \
        -e MPLBACKEND=Agg \
        -e TORCH_HOME=/app/.cache/torch \
        -e CAUVID_RAW_DRIVING_DATASET=/app/raw_driving_data \
        -e CAUVID_DRIVING_MINI_PATH=/app/dataset/driving_mini \
        -e CAUVID_NUSCENES_PATH=/app/dataset/nuScenes \
        -e CAUVID_PIPELINE_OUTPUT_PATH=/app/pipeline_output \
        -e CAUVID_OUTPUT_PATH=/app/output \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME}:${VERSION} \
        python -m src.exp_driving_videos.pipe_utils.percept2matrix
    
    log "Preprocessing completed. Check $pipeline_output_dir for per-video pipeline_data.pkl files."
}

prepare_driving_dataset() {
    log "Preparing raw driving dataset..."

    local raw_dataset prepared_dataset nuscenes_dataset_root output_dir logs_dir torch_cache_dir
    raw_dataset="$(get_raw_dataset)"
    prepared_dataset="$(get_prepared_dataset)"
    nuscenes_dataset_root="$(get_nuscenes_dataset_root)"
    output_dir="$(get_output_dir)"
    logs_dir="$(get_logs_dir)"
    torch_cache_dir="$(get_torch_cache_dir)"
    mkdir -p "$prepared_dataset" "$nuscenes_dataset_root" "$output_dir" "$logs_dir" "$torch_cache_dir"

    docker run --rm \
        ${CAUVID_DOCKER_GPU_ARGS:-} \
        -v "$raw_dataset:/app/raw_driving_data:ro" \
        -v "$prepared_dataset:/app/dataset/driving_mini" \
        -v "$nuscenes_dataset_root:/app/dataset/nuScenes" \
        -v "$(pwd)/external:/app/external:ro" \
        -v "$output_dir:/app/output" \
        -v "$logs_dir:/app/logs" \
        -v "$torch_cache_dir:/app/.cache/torch" \
        -e PYTHONPATH=/app \
        -e MPLBACKEND=Agg \
        -e TORCH_HOME=/app/.cache/torch \
        -e CAUVID_RAW_DRIVING_DATASET=/app/raw_driving_data \
        -e CAUVID_DRIVING_MINI_PATH=/app/dataset/driving_mini \
        -e CAUVID_NUSCENES_PATH=/app/dataset/nuScenes \
        -e CAUVID_OUTPUT_PATH=/app/output \
        --name ${CONTAINER_NAME}-prepare \
        ${IMAGE_NAME}:${VERSION} \
        python -m src.exp_driving_videos.prepare_driving_dataset "$@"

    log "Dataset preparation completed: $prepared_dataset"
}

run_demo() {
    local demo_type=${1:-"advanced"}
    
    case $demo_type in
        "advanced")
            log "Running advanced features demonstration..."
            demo_script="demo_advanced_features.py"
            ;;
        "bonds")
            log "Running bond type classification demonstration..."
            demo_script="demo_bond_types.py"
            ;;
        *)
            error "Unknown demo type: $demo_type"
            ;;
    esac
    
    local raw_dataset prepared_dataset nuscenes_dataset_root pipeline_output_dir output_dir logs_dir torch_cache_dir
    raw_dataset="$(get_raw_dataset)"
    prepared_dataset="$(get_prepared_dataset)"
    nuscenes_dataset_root="$(get_nuscenes_dataset_root)"
    pipeline_output_dir="$(get_pipeline_output_dir)"
    output_dir="$(get_output_dir)"
    logs_dir="$(get_logs_dir)"
    torch_cache_dir="$(get_torch_cache_dir)"
    mkdir -p "$prepared_dataset" "$nuscenes_dataset_root" "$pipeline_output_dir" "$output_dir" "$logs_dir" "$torch_cache_dir"
    
    docker run --rm \
        ${CAUVID_DOCKER_GPU_ARGS:-} \
        -v "$raw_dataset:/app/raw_driving_data:ro" \
        -v "$prepared_dataset:/app/dataset/driving_mini" \
        -v "$nuscenes_dataset_root:/app/dataset/nuScenes" \
        -v "$pipeline_output_dir:/app/pipeline_output" \
        -v "$output_dir:/app/output" \
        -v "$logs_dir:/app/logs" \
        -v "$torch_cache_dir:/app/.cache/torch" \
        -e PYTHONPATH=/app \
        -e MPLBACKEND=Agg \
        -e TORCH_HOME=/app/.cache/torch \
        -e CAUVID_RAW_DRIVING_DATASET=/app/raw_driving_data \
        -e CAUVID_DRIVING_MINI_PATH=/app/dataset/driving_mini \
        -e CAUVID_NUSCENES_PATH=/app/dataset/nuScenes \
        -e CAUVID_PIPELINE_OUTPUT_PATH=/app/pipeline_output \
        -e CAUVID_OUTPUT_PATH=/app/output \
        --name ${CONTAINER_NAME}-demo \
        ${IMAGE_NAME}:${VERSION} \
        python $demo_script
    
    log "Demo completed. Check pipeline_output/ for results."
}

start_dev() {
    log "Starting development container..."
    
    local raw_dataset prepared_dataset nuscenes_dataset_root pipeline_output_dir output_dir logs_dir torch_cache_dir
    raw_dataset="$(get_raw_dataset)"
    prepared_dataset="$(get_prepared_dataset)"
    nuscenes_dataset_root="$(get_nuscenes_dataset_root)"
    pipeline_output_dir="$(get_pipeline_output_dir)"
    output_dir="$(get_output_dir)"
    logs_dir="$(get_logs_dir)"
    torch_cache_dir="$(get_torch_cache_dir)"
    mkdir -p "$prepared_dataset" "$nuscenes_dataset_root" "$pipeline_output_dir" "$output_dir" "$logs_dir" "$torch_cache_dir"
    
    docker run -it --rm \
        ${CAUVID_DOCKER_GPU_ARGS:-} \
        -v "$(pwd)/src:/app/src" \
        -v "$(pwd)/configs:/app/configs" \
        -v "$raw_dataset:/app/raw_driving_data:ro" \
        -v "$prepared_dataset:/app/dataset/driving_mini" \
        -v "$nuscenes_dataset_root:/app/dataset/nuScenes" \
        -v "$(pwd)/external:/app/external:ro" \
        -v "$pipeline_output_dir:/app/pipeline_output" \
        -v "$output_dir:/app/output" \
        -v "$logs_dir:/app/logs" \
        -v "$torch_cache_dir:/app/.cache/torch" \
        -e PYTHONPATH=/app \
        -e MPLBACKEND=Agg \
        -e TORCH_HOME=/app/.cache/torch \
        -e CAUVID_RAW_DRIVING_DATASET=/app/raw_driving_data \
        -e CAUVID_DRIVING_MINI_PATH=/app/dataset/driving_mini \
        -e CAUVID_NUSCENES_PATH=/app/dataset/nuScenes \
        -e CAUVID_PIPELINE_OUTPUT_PATH=/app/pipeline_output \
        -e CAUVID_OUTPUT_PATH=/app/output \
        --name ${CONTAINER_NAME}-dev \
        ${IMAGE_NAME}:${VERSION} \
        /bin/bash
}

download_nuscenes() {
    log "Downloading nuScenes archives from config..."

    local nuscenes_dataset_root logs_dir torch_cache_dir user_cache_dir write_probe cache_probe
    nuscenes_dataset_root="$(get_nuscenes_dataset_root)"
    logs_dir="$(get_logs_dir)"
    torch_cache_dir="$(get_torch_cache_dir)"
    user_cache_dir="$(get_storage_root)/.cache/user"
    mkdir -p "$nuscenes_dataset_root" "$logs_dir" "$torch_cache_dir" "$user_cache_dir"

    write_probe="$nuscenes_dataset_root/.cauvid_write_test.$$"
    if ! (: > "$write_probe") 2>/dev/null; then
        warn "No write permission for nuScenes host directory: $nuscenes_dataset_root"
        warn "If files were previously created by root, fix ownership with:"
        warn "  sudo chown -R $(id -u):$(id -g) \"$nuscenes_dataset_root\""
        error "Cannot continue until host write permissions are fixed."
    fi
    rm -f "$write_probe"

    cache_probe="$user_cache_dir/.cauvid_cache_write_test.$$"
    if ! (: > "$cache_probe") 2>/dev/null; then
        warn "No write permission for cache host directory: $user_cache_dir"
        warn "Matplotlib/Fontconfig cache creation will fail unless ownership is fixed:"
        warn "  sudo chown -R $(id -u):$(id -g) \"$user_cache_dir\""
        error "Cannot continue until host cache permissions are fixed."
    fi
    rm -f "$cache_probe"

    docker run --rm \
        --user "$(id -u):$(id -g)" \
        ${CAUVID_DOCKER_GPU_ARGS:-} \
        -v "$(pwd)/src:/app/src" \
        -v "$(pwd)/configs:/app/configs" \
        -v "$nuscenes_dataset_root:/app/dataset/nuScenes" \
        -v "$logs_dir:/app/logs" \
        -v "$torch_cache_dir:/app/.cache/torch" \
        -v "$user_cache_dir:/app/.cache/user" \
        -e PYTHONPATH=/app \
        -e MPLBACKEND=Agg \
        -e TORCH_HOME=/app/.cache/torch \
        -e HOME=/app/.cache/user \
        -e MPLCONFIGDIR=/app/.cache/user/matplotlib \
        -e XDG_CACHE_HOME=/app/.cache/user \
        -e FONTCONFIG_CACHE=/app/.cache/user/fontconfig \
        -e CAUVID_NUSCENES_PATH=/app/dataset/nuScenes \
        --name ${CONTAINER_NAME}-download-nuscenes \
        ${IMAGE_NAME}:${VERSION} \
        python -m src.exp_nuScenes.download_nuscenes_dataset --config configs/exp_nuScenes/default.yaml "$@"

    log "nuScenes download step completed. Data root: $nuscenes_dataset_root"
}

stop_containers() {
    log "Stopping CauVid containers..."
    
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker stop ${CONTAINER_NAME}-dev 2>/dev/null || true
    docker stop ${CONTAINER_NAME}-demo 2>/dev/null || true
    
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME}-dev 2>/dev/null || true
    docker rm ${CONTAINER_NAME}-demo 2>/dev/null || true
    
    log "Containers stopped and removed"
}

clean_up() {
    log "Cleaning up CauVid Docker resources..."
    
    stop_containers
    
    # Remove image
    docker rmi ${IMAGE_NAME}:${VERSION} 2>/dev/null || true
    
    # Remove any dangling images
    docker image prune -f
    
    log "Cleanup completed"
}

show_logs() {
    log "Showing container logs..."
    
    if docker ps -a --format "table {{.Names}}" | grep -q ${CONTAINER_NAME}; then
        docker logs -f ${CONTAINER_NAME}
    else
        warn "No running container found"
    fi
}

open_shell() {
    log "Opening shell in running container..."
    
    if docker ps --format "table {{.Names}}" | grep -q ${CONTAINER_NAME}; then
        docker exec -it ${CONTAINER_NAME} /bin/bash
    else
        warn "No running container found. Use 'dev' command to start development container."
    fi
}

# Parse arguments
VERBOSE=false
FORCE=false
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        build)
            COMMAND="build"
            shift
            ;;
        prepare)
            COMMAND="prepare"
            shift
            PASSTHROUGH_ARGS=("$@")
            break
            ;;
        run)
            COMMAND="run"
            shift
            ;;
        preprocess)
            COMMAND="run"
            shift
            ;;
        download-nuscenes)
            COMMAND="download-nuscenes"
            shift
            PASSTHROUGH_ARGS=("$@")
            break
            ;;
        demo)
            COMMAND="demo"
            shift
            ;;
        bonds)
            COMMAND="demo"
            DEMO_TYPE="bonds"
            shift
            ;;
        dev)
            COMMAND="dev"
            shift
            ;;
        stop)
            COMMAND="stop"
            shift
            ;;
        clean)
            COMMAND="clean"
            shift
            ;;
        logs)
            COMMAND="logs"
            shift
            ;;
        shell)
            COMMAND="shell"
            shift
            ;;
        help|--help|-h)
            print_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Main execution
if [ -z "${COMMAND:-}" ]; then
    print_help
    exit 1
fi

# Check dependencies
check_dependencies

# Execute command
case $COMMAND in
    build)
        build_image $FORCE
        ;;
    prepare)
        build_image false
        prepare_driving_dataset ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
        ;;
    run)
        build_image false
        run_pipeline
        ;;
    download-nuscenes)
        build_image false
        download_nuscenes ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
        ;;
    demo)
        build_image false
        run_demo "${DEMO_TYPE:-advanced}"
        ;;
    dev)
        build_image false
        start_dev
        ;;
    stop)
        stop_containers
        ;;
    clean)
        clean_up
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    *)
        error "Unknown command: $COMMAND"
        ;;
esac

log "Operation completed successfully!"
