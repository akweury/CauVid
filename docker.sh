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

# Functions
print_help() {
    echo -e "${BLUE}CauVid Docker Management Script${NC}"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Docker image"
    echo "  run           Run the pipeline with example data"
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
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build the Docker image"
    echo "  $0 run                      # Run example pipeline"
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
    
    if [ ! -d "two_parts" ]; then
        warn "two_parts directory not found. Make sure your data is available."
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

run_pipeline() {
    log "Running CauVid pipeline..."
    
    # Create output directory if it doesn't exist
    mkdir -p pipeline_output logs
    
    docker run --rm \
        -v "$(pwd)/two_parts:/app/two_parts:ro" \
        -v "$(pwd)/pipeline_output:/app/pipeline_output" \
        -v "$(pwd)/logs:/app/logs" \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME}:${VERSION} \
        python example_usage.py
    
    log "Pipeline completed. Check pipeline_output/ for results."
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
    
    mkdir -p pipeline_output logs
    
    docker run --rm \
        -v "$(pwd)/two_parts:/app/two_parts:ro" \
        -v "$(pwd)/pipeline_output:/app/pipeline_output" \
        -v "$(pwd)/logs:/app/logs" \
        --name ${CONTAINER_NAME}-demo \
        ${IMAGE_NAME}:${VERSION} \
        python $demo_script
    
    log "Demo completed. Check pipeline_output/ for results."
}

start_dev() {
    log "Starting development container..."
    
    mkdir -p pipeline_output logs
    
    docker run -it --rm \
        -v "$(pwd)/src:/app/src" \
        -v "$(pwd):/app/project" \
        -v "$(pwd)/two_parts:/app/two_parts:ro" \
        -v "$(pwd)/pipeline_output:/app/pipeline_output" \
        -v "$(pwd)/logs:/app/logs" \
        --name ${CONTAINER_NAME}-dev \
        ${IMAGE_NAME}:${VERSION} \
        /bin/bash
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

while [[ $# -gt 0 ]]; do
    case $1 in
        build)
            COMMAND="build"
            shift
            ;;
        run)
            COMMAND="run"
            shift
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
    run)
        build_image false
        run_pipeline
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