# CauVid Docker Deployment Guide

This guide explains how to build and run the CauVid video processing pipeline using Docker.

## 🐳 Quick Start

### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for advanced deployment)
- Your video data in the `two_parts/` directory

### Simple Usage

```bash
# Build and run the pipeline
./docker.sh run

# Run advanced features demo
./docker.sh demo

# Run bond type classification demo
./docker.sh bonds
```

## 📋 Available Commands

### Build & Run
```bash
./docker.sh build          # Build Docker image
./docker.sh run             # Run example pipeline
./docker.sh demo            # Run advanced features demo
./docker.sh bonds           # Run bond classification demo
```

### Development
```bash
./docker.sh dev             # Start development container with shell
./docker.sh shell           # Open shell in running container
./docker.sh logs            # Show container logs
```

### Management
```bash
./docker.sh stop            # Stop containers
./docker.sh clean           # Clean up containers and images
./docker.sh help            # Show help
```

## 🔧 Manual Docker Commands

### Build Image
```bash
docker build -t cauvid-pipeline:latest .
```

### Run Pipeline
```bash
docker run --rm \
  -v "$(pwd)/two_parts:/app/two_parts:ro" \
  -v "$(pwd)/pipeline_output:/app/pipeline_output" \
  cauvid-pipeline:latest \
  python example_usage.py
```

### Development Mode
```bash
docker run -it --rm \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/two_parts:/app/two_parts:ro" \
  -v "$(pwd)/pipeline_output:/app/pipeline_output" \
  cauvid-pipeline:latest \
  /bin/bash
```

## 🐙 Docker Compose

### Standard Deployment
```bash
docker-compose up cauvid
```

### Development Environment
```bash
docker-compose up -d cauvid-dev
docker-compose exec cauvid-dev bash
```

### Custom Commands
```bash
# Run specific demo
docker-compose run --rm cauvid python demo_bond_types.py

# Run with different parameters
docker-compose run --rm cauvid python -c "
from src.video_pipeline import VideoPipeline
pipeline = VideoPipeline('./two_parts', bond_threshold=0.9)
# ... custom processing
"
```

## 📁 Directory Structure

The Docker container expects the following structure:

```
CauVid/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration
├── docker.sh              # Management script
├── requirements.txt        # Python dependencies
├── src/                   # Source code
│   └── video_pipeline.py
├── two_parts/            # Input data (mounted as volume)
│   └── observation/
├── pipeline_output/      # Output results (mounted as volume)
└── logs/                # Application logs (mounted as volume)
```

## 🔧 Environment Variables

Configure the container behavior with environment variables:

```bash
# In docker-compose.yml or with -e flags
PYTHONPATH=/app/src         # Python module path
LOG_LEVEL=INFO             # Logging level (DEBUG, INFO, WARNING, ERROR)
BOND_THRESHOLD=0.8         # Default bond similarity threshold
FRAME_HEIGHT=480           # Video frame height
FRAME_WIDTH=640            # Video frame width
```

## 🚀 Production Deployment

### Using Docker Compose
```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# Scale processing (if needed)
docker-compose up --scale cauvid=3

# Monitor logs
docker-compose logs -f cauvid
```

### Custom Configuration
Create a production `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  cauvid:
    image: cauvid-pipeline:latest
    restart: always
    volumes:
      - /data/video_input:/app/two_parts:ro
      - /data/results:/app/pipeline_output
      - /var/log/cauvid:/app/logs
    environment:
      - LOG_LEVEL=WARNING
      - BOND_THRESHOLD=0.85
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 📊 Performance Considerations

### Resource Requirements
- **Memory**: 2-4GB RAM recommended
- **CPU**: Multi-core for faster processing
- **Storage**: Space for input data and output results

### Volume Mounts
- `two_parts/`: Read-only input data
- `pipeline_output/`: Results output (ensure write permissions)
- `logs/`: Application logs
- `src/`: Source code (development only)

### Optimization Tips
```bash
# Use multi-stage build for smaller production image
# Enable BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t cauvid-pipeline .

# Use Docker layer caching
docker build --cache-from cauvid-pipeline:latest -t cauvid-pipeline:latest .
```

## 🐛 Troubleshooting

### Common Issues

1. **Permission Issues**
   ```bash
   # Fix output directory permissions
   sudo chown -R $(id -u):$(id -g) pipeline_output/
   ```

2. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop: Settings > Resources > Memory
   # Linux: Edit /etc/docker/daemon.json
   ```

3. **Missing Data**
   ```bash
   # Ensure data directory exists and is mounted
   ls -la two_parts/
   docker run --rm -v "$(pwd)/two_parts:/app/two_parts:ro" cauvid-pipeline ls /app/two_parts
   ```

4. **OpenCV Issues**
   ```bash
   # Test OpenCV installation
   docker run --rm cauvid-pipeline python -c "import cv2; print(cv2.__version__)"
   ```

### Debug Mode
```bash
# Run with debug logging
docker run --rm \
  -e LOG_LEVEL=DEBUG \
  -v "$(pwd)/two_parts:/app/two_parts:ro" \
  -v "$(pwd)/pipeline_output:/app/pipeline_output" \
  cauvid-pipeline:latest
```

### Health Check
```bash
# Check container health
docker run --rm cauvid-pipeline python -c "
import cv2, numpy, sklearn, scipy
print('All dependencies working!')
"
```

## 🔍 Advanced Usage

### Custom Processing Scripts
```bash
# Run custom analysis
docker run --rm -it \
  -v "$(pwd)/two_parts:/app/two_parts:ro" \
  -v "$(pwd)/pipeline_output:/app/pipeline_output" \
  -v "$(pwd)/custom_scripts:/app/custom" \
  cauvid-pipeline:latest \
  python /app/custom/my_analysis.py
```

### Batch Processing
```bash
# Process multiple observations
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  cauvid-pipeline:latest \
  python -c "
from src.video_pipeline import VideoPipeline
pipeline = VideoPipeline('/app/data')
results = pipeline.process_all_observations()
print(f'Processed {len(results)} observations')
"
```

### Pipeline Monitoring
```bash
# Monitor resource usage
docker stats cauvid-pipeline

# Monitor logs in real-time
docker logs -f cauvid-pipeline

# Export processing metrics
docker run --rm \
  -v "$(pwd)/two_parts:/app/two_parts:ro" \
  -v "$(pwd)/metrics:/app/metrics" \
  cauvid-pipeline:latest \
  python -c "
# Custom metrics collection script
import json, time
from src.video_pipeline import VideoPipeline

start_time = time.time()
pipeline = VideoPipeline('./two_parts')
# ... processing ...
end_time = time.time()

metrics = {
    'processing_time': end_time - start_time,
    'timestamp': time.time()
}

with open('/app/metrics/run_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
"
```

## 📈 Scaling & Distribution

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  cauvid-worker:
    image: cauvid-pipeline:latest
    deploy:
      replicas: 3
    volumes:
      - data_input:/app/two_parts:ro
      - data_output:/app/pipeline_output
```

### Load Balancing
```bash
# Use Docker Swarm for production scaling
docker swarm init
docker service create --replicas 3 cauvid-pipeline:latest
```

This Docker setup provides a complete containerized environment for the CauVid video processing pipeline with all dependencies and configurations managed automatically! 🐳