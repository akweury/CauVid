# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg
ENV TORCH_HOME=/.cache/torch
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV FONTCONFIG_CACHE=/tmp/.cache/fontconfig

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional system tools
    libgcc-s1 \
    # Build essentials for some Python packages
    build-essential \
    gcc \
    g++ \
    # Git for any git-based dependencies
    git \
    # ffprobe/ffmpeg for video metadata and media writing
    ffmpeg \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY *.py ./
COPY README.md ./
COPY LICENSE ./

# Create necessary directories (data/output are mounted externally, not under /app)
RUN mkdir -p temp external && \
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git external/Depth-Anything-3

# Set Python path to include the project root for src.* imports and the
# vendored Depth Anything 3 source tree.
ENV PYTHONPATH="/app:/app/external/Depth-Anything-3/src"

# Expose port if needed (for future web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2, imageio, matplotlib, numpy, pandas, sklearn, scipy, torch, torchvision; print('Dependencies OK')" || exit 1

# Default command
CMD ["python", "-m", "src.exp_driving_videos.exp_driving_pipeline"]
