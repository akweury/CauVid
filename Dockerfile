# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
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
COPY *.py ./
COPY *.md ./
COPY LICENSE ./

# Create necessary directories
RUN mkdir -p pipeline_output logs

# Set Python path to include src directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash cauvid && \
    chown -R cauvid:cauvid /app
USER cauvid

# Expose port if needed (for future web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2, numpy, sklearn, scipy; print('Dependencies OK')" || exit 1

# Default command
CMD ["python", "example_usage.py"]