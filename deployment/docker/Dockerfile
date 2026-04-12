# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# We use a direct pip install for the core requirements
RUN pip3 install --no-cache-dir \
    torch \
    transformers \
    fastapi \
    uvicorn \
    accelerate \
    sentencepiece \
    mlflow

# Copy the server script and necessary files
# Note: In production, we usually copy only what's needed
COPY scripts/serve_openai_api.py /app/server.py
# We will mount the model weights as a volume later for flexibility

# Expose the port
EXPOSE 8998

# Start the server
# Using 0.0.0.0 is critical for Docker networking
CMD ["python3", "/app/server.py", "--host", "0.0.0.0", "--port", "8998"]
