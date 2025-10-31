#!/bin/bash
# Test vLLM Docker Image Locally with Omega-17 Model
# This script tests the Docker image on H100 before pushing to Docker Hub

set -e  # Exit on error

# Configuration
IMAGE_NAME="vllm-usf:1.0.6"
CONTAINER_NAME="vllm-omega-test"
HF_TOKEN="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"

echo "========================================"
echo "Testing vLLM Docker Image Locally"
echo "========================================"
echo ""
echo "Image: ${IMAGE_NAME}"
echo "Model: arpitsh018/omega-17-exp-vl-v0.1-checkpoint-1020"
echo "GPU: H100 (Compute Capability 9.0)"
echo ""

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "❌ Error: Image '${IMAGE_NAME}' not found!"
    echo ""
    echo "Please build the image first:"
    echo "  touch vllm/__init__.py && ./build.sh"
    exit 1
fi

echo "✓ Docker image found"
echo ""

# Stop and remove any existing container with the same name
if docker ps -a | grep -q "${CONTAINER_NAME}"; then
    echo "Stopping and removing existing container..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

echo "Starting vLLM server in Docker container..."
echo ""
echo "Access the server at: http://localhost:8000"
echo "API Key: sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

# Run the container
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --shm-size=8g \
  -p 8000:8000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  "${IMAGE_NAME}" \
  --served-model-name omega \
  --model arpitsh018/omega-17-exp-vl-v0.1-checkpoint-1020 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --api-key sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86 \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --trust-remote-code

echo "Container started! Showing logs..."
echo "Press Ctrl+C to stop following logs (container will keep running)"
echo ""

# Follow logs
docker logs -f "${CONTAINER_NAME}"