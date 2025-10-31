#!/bin/bash
# vLLM Docker Build Script
# This script builds the vLLM Docker image with custom configuration

set -e  # Exit on error

echo "Building vLLM Docker image..."
echo "Configuration:"
echo "  - CUDA Version: 12.4.0"
echo "  - Python Version: 3.11"
echo "  - GPU Architecture: 9.0 (H100 - Hopper)"
echo "  - Max Jobs: 16"
echo "  - Target: vllm-openai"
echo "  - Tag: vllm-usf:1.0.6"
echo ""

docker buildx build \
  --build-arg CUDA_VERSION=12.4.0 \
  --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
  --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg max_jobs=16 \
  --build-arg torch_cuda_arch_list='9.0' \
  --build-arg FLASHINFER_AOT_COMPILE=false \
  --build-arg RUN_WHEEL_CHECK=false \
  --tag vllm-usf:1.0.6 \
  --target vllm-openai \
  --load \
  --progress=plain \
  -f docker/Dockerfile \
  .

echo ""
echo "Build complete! Image tagged as: vllm-usf:1.0.6"
echo ""
echo "To push to Docker Hub (optional):"
echo "  docker tag vllm-usf:1.0.6 arpitsh018/vllm-usf:v1.0.6"
echo "  docker push arpitsh018/vllm-usf:v1.0.6"