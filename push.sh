#!/bin/bash
# vLLM Docker Push Script
# This script tags and pushes the vLLM image to Docker Hub

set -e  # Exit on error

# Configuration
LOCAL_IMAGE="vllm-usf:1.0.6"
DOCKER_HUB_USER="arpitsh018"
DOCKER_HUB_IMAGE="vllm-usf"
VERSION="v1.0.6"
REMOTE_TAG="${DOCKER_HUB_USER}/${DOCKER_HUB_IMAGE}:${VERSION}"

echo "========================================"
echo "vLLM Docker Image Push Script"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Local image:  ${LOCAL_IMAGE}"
echo "  Remote tag:   ${REMOTE_TAG}"
echo ""

# Check if the local image exists
if ! docker image inspect "${LOCAL_IMAGE}" > /dev/null 2>&1; then
    echo "❌ Error: Local image '${LOCAL_IMAGE}' not found!"
    echo ""
    echo "Please build the image first by running:"
    echo "  ./build.sh"
    exit 1
fi

echo "✓ Local image found"
echo ""

# Check if logged in to Docker Hub
if ! docker info | grep -q "Username: ${DOCKER_HUB_USER}"; then
    echo "⚠️  You are not logged in to Docker Hub as '${DOCKER_HUB_USER}'"
    echo ""
    echo "Please login first:"
    echo "  docker login -u ${DOCKER_HUB_USER}"
    echo ""
    read -p "Press Enter after logging in, or Ctrl+C to cancel..."
fi

echo "Step 1/2: Tagging image..."
docker tag "${LOCAL_IMAGE}" "${REMOTE_TAG}"
echo "✓ Image tagged as: ${REMOTE_TAG}"
echo ""

echo "Step 2/2: Pushing to Docker Hub..."
echo "This may take several minutes depending on your connection..."
docker push "${REMOTE_TAG}"

echo ""
echo "========================================"
echo "✅ Successfully pushed to Docker Hub!"
echo "========================================"
echo ""
echo "Your image is now available at:"
echo "  docker pull ${REMOTE_TAG}"
echo ""
echo "Or view it on Docker Hub:"
echo "  https://hub.docker.com/r/${DOCKER_HUB_USER}/${DOCKER_HUB_IMAGE}"
echo ""