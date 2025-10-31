#!/bin/bash
# Stop and cleanup the test vLLM container

CONTAINER_NAME="vllm-omega-test"

echo "Stopping vLLM test container..."

if docker ps -a | grep -q "${CONTAINER_NAME}"; then
    docker stop "${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}"
    echo "✓ Container stopped and removed"
else
    echo "⚠️  No running container found with name: ${CONTAINER_NAME}"
fi