#!/bin/bash
# CoSpec Installation Script
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_NAME="cospec-vllm"
BASE_IMAGE="vllm/vllm-openai:v0.8.5"

echo "=== CoSpec Installation ==="
echo ""

# 1. Pull base image
echo "[1/3] Pulling base image ..."
if docker image inspect "$BASE_IMAGE" &>/dev/null; then
    echo "    Image already exists"
else
    docker pull "$BASE_IMAGE"
fi

# 2. Launch container, install dependencies, then drop into bash
echo "[2/2] Launching container ..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi
exec docker run -it \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v "$REPO_DIR:/workspace" \
    -e VLLM_USE_V1=0 \
    -w /workspace \
    --entrypoint bash \
    "$BASE_IMAGE" \
    -c "pip install UltraDict scikit-learn matplotlib tqdm && pip install -e . && exec bash"
