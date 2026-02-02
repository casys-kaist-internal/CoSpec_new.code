#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-cospec-vllm}"
IMAGE="nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04"

# HF cache on /mnt/sdb to avoid filling root
HF_CACHE="${HF_CACHE:-/mnt/sdb/sjchoi/.cache/huggingface/hub}"

# Setup commands: install precompiled vllm wheel, then overlay source in editable mode
read -r -d '' SETUP_COMMANDS << 'EOF' || true
set -e
apt-get update
apt-get install -y python3 python3-pip git python-is-python3
pip3 install --upgrade pip

# Install precompiled vllm wheel (includes compiled C++/CUDA extensions + deps)
pip3 install vllm==0.8.5

# Re-install in editable mode, reusing precompiled extensions.
# --no-deps prevents pip from pulling a different torch build (CPU vs CUDA).
cd /workspace/vllm
VLLM_USE_PRECOMPILED=1 pip3 install -e . --no-deps

# Extra dependencies
pip3 install UltraDict pytest

echo "===== SETUP COMPLETE ====="
exec bash
EOF

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Reusing existing container: $CONTAINER_NAME"
    docker start "$CONTAINER_NAME" 2>/dev/null || true
    docker exec -it -w /workspace/vllm "$CONTAINER_NAME" bash
else
    echo "Creating new container: $CONTAINER_NAME"
    docker run -it \
        --gpus all \
        --name "$CONTAINER_NAME" \
        --shm-size=16g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$VLLM_SRC":/workspace/vllm \
        -v "$HF_CACHE":/root/.cache/huggingface/hub \
        -e HF_HOME=/root/.cache/huggingface \
        -w /workspace/vllm \
        "$IMAGE" \
        bash -c "$SETUP_COMMANDS"
fi
