#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-cospec-vllm}"
IMAGE="vllm/vllm-openai:v0.8.5"

# HF cache on /mnt/sdb to avoid filling root
HF_CACHE="${HF_CACHE:-/mnt/sdb/sjchoi/.cache/huggingface/hub}"

# Setup commands on first creation
read -r -d '' SETUP_COMMANDS << 'EOF' || true
set -e

# Extra dependencies (install before vllm since UltraDict needs Cython)
pip3 install Cython UltraDict pytest

# Overlay CoSpec source in editable mode, rebuilding C extensions
# against the container's PyTorch to avoid ABI mismatches
cd /workspace/vllm
pip3 install -e . --no-build-isolation

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
        --entrypoint /bin/bash \
        -v "$VLLM_SRC":/workspace/vllm \
        -v "$HF_CACHE":/root/.cache/huggingface/hub \
        -e HF_HOME=/root/.cache/huggingface \
        -w /workspace/vllm \
        "$IMAGE" \
        -c "$SETUP_COMMANDS"
fi
