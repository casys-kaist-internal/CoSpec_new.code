#!/bin/bash
# Launch CoSpec Docker container with GPU access.
# Runs install.sh inside the container on first creation.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-cospec-vllm}"
IMAGE="vllm/vllm-openai:v0.8.5"

HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface/hub}"

read -r -d '' SETUP_COMMANDS << 'EOF' || true
set -e
cd /workspace
bash cospec/scripts/install.sh
exec bash
EOF

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Reusing existing container: $CONTAINER_NAME"
    docker start "$CONTAINER_NAME" 2>/dev/null || true
    docker exec -it -w /workspace "$CONTAINER_NAME" bash
else
    echo "Creating new container: $CONTAINER_NAME"
    docker run -it \
        --gpus all \
        --name "$CONTAINER_NAME" \
        --shm-size=16g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --entrypoint /bin/bash \
        -v "$PROJECT_ROOT":/workspace \
        -v "$HF_CACHE":/root/.cache/huggingface/hub \
        -e HF_HOME=/root/.cache/huggingface \
        -w /workspace \
        "$IMAGE" \
        -c "$SETUP_COMMANDS"
fi
