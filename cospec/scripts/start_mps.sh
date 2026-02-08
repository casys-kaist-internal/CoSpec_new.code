#!/bin/bash
# Start CUDA MPS daemon (works inside container or from host)
set -e

CONTAINER_NAME="cospec-vllm"

if command -v docker &>/dev/null && [ ! -f /.dockerenv ]; then
    PREFIX="docker exec $CONTAINER_NAME"
else
    PREFIX=""
fi

$PREFIX bash -c "nvidia-cuda-mps-control -d 2>/dev/null || true"
if $PREFIX bash -c "echo get_default_active_thread_percentage | nvidia-cuda-mps-control" &>/dev/null; then
    echo "MPS running"
else
    echo "WARNING: MPS failed to start"
    exit 1
fi
