#!/bin/bash
# Stop CUDA MPS daemon (works inside container or from host)
set -e

CONTAINER_NAME="cospec-vllm"

if command -v docker &>/dev/null && [ ! -f /.dockerenv ]; then
    PREFIX="docker exec $CONTAINER_NAME"
else
    PREFIX=""
fi

$PREFIX bash -c "echo quit | nvidia-cuda-mps-control 2>/dev/null || true"
echo "MPS stopped"
