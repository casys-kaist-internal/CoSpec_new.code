#!/bin/bash
# Stop NVIDIA MPS daemon.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPSDIR="${PROJECT_ROOT}/log/mps"

export CUDA_MPS_PIPE_DIRECTORY="${MPSDIR}/nvidia-mps"
echo quit | nvidia-cuda-mps-control
echo "MPS stopped."
