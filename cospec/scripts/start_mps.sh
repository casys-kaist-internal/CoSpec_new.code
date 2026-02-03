#!/bin/bash
# Start NVIDIA MPS daemon. Requires root or sufficient permissions.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPSDIR="${PROJECT_ROOT}/log/mps"

mkdir -p "${MPSDIR}/nvidia-mps"
mkdir -p "${MPSDIR}/nvidia-log"
chmod 777 "${MPSDIR}/nvidia-log"

export CUDA_MPS_PIPE_DIRECTORY="${MPSDIR}/nvidia-mps"
export CUDA_MPS_LOG_DIRECTORY="${MPSDIR}/nvidia-log"

nvidia-cuda-mps-control -d
echo "MPS started. Pipe: ${CUDA_MPS_PIPE_DIRECTORY}"
