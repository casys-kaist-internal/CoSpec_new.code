#!/bin/bash
# CoSpec server with nsys profiling
# Usage: ./server_nsys.sh [model] [draft_model]
# Output: cospec_profile.nsys-rep (nsys report), server.log (server log)
#
# After profiling, copy the .nsys-rep file to your local machine and open
# with Nsight Systems GUI, or use:
#   nsys stats cospec_profile.nsys-rep

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${1:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${2:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GAMMA="${GAMMA:-5}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
NSYS_OUTPUT="${NSYS_OUTPUT:-cospec_profile}"
LOG_FILE="server.log"

echo "Starting CoSpec server with nsys profiling..."
echo "  Target model: $MODEL"
echo "  Draft model:  $DRAFT_MODEL"
echo "  Gamma:        $GAMMA"
echo "  Port:         $PORT"
echo "  GPU mem util: $GPU_MEM_UTIL"
echo "  nsys output:  ${NSYS_OUTPUT}.nsys-rep"
echo "  Log:          $LOG_FILE"

# Check and start MPS if needed
if ! pgrep -f "nvidia-cuda-mps-control" > /dev/null 2>&1; then
    echo "MPS not running. Starting MPS..."
    bash "$SCRIPT_DIR/start_mps.sh" || true
    sleep 2
fi

# Set MPS pipe directory
export CUDA_MPS_PIPE_DIRECTORY="${PROJECT_ROOT}/log/mps/nvidia-mps"

export CUDA_VISIBLE_DEVICES=0
export COSPEC=1
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1

NSYS="/opt/nvidia/nsight-compute/2024.1.0/host/target-linux-x64/nsys"
if [ ! -f "$NSYS" ]; then
    NSYS="$(which nsys 2>/dev/null || echo nsys)"
fi

exec "$NSYS" profile \
    -t cuda,nvtx,osrt \
    -o "$NSYS_OUTPUT" \
    --force-overwrite=true \
    --cuda-memory-usage=true \
    --wait=primary \
    -s none \
    vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
        --seed 42 \
        --enable-chunked-prefill \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --disable-log-requests \
        --disable-frontend-multiprocessing
