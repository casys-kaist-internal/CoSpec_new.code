#!/bin/bash
# CoSpec server script
# Usage: ./server.sh [model] [draft_model]
# Output: server.log

set -e

MODEL="${1:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${2:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
LOG_FILE="server.log"

echo "Starting CoSpec server..."
echo "  Target model: $MODEL"
echo "  Draft model:  $DRAFT_MODEL"
echo "  Port:         $PORT"
echo "  Log:          $LOG_FILE"

# Check MPS
if ! pgrep -f "nvidia-cuda-mps-control" > /dev/null 2>&1; then
    echo "WARNING: MPS not running. Start with: bash cospec/scripts/start_mps.sh"
fi

export COSPEC=1
export VLLM_USE_V1=0
export COSPEC_PROFILE=1

vllm serve "$MODEL" \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5}" \
    --port "$PORT" \
    2>&1 | tee "$LOG_FILE"
