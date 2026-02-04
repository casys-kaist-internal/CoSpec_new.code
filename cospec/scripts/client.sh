#!/bin/bash
# CoSpec benchmark client script
# Usage: ./client.sh [num_prompts] [output_len]
# Output: client.log

set -e

NUM_PROMPTS="${1:-100}"
OUTPUT_LEN="${2:-128}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
LOG_FILE="client.log"

echo "Running benchmark..."
echo "  Num prompts:  $NUM_PROMPTS"
echo "  Output len:   $OUTPUT_LEN"
echo "  Model:        $MODEL"
echo "  Server:       http://localhost:$PORT"
echo "  Log:          $LOG_FILE"

vllm bench serve \
    --backend openai-chat \
    --base-url "http://localhost:$PORT/v1" \
    --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len 128 \
    --random-output-len "$OUTPUT_LEN" \
    2>&1 | tee "$LOG_FILE"
