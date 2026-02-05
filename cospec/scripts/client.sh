#!/bin/bash
# CoSpec benchmark client script
# Usage: ./client.sh [num_prompts]
# Output: client.log

set -e

NUM_PROMPTS="${1:-100}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
# Dataset path - default to cospec/data/ directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/../data/ShareGPT_V3_unfiltered_cleaned_split.json}"
LOG_FILE="client.log"

echo "Running benchmark..."
echo "  Dataset:      sharegpt ($DATASET_PATH)"
echo "  Num prompts:  $NUM_PROMPTS"
echo "  Output len:   from dataset (ignore EOS)"
echo "  Model:        $MODEL"
echo "  Request rate: $REQUEST_RATE req/s"
echo "  Server:       http://localhost:$PORT"
echo "  Log:          $LOG_FILE"

vllm bench serve \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --ignore-eos \
    --request-rate "$REQUEST_RATE" \
    2>&1 | tee "$LOG_FILE"
