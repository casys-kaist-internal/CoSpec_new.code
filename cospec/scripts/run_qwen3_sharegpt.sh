#!/bin/bash
# CoSpec benchmark: Qwen3-8B (target) + Qwen3-0.6B (draft)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH="$PROJECT_ROOT/cospec/benchmark/bench_serving.py"

TARGET_MODEL="Qwen/Qwen3-8B"
DRAFT_MODEL="Qwen/Qwen3-0.6B"
NUM_SPEC_TOKENS=5
PORT=8100
GPU_MEM_UTIL=0.85
NUM_PROMPTS=200
REQUEST_RATE=4
DATASET="${1:-sharegpt}"  # sharegpt, random, sonnet, burstgpt

# Start MPS if not running
if ! pgrep -f nvidia-cuda-mps-control > /dev/null 2>&1; then
    echo "Starting MPS..."
    bash "$SCRIPT_DIR/start_mps.sh"
    sleep 2
fi

export CUDA_MPS_PIPE_DIRECTORY="${PROJECT_ROOT}/log/mps/nvidia-mps"
export COSPEC=1
# Use default attention backend (FLASH_ATTN) for max performance

# Download ShareGPT if needed
SHAREGPT_PATH="${PROJECT_ROOT}/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ "$DATASET" = "sharegpt" ] && [ ! -f "$SHAREGPT_PATH" ]; then
    echo "Downloading ShareGPT dataset..."
    curl -sL 'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json' \
        -o "$SHAREGPT_PATH"
fi

echo "=== CoSpec: $TARGET_MODEL + $DRAFT_MODEL ==="
echo "Dataset: $DATASET | Spec tokens: $NUM_SPEC_TOKENS | Port: $PORT"

# Launch server
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port $PORT \
    --model $TARGET_MODEL \
    --seed 42 \
    --enforce-eager \
    --enable-chunked-prefill \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --disable-log-requests \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

# Wait for server (check /v1/models which only works after model is loaded)
echo "Waiting for server to load model..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/v1/models | grep -q "$TARGET_MODEL" 2>/dev/null; then
        echo "Server ready."
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died. Check logs."
        exit 1
    fi
    sleep 2
done

# Run benchmark
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_ROOT}/results/qwen3_cospec_${DATASET}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

BENCH_ARGS=(
    --model "$TARGET_MODEL"
    --port "$PORT"
    --dataset "$DATASET"
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$REQUEST_RATE"
    --save-result
    --result-dir "$RESULTS_DIR"
)

if [ "$DATASET" = "sharegpt" ]; then
    BENCH_ARGS+=(--dataset-path "$SHAREGPT_PATH")
elif [ "$DATASET" = "random" ]; then
    BENCH_ARGS+=(--random-input-len 256 --random-output-len 128)
fi

echo "=== Running $DATASET benchmark ==="
python3 "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "$RESULTS_DIR/output.txt"
echo "Done. Results: $RESULTS_DIR/"
