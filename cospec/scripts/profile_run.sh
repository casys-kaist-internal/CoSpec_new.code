#!/bin/bash
# Profile run: starts server with profiling, waits for ready, runs client, collects trace
# Usage: ./profile_run.sh [iteration_name]
set -e

ITERATION="${1:-baseline}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GAMMA="${GAMMA:-5}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
PROFILE_SKIP="${PROFILE_SKIP:-15}"
PROFILE_STEPS="${PROFILE_STEPS:-80}"
TRACE_OUTPUT="/workspace/cospec_trace_${ITERATION}.json"
SM_RATIO="${SM_RATIO:-0.6}"

echo "=== CoSpec Profile Run: $ITERATION ==="
echo "  SM Ratio:       $SM_RATIO"
echo "  Profile skip:   $PROFILE_SKIP steps"
echo "  Profile steps:  $PROFILE_STEPS steps"
echo "  Trace output:   $TRACE_OUTPUT"
echo "  Num prompts:    $NUM_PROMPTS"

# Kill any stale vllm processes
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "python.*vllm" 2>/dev/null || true
sleep 2

# Check and start MPS if needed
if ! pgrep -f "nvidia-cuda-mps-control" > /dev/null 2>&1; then
    echo "MPS not running. Starting MPS..."
    bash "$SCRIPT_DIR/start_mps.sh" || true
    sleep 2
fi

export CUDA_MPS_PIPE_DIRECTORY="${PROJECT_ROOT}/log/mps/nvidia-mps"
export CUDA_VISIBLE_DEVICES=0
export COSPEC=1
export COSPEC_LOG=1
export COSPEC_PROFILE=1
export COSPEC_PROFILE_SKIP="$PROFILE_SKIP"
export COSPEC_PROFILE_STEPS="$PROFILE_STEPS"
export COSPEC_PROFILE_OUTPUT="$TRACE_OUTPUT"
export COSPEC_TARGET_SM_RATIO="$SM_RATIO"
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1

# Start server in background
echo "Starting server..."
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
    --seed 42 \
    --enable-chunked-prefill \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    > "server_profile_${ITERATION}.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died!"
        cat "server_profile_${ITERATION}.log" | tail -50
        exit 1
    fi
    sleep 1
done

# Check server is actually responding
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server not ready after 120s"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run benchmark
DATASET_PATH="$SCRIPT_DIR/../data/ShareGPT_V3_unfiltered_cleaned_split.json"
echo "Running benchmark with $NUM_PROMPTS prompts..."
vllm bench serve \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --sharegpt-output-len 128 \
    --ignore-eos \
    --request-rate inf \
    --seed 42 \
    2>&1 | tee "client_profile_${ITERATION}.log"

# Wait a bit for profiler to flush
sleep 5

# Kill server
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 2

# Check trace exists
if [ -f "$TRACE_OUTPUT" ]; then
    TRACE_SIZE=$(stat -c%s "$TRACE_OUTPUT" 2>/dev/null || echo 0)
    echo "Trace saved: $TRACE_OUTPUT (${TRACE_SIZE} bytes)"
else
    echo "WARNING: Trace file not found at $TRACE_OUTPUT"
fi

echo "=== Profile run $ITERATION complete ==="
