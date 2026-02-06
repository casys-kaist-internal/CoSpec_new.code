#!/bin/bash
# Benchmark a single mode. Usage: bash bench_one.sh <mode> [num_prompts]
# Modes: ar, vanilla_sd, cospec_0.7, cospec_0.6, cospec_0.8
set -e

MODE="${1:?Usage: bench_one.sh <mode> [num_prompts]}"
NUM_PROMPTS="${2:-200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000
MODEL="Qwen/Qwen3-8B"
DRAFT_MODEL="Qwen/Qwen3-0.6B"
GAMMA=5
GPU_MEM_UTIL=0.80
MAX_MODEL_LEN=8192

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export CUDA_MPS_PIPE_DIRECTORY="$(cd "$SCRIPT_DIR/../.." && pwd)/log/mps/nvidia-mps"

echo "=== Benchmarking mode: $MODE (${NUM_PROMPTS} prompts) ==="

case "$MODE" in
    ar)
        export COSPEC=0
        vllm serve "$MODEL" \
            --host 0.0.0.0 --port $PORT --seed 42 \
            --enable-chunked-prefill \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --disable-log-requests \
            --disable-frontend-multiprocessing \
            > /tmp/server_bench.log 2>&1 &
        ;;
    vanilla_sd)
        export COSPEC=0
        vllm serve "$MODEL" \
            --host 0.0.0.0 --port $PORT --seed 42 \
            --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
            --enable-chunked-prefill \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --disable-log-requests \
            --disable-frontend-multiprocessing \
            > /tmp/server_bench.log 2>&1 &
        ;;
    cospec_*)
        RATIO="${MODE#cospec_}"
        export COSPEC=1
        export COSPEC_LOG=0
        export COSPEC_TARGET_SM_RATIO=$RATIO
        # Ensure MPS is running
        if ! pgrep -f "nvidia-cuda-mps-control" > /dev/null 2>&1; then
            nvidia-cuda-mps-control -d 2>/dev/null || true
            sleep 2
            echo "MPS started"
        fi
        vllm serve "$MODEL" \
            --host 0.0.0.0 --port $PORT --seed 42 \
            --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
            --enable-chunked-prefill \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --disable-log-requests \
            --disable-frontend-multiprocessing \
            > /tmp/server_bench.log 2>&1 &
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid: ar, vanilla_sd, cospec_0.7, cospec_0.6, cospec_0.8"
        exit 1
        ;;
esac

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 120); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died. Check /tmp/server_bench.log"
        tail -20 /tmp/server_bench.log
        exit 1
    fi
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
        echo "Server ready after $((i*5))s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "TIMEOUT after 600s"
        kill -9 $SERVER_PID 2>/dev/null
        tail -20 /tmp/server_bench.log
        exit 1
    fi
    sleep 5
done

# Run benchmark
echo "Running benchmark..."
bash "$SCRIPT_DIR/client.sh" "$NUM_PROMPTS" 2>&1

# Kill server
echo ""
echo "Cleaning up..."
kill -9 $SERVER_PID 2>/dev/null || true
pkill -9 -f "multiprocessing" 2>/dev/null || true
sleep 2
echo "Done."
