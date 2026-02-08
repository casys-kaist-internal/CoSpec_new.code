#!/bin/bash
# nsys profile for CoSpec
# Usage: ./nsys_profile.sh [iteration_name]
# Runs inside the container (not via docker exec wrapper)
set -e

ITERATION="${1:-cospec_b32_g3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GAMMA="${GAMMA:-3}"
BATCH="${BATCH:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
NSYS_BIN="nsys"  # nsight-systems-2024.6.2 installed via apt
NSYS_OUTPUT="/workspace/nsys_${ITERATION}"
# Delay: seconds to wait after launch before capture starts (skip model loading)
NSYS_DELAY="${NSYS_DELAY:-90}"
# Duration: seconds of capture
NSYS_DURATION="${NSYS_DURATION:-30}"

echo "=== nsys Profile: $ITERATION ==="
echo "  Model:        $MODEL"
echo "  Draft:        $DRAFT_MODEL"
echo "  Gamma:        $GAMMA"
echo "  Batch:        $BATCH (max-num-seqs)"
echo "  Num prompts:  $NUM_PROMPTS"
echo "  Output len:   $OUTPUT_LEN"
echo "  nsys delay:   ${NSYS_DELAY}s"
echo "  nsys duration:${NSYS_DURATION}s"
echo "  nsys output:  ${NSYS_OUTPUT}"

# Kill stale processes
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "python.*vllm" 2>/dev/null || true
sleep 2

# Check and start MPS
if ! pgrep -f "nvidia-cuda-mps-control" > /dev/null 2>&1; then
    echo "MPS not running. Starting MPS..."
    bash "$SCRIPT_DIR/start_mps.sh" || true
    sleep 2
fi

export CUDA_MPS_PIPE_DIRECTORY="${PROJECT_ROOT}/log/mps/nvidia-mps"
export CUDA_VISIBLE_DEVICES=0
export COSPEC=1
export COSPEC_LOG=1
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
# Do NOT set COSPEC_PROFILE (that's the pytorch profiler, conflicts with nsys)

# Remove old output
rm -f "${NSYS_OUTPUT}".* 2>/dev/null

# Start server under nsys
# -s none: skip CPU sampling (needs privileged), just trace CUDA/NVTX
# -t cuda,nvtx,osrt: CUDA API + NVTX annotations + OS runtime
# -y DELAY: seconds to wait before capture starts (skip model loading)
# -d DURATION: seconds of capture
echo "Starting server under nsys (delay=${NSYS_DELAY}s, capture=${NSYS_DURATION}s)..."
"$NSYS_BIN" profile \
    -s none \
    -t cuda,nvtx,osrt \
    -y "$NSYS_DELAY" \
    -d "$NSYS_DURATION" \
    -o "$NSYS_OUTPUT" \
    -f true \
    --export=sqlite \
    -- \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
        --seed 42 \
        --enable-chunked-prefill \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-num-seqs "$BATCH" \
        --disable-log-requests \
        --disable-frontend-multiprocessing \
    > "server_nsys_${ITERATION}.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died! Last 50 lines of log:"
        tail -50 "server_nsys_${ITERATION}.log"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server not ready after 180s"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run benchmark — send enough prompts to fill the batch
DATASET_PATH="$SCRIPT_DIR/../data/ShareGPT_V3_unfiltered_cleaned_split.json"
echo "Running benchmark: $NUM_PROMPTS prompts, output_len=$OUTPUT_LEN, request_rate=inf..."
vllm bench serve \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --sharegpt-output-len "$OUTPUT_LEN" \
    --ignore-eos \
    --request-rate inf \
    --seed 42 \
    2>&1 | tee "client_nsys_${ITERATION}.log"

echo "Benchmark complete."
echo "Waiting for nsys to finish (delay=${NSYS_DELAY}s + duration=${NSYS_DURATION}s + conversion)..."
echo "Do NOT interrupt — nsys needs to finalize the report."

# Let nsys finish naturally. After delay+duration, nsys:
# 1. Stops capture
# 2. Sends SIGTERM to server (--kill=sigterm default)
# 3. Collects data from qdstrm
# 4. Converts qdstrm → nsys-rep
# This can take several minutes for large traces.
TIMEOUT=600
for i in $(seq 1 $TIMEOUT); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "nsys exited after waiting ${i}s"
        break
    fi
    if [ $((i % 30)) -eq 0 ]; then
        SIZE=$(stat -c%s "${NSYS_OUTPUT}.nsys-rep" 2>/dev/null || echo 0)
        echo "  Waiting... ${i}s (nsys-rep: ${SIZE} bytes)"
    fi
    sleep 1
done

if kill -0 $SERVER_PID 2>/dev/null; then
    echo "nsys still running after ${TIMEOUT}s, force killing..."
    kill -9 $SERVER_PID 2>/dev/null || true
fi
wait $SERVER_PID 2>/dev/null || true

# Check outputs
echo ""
echo "=== Output files ==="
ls -lh ${NSYS_OUTPUT}.* 2>/dev/null || echo "WARNING: No output files found"
echo ""
echo "=== nsys profile $ITERATION complete ==="
echo "View with: nsys-ui ${NSYS_OUTPUT}.nsys-rep"
echo "Or query:  sqlite3 ${NSYS_OUTPUT}.sqlite"
