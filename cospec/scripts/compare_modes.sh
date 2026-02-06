#!/bin/bash
# Compare AR vs Vanilla SD vs CoSpec performance
# Usage: docker exec -it -w /workspace cospec-vllm bash cospec/scripts/compare_modes.sh [num_prompts]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_PROMPTS="${1:-200}"
PORT=8000
MODEL="Qwen/Qwen3-8B"
DRAFT_MODEL="Qwen/Qwen3-0.6B"
GAMMA=5
GPU_MEM_UTIL=0.80
MAX_MODEL_LEN=8192
RESULTS_FILE="/tmp/compare_results.txt"

echo "Mode Comparison: AR vs Vanilla SD vs CoSpec" > "$RESULTS_FILE"
echo "Model: $MODEL | Draft: $DRAFT_MODEL | Gamma: $GAMMA | Prompts: $NUM_PROMPTS" >> "$RESULTS_FILE"
echo "============================================" >> "$RESULTS_FILE"

stop_mps() {
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 3
}

start_mps() {
    if ! pgrep -f "nvidia-cuda-mps-control" > /dev/null 2>&1; then
        export CUDA_VISIBLE_DEVICES=0
        nvidia-cuda-mps-control -d 2>/dev/null || true
        sleep 2
        echo "MPS started"
    fi
}

cleanup() {
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "multiprocessing" 2>/dev/null || true
    sleep 5
}

full_cleanup() {
    cleanup
    stop_mps
    sleep 2
}

wait_for_server() {
    local label=$1
    echo "Waiting for server ($label)..."
    for i in $(seq 1 80); do
        code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "Server ready after $((i*5))s"
            return 0
        fi
        sleep 5
    done
    echo "TIMEOUT waiting for server ($label)"
    return 1
}

run_benchmark() {
    local label=$1
    echo "Running benchmark ($label, $NUM_PROMPTS prompts)..."
    local OUTPUT
    OUTPUT=$(bash "$SCRIPT_DIR/client.sh" "$NUM_PROMPTS" 2>&1)

    local SUCCESS=$(echo "$OUTPUT" | grep "Successful" | awk '{print $NF}')
    local DURATION=$(echo "$OUTPUT" | grep "Benchmark duration" | awk '{print $NF}')
    local GEN_TOKENS=$(echo "$OUTPUT" | grep "Total generated" | awk '{print $NF}')
    local OUT_TPUT=$(echo "$OUTPUT" | grep "Output token throughput" | awk '{print $NF}')
    local TOT_TPUT=$(echo "$OUTPUT" | grep "Total Token throughput" | awk '{print $NF}')
    local MEAN_TTFT=$(echo "$OUTPUT" | grep "Mean TTFT" | awk '{print $NF}')
    local MED_TTFT=$(echo "$OUTPUT" | grep "Median TTFT" | awk '{print $NF}')
    local P99_TTFT=$(echo "$OUTPUT" | grep "P99 TTFT" | awk '{print $NF}')
    local MEAN_TPOT=$(echo "$OUTPUT" | grep "Mean TPOT" | awk '{print $NF}')
    local MED_TPOT=$(echo "$OUTPUT" | grep "Median TPOT" | awk '{print $NF}')
    local P99_TPOT=$(echo "$OUTPUT" | grep "P99 TPOT" | awk '{print $NF}')
    local MEAN_ITL=$(echo "$OUTPUT" | grep "Mean ITL" | awk '{print $NF}')

    local LINE="$label | success=$SUCCESS dur=${DURATION}s gen=$GEN_TOKENS out_tok/s=$OUT_TPUT tot_tok/s=$TOT_TPUT | TTFT(mean/med/p99)=${MEAN_TTFT}/${MED_TTFT}/${P99_TTFT}ms | TPOT(mean/med/p99)=${MEAN_TPOT}/${MED_TPOT}/${P99_TPOT}ms | ITL=${MEAN_ITL}ms"
    echo "$LINE"
    echo "$LINE" >> "$RESULTS_FILE"
}

export CUDA_MPS_PIPE_DIRECTORY="$(cd "$SCRIPT_DIR/../.." && pwd)/log/mps/nvidia-mps"

# ═══════════════════════════════════════
# 1. AR (Autoregressive, no speculation)
# ═══════════════════════════════════════
echo ""
echo "=========================================="
echo "1. AR (Autoregressive, no speculation)"
echo "=========================================="
full_cleanup

export CUDA_VISIBLE_DEVICES=0
export COSPEC=0
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1

# AR: no MPS needed
vllm serve "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    --seed 42 \
    --enable-chunked-prefill \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    > /tmp/server_ar.log 2>&1 &

if wait_for_server "AR"; then
    run_benchmark "AR"
fi
full_cleanup

# ═══════════════════════════════════════
# 2. Vanilla SD (regular speculative decoding)
# ═══════════════════════════════════════
echo ""
echo "=========================================="
echo "2. Vanilla SD (regular spec decode)"
echo "=========================================="

export COSPEC=0

# Vanilla SD: no MPS needed
vllm serve "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
    --seed 42 \
    --enable-chunked-prefill \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    > /tmp/server_vanilla_sd.log 2>&1 &

if wait_for_server "Vanilla SD"; then
    run_benchmark "Vanilla_SD"
fi
full_cleanup

# ═══════════════════════════════════════
# 3. CoSpec (SM ratio 0.7, default)
# ═══════════════════════════════════════
echo ""
echo "=========================================="
echo "3. CoSpec (SM ratio 0.7)"
echo "=========================================="

# CoSpec needs MPS
start_mps

export COSPEC=1
export COSPEC_LOG=0
export COSPEC_TARGET_SM_RATIO=0.7

vllm serve "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
    --seed 42 \
    --enable-chunked-prefill \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    > /tmp/server_cospec_07.log 2>&1 &

if wait_for_server "CoSpec 0.7"; then
    run_benchmark "CoSpec_0.7"
fi
full_cleanup

# ═══════════════════════════════════════
# 4. CoSpec (SM ratio 0.6)
# ═══════════════════════════════════════
echo ""
echo "=========================================="
echo "4. CoSpec (SM ratio 0.6)"
echo "=========================================="

start_mps

export COSPEC=1
export COSPEC_TARGET_SM_RATIO=0.6

vllm serve "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
    --seed 42 \
    --enable-chunked-prefill \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    > /tmp/server_cospec_06.log 2>&1 &

if wait_for_server "CoSpec 0.6"; then
    run_benchmark "CoSpec_0.6"
fi
full_cleanup

# ═══════════════════════════════════════
# 5. CoSpec (SM ratio 0.8)
# ═══════════════════════════════════════
echo ""
echo "=========================================="
echo "5. CoSpec (SM ratio 0.8)"
echo "=========================================="

start_mps

export COSPEC=1
export COSPEC_TARGET_SM_RATIO=0.8

vllm serve "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
    --seed 42 \
    --enable-chunked-prefill \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    > /tmp/server_cospec_08.log 2>&1 &

if wait_for_server "CoSpec 0.8"; then
    run_benchmark "CoSpec_0.8"
fi
full_cleanup

# ═══════════════════════════════════════
# Results Summary
# ═══════════════════════════════════════
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
cat "$RESULTS_FILE"
