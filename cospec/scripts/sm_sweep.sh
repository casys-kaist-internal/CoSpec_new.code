#!/bin/bash
# SM ratio sweep: test different SM partition ratios vs full-GPU concurrent mode
# Runs inside the docker container

set -e

GAMMA="${GAMMA:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
PORT=8000
MODEL="Qwen/Qwen3-8B"
DRAFT_MODEL="Qwen/Qwen3-0.6B"
DATASET_PATH="/workspace/cospec/data/ShareGPT_V3_unfiltered_cleaned_split.json"
RESULTS_FILE="/workspace/sm_sweep_results.txt"

echo "=== SM Ratio Sweep (gamma=$GAMMA, $NUM_PROMPTS prompts) ===" | tee "$RESULTS_FILE"
echo "Format: Mode | SM_Ratio | Out_tok/s | Duration_s" | tee -a "$RESULTS_FILE"
echo "---" | tee -a "$RESULTS_FILE"

run_benchmark() {
    local label="$1"
    local sm_partition="$2"
    local sm_ratio="$3"
    local log_prefix="$4"

    echo "" | tee -a "$RESULTS_FILE"
    echo ">>> Running: $label (SM_PARTITION=$sm_partition, SM_RATIO=$sm_ratio)" | tee -a "$RESULTS_FILE"

    # Kill any stale processes
    for p in $(pgrep -f "python|vllm" 2>/dev/null); do kill -9 $p 2>/dev/null; done
    sleep 3

    # Start server
    CUDA_VISIBLE_DEVICES=0 \
    COSPEC=1 \
    COSPEC_LOG=0 \
    COSPEC_SM_PARTITION="$sm_partition" \
    COSPEC_TARGET_SM_RATIO="$sm_ratio" \
    VLLM_USE_V1=0 \
    PYTHONUNBUFFERED=1 \
    vllm serve "$MODEL" \
        --host 0.0.0.0 --port $PORT \
        --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $GAMMA}" \
        --seed 42 --enable-chunked-prefill \
        --gpu-memory-utilization 0.80 \
        --disable-log-requests \
        --disable-frontend-multiprocessing \
        > "/workspace/${log_prefix}_server.log" 2>&1 &
    local server_pid=$!

    # Wait for ready
    local ready=0
    for i in $(seq 1 150); do
        if grep -q "Application startup complete" "/workspace/${log_prefix}_server.log" 2>/dev/null; then
            echo "  Server ready after ${i}s"
            ready=1
            break
        fi
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "  ERROR: Server died!" | tee -a "$RESULTS_FILE"
            tail -10 "/workspace/${log_prefix}_server.log"
            return 1
        fi
        sleep 1
    done

    if [ $ready -eq 0 ]; then
        echo "  ERROR: Server timeout!" | tee -a "$RESULTS_FILE"
        kill $server_pid 2>/dev/null
        return 1
    fi

    # Run benchmark
    CUDA_VISIBLE_DEVICES=0 vllm bench serve \
        --base-url "http://localhost:$PORT" \
        --model "$MODEL" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --num-prompts $NUM_PROMPTS \
        --sharegpt-output-len 128 \
        --ignore-eos \
        --request-rate inf \
        --seed 42 \
        > "/workspace/${log_prefix}_client.log" 2>&1

    # Extract results
    local tok_s=$(grep "Output token throughput" "/workspace/${log_prefix}_client.log" | awk '{print $NF}')
    local duration=$(grep "Benchmark duration" "/workspace/${log_prefix}_client.log" | awk '{print $NF}')

    echo "  $label | SM=$sm_ratio | $tok_s tok/s | ${duration}s" | tee -a "$RESULTS_FILE"

    # Kill server
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    sleep 3
}

# 1. Full-GPU concurrent (no SM partition) — baseline
run_benchmark "CoSpec-FullGPU" 0 0.7 "sweep_fullgpu"

# 2. SM partition 0.6
run_benchmark "CoSpec-SM0.6" 1 0.6 "sweep_sm06"

# 3. SM partition 0.7
run_benchmark "CoSpec-SM0.7" 1 0.7 "sweep_sm07"

# 4. SM partition 0.8
run_benchmark "CoSpec-SM0.8" 1 0.8 "sweep_sm08"

# 5. SM partition 0.9
run_benchmark "CoSpec-SM0.9" 1 0.9 "sweep_sm09"

echo "" | tee -a "$RESULTS_FILE"
echo "=== Sweep Complete ===" | tee -a "$RESULTS_FILE"
cat "$RESULTS_FILE"
