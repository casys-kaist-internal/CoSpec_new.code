#!/bin/bash
# SM Ratio Sweep: measure throughput at different target SM ratios
# Usage: docker exec -it -w /workspace cospec-vllm bash cospec/scripts/sm_ratio_sweep.sh
set -e

RATIOS="0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"
NUM_PROMPTS=200
PORT=8000
RESULTS_FILE="/tmp/sm_ratio_results.txt"

echo "SM Ratio Sweep" > "$RESULTS_FILE"
echo "==============" >> "$RESULTS_FILE"

for RATIO in $RATIOS; do
    echo ""
    echo "=========================================="
    echo "Testing SM ratio: $RATIO"
    echo "=========================================="

    # Kill any existing server
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "multiprocessing" 2>/dev/null || true
    sleep 5

    # Start server with this ratio
    export CUDA_VISIBLE_DEVICES=0
    export COSPEC=1
    export COSPEC_LOG=0
    export COSPEC_TARGET_SM_RATIO=$RATIO
    export VLLM_USE_V1=0
    export PYTHONUNBUFFERED=1

    bash cospec/scripts/server.sh > /tmp/server_sweep_${RATIO}.log 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server (ratio=$RATIO)..."
    for i in $(seq 1 60); do
        code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "Server ready after $((i*5))s"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "TIMEOUT waiting for server at ratio=$RATIO"
            kill -9 $SERVER_PID 2>/dev/null || true
            continue 2
        fi
        sleep 5
    done

    # Run benchmark
    echo "Running benchmark (ratio=$RATIO)..."
    OUTPUT=$(bash cospec/scripts/client.sh $NUM_PROMPTS 2>&1)

    # Extract key metrics
    OUT_TPUT=$(echo "$OUTPUT" | grep "Output token throughput" | awk '{print $NF}')
    TOT_TPUT=$(echo "$OUTPUT" | grep "Total Token throughput" | awk '{print $NF}')
    MEAN_TTFT=$(echo "$OUTPUT" | grep "Mean TTFT" | awk '{print $NF}')
    MEAN_TPOT=$(echo "$OUTPUT" | grep "Mean TPOT" | awk '{print $NF}')
    MEAN_ITL=$(echo "$OUTPUT" | grep "Mean ITL" | awk '{print $NF}')
    SUCCESS=$(echo "$OUTPUT" | grep "Successful" | awk '{print $NF}')
    GEN_TOKENS=$(echo "$OUTPUT" | grep "Total generated" | awk '{print $NF}')

    echo "RATIO=$RATIO | out_tok/s=$OUT_TPUT | tot_tok/s=$TOT_TPUT | ttft=$MEAN_TTFT | tpot=$MEAN_TPOT | itl=$MEAN_ITL | success=$SUCCESS | gen=$GEN_TOKENS"
    echo "RATIO=$RATIO | out_tok/s=$OUT_TPUT | tot_tok/s=$TOT_TPUT | ttft=$MEAN_TTFT | tpot=$MEAN_TPOT | itl=$MEAN_ITL | success=$SUCCESS | gen=$GEN_TOKENS" >> "$RESULTS_FILE"

    # Kill server
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "multiprocessing" 2>/dev/null || true
    sleep 3
done

echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
cat "$RESULTS_FILE"
