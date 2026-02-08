#!/bin/bash
# Quick sweep of just the accept_rate dimension
# Appends to existing sweep_results.tsv

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
COOLDOWN="${COOLDOWN:-5}"
OUTPUT_FILE="${1:-/workspace/sweep_results.tsv}"

log() { echo "[sweep $(date +%H:%M:%S)] $*"; }

kill_server() {
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 1
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
}

wait_for_server() {
    local timeout=$WAIT_TIMEOUT
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log "ERROR: Server did not start within ${timeout}s"
    return 1
}

start_server() {
    local accept_rate="$1"
    kill_server

    local env_vars=(
        CUDA_VISIBLE_DEVICES=0
        VLLM_USE_V1=0
        PYTHONUNBUFFERED=1
        HF_HUB_OFFLINE=1
        COSPEC=1
        COSPEC_SM_PARTITION=0
        COSPEC_LOG=0
    )
    if [ "$accept_rate" != "-1" ]; then
        env_vars+=(COSPEC_ACCEPT_RATE="$accept_rate")
    fi

    local server_args=(
        "$MODEL"
        --host 0.0.0.0
        --port "$PORT"
        --seed 42
        --enable-chunked-prefill
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-seqs 32
        --disable-log-requests
        --disable-frontend-multiprocessing
        --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5}"
    )

    log "Starting CoSpec server: accept_rate=$accept_rate"
    env "${env_vars[@]}" vllm serve "${server_args[@]}" > /tmp/sweep_server.log 2>&1 &

    if ! wait_for_server; then
        log "Server log tail:"
        tail -30 /tmp/sweep_server.log
        kill_server
        return 1
    fi
    log "Server ready"
}

run_bench() {
    local label="$1"
    log "  Bench: $label"
    local bench_log="/tmp/sweep_bench_${label}.log"

    vllm bench serve \
        --base-url "http://localhost:${PORT}" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len 512 \
        --random-output-len 512 \
        --random-range-ratio 0.0 \
        --num-prompts 128 \
        --ignore-eos \
        --request-rate inf \
        --seed 42 \
        > "$bench_log" 2>&1

    local out_tok_s
    out_tok_s=$(grep -oP 'Output token throughput[^:]*:\s+\K[0-9.]+' "$bench_log" || echo "0")
    local duration_s
    duration_s=$(grep -oP 'Benchmark duration[^:]*:\s+\K[0-9.]+' "$bench_log" || echo "0")

    echo "$out_tok_s $duration_s" > /tmp/sweep_result.txt
}

# Remove old accept_rate lines from results
if [ -f "$OUTPUT_FILE" ]; then
    grep -v "^accept_rate" "$OUTPUT_FILE" > /tmp/sweep_results_clean.tsv
    cp /tmp/sweep_results_clean.tsv "$OUTPUT_FILE"
fi

log "=== Accept rate sweep (batch=32, input=512, output=512, gamma=5, CoSpec) ==="

for rate in 1.0 0.9 0.8 0.7 0.6 0.5; do
    if ! start_server "$rate"; then
        log "FAILED to start server for rate=$rate"
        printf 'accept_rate\tcospec\t32\t512\t512\t5\t%s\tFAIL\t0\n' "$rate" >> "$OUTPUT_FILE"
        continue
    fi

    run_bench "accept_rate_${rate}"
    local_tok_s=$(awk '{print $1}' /tmp/sweep_result.txt)
    local_dur=$(awk '{print $2}' /tmp/sweep_result.txt)

    printf 'accept_rate\tcospec\t32\t512\t512\t5\t%s\t%s\t%s\n' \
        "$rate" "$local_tok_s" "$local_dur" >> "$OUTPUT_FILE"

    log "  Result: ${local_tok_s} tok/s (${local_dur}s)"
    kill_server
    sleep "$COOLDOWN"
done

kill_server
log "=== Accept rate sweep complete ==="
grep "accept_rate" "$OUTPUT_FILE"
