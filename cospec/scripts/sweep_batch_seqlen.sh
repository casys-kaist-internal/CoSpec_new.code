#!/bin/bash
# CoSpec sweep: batch size, input/output length, acceptance rate, draft size (gamma)
#
# Sweeps:
#   1. Batch:      batch ∈ {1,4,8,16,32,64,128}, input=512, output=512, gamma=5
#   2. Input len:  input ∈ {128,256,512,1024,2048}, batch=32, output=512, gamma=5
#   3. Output len: output ∈ {128,256,512,1024,2048}, batch=32, input=512, gamma=5
#   4. Accept rate: rate ∈ {0.9,0.8,0.7,0.6}, batch=32, input=512, output=512, gamma=5 (CoSpec only)
#   5. Draft size: gamma ∈ {1,3,5,7}, batch=32, input=512, output=512 (Vanilla SD & CoSpec)
#
# Modes: AR, Vanilla SD, CoSpec (full-GPU, no SM partition)
# Output: TSV file
#
# Usage: ./sweep_batch_seqlen.sh [output_file]
#        SWEEP=batch,gamma ./sweep_batch_seqlen.sh   # run subset of sweeps

set -uo pipefail
# Note: no set -e — we handle errors per run to continue the sweep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Config ----
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"   # seconds to wait for server ready
COOLDOWN="${COOLDOWN:-5}"             # seconds between runs
OUTPUT_FILE="${1:-sweep_results.tsv}"

# Which sweeps to run (comma-separated, default: all)
SWEEPS="${SWEEP:-batch,input_len,output_len,accept_rate,gamma}"

# ---- Helpers ----

log() { echo "[sweep $(date +%H:%M:%S)] $*"; }

kill_server() {
    # Kill any vllm serve processes
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 1
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    # Force kill stragglers
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
    # Args: mode max_num_seqs gamma [extra_env...]
    local mode="$1"
    local max_num_seqs="$2"
    local gamma="$3"
    shift 3
    local extra_env=("$@")

    kill_server

    # Base env
    local env_vars=(
        CUDA_VISIBLE_DEVICES=0
        VLLM_USE_V1=0
        PYTHONUNBUFFERED=1
        HF_HUB_OFFLINE=1
    )
    env_vars+=("${extra_env[@]}")

    # Build server args
    local server_args=(
        "$MODEL"
        --host 0.0.0.0
        --port "$PORT"
        --seed 42
        --enable-chunked-prefill
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-seqs "$max_num_seqs"
        --disable-log-requests
        --disable-frontend-multiprocessing
    )

    case "$mode" in
        ar)
            env_vars+=(COSPEC=0)
            # No speculative config for AR
            ;;
        vanilla_sd)
            env_vars+=(COSPEC=0)
            server_args+=(--speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $gamma}")
            ;;
        cospec)
            env_vars+=(COSPEC=1 COSPEC_SM_PARTITION=0 COSPEC_LOG=0)
            server_args+=(--speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $gamma}")
            ;;
        *)
            log "ERROR: Unknown mode $mode"
            return 1
            ;;
    esac

    log "Starting server: mode=$mode max_num_seqs=$max_num_seqs gamma=$gamma env=[${extra_env[*]:-}]"

    # Launch server in background
    env "${env_vars[@]}" vllm serve "${server_args[@]}" > /tmp/sweep_server.log 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        log "Server log tail:"
        tail -30 /tmp/sweep_server.log
        kill_server
        return 1
    fi
    log "Server ready (PID=$SERVER_PID)"
}

run_bench() {
    # Args: label num_prompts input_len output_len
    # Writes result to /tmp/sweep_result.txt (out_tok_s duration_s)
    local label="$1"
    local num_prompts="$2"
    local input_len="$3"
    local output_len="$4"

    log "  Bench: $label  prompts=$num_prompts  in=$input_len  out=$output_len"

    local bench_log="/tmp/sweep_bench_${label}.log"

    vllm bench serve \
        --base-url "http://localhost:${PORT}" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio 0.0 \
        --num-prompts "$num_prompts" \
        --ignore-eos \
        --request-rate inf \
        --seed 42 \
        > "$bench_log" 2>&1

    # Parse output token throughput and duration from benchmark output
    # Format: "Output token throughput (tok/s):         1234.56"
    #         "Benchmark duration (s):                  12.34"
    local out_tok_s
    out_tok_s=$(grep -oP 'Output token throughput[^:]*:\s+\K[0-9.]+' "$bench_log" || echo "0")
    local duration_s
    duration_s=$(grep -oP 'Benchmark duration[^:]*:\s+\K[0-9.]+' "$bench_log" || echo "0")

    echo "$out_tok_s $duration_s" > /tmp/sweep_result.txt
}

run_one() {
    # Args: sweep mode batch_size input_len output_len gamma accept_rate
    local sweep="$1"
    local mode="$2"
    local batch_size="$3"
    local input_len="$4"
    local output_len="$5"
    local gamma="$6"
    local accept_rate="$7"

    local label="${sweep}_${mode}_b${batch_size}_i${input_len}_o${output_len}_g${gamma}_a${accept_rate}"

    # Build extra env for server
    local extra_env=()
    if [ "$accept_rate" != "-1" ]; then
        extra_env+=(COSPEC_ACCEPT_RATE="$accept_rate")
    fi

    # Start server with appropriate config
    local effective_gamma="$gamma"
    if [ "$mode" = "ar" ]; then
        effective_gamma=5  # doesn't matter for AR, but server needs a value
    fi

    if ! start_server "$mode" "$batch_size" "$effective_gamma" "${extra_env[@]}"; then
        log "  FAILED to start server for $label — skipping"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$sweep" "$mode" "$batch_size" "$input_len" "$output_len" \
            "$gamma" "$accept_rate" "FAIL" "0" \
            >> "$OUTPUT_FILE"
        kill_server
        sleep "$COOLDOWN"
        return 0
    fi

    # Run benchmark — num_prompts = batch_size * 4 to ensure steady state
    # (at least 4x batch worth of prompts, minimum 16)
    local num_prompts=$((batch_size * 4))
    if [ "$num_prompts" -lt 16 ]; then
        num_prompts=16
    fi
    # Cap at 256 to keep sweep time reasonable
    if [ "$num_prompts" -gt 256 ]; then
        num_prompts=256
    fi

    run_bench "$label" "$num_prompts" "$input_len" "$output_len"
    local out_tok_s duration_s
    out_tok_s=$(awk '{print $1}' /tmp/sweep_result.txt)
    duration_s=$(awk '{print $2}' /tmp/sweep_result.txt)

    # Append to TSV
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sweep" "$mode" "$batch_size" "$input_len" "$output_len" \
        "$gamma" "$accept_rate" "$out_tok_s" "$duration_s" \
        >> "$OUTPUT_FILE"

    log "  Result: ${out_tok_s} tok/s (${duration_s}s)"

    kill_server
    sleep "$COOLDOWN"
}

# ---- Main ----

# Write TSV header
echo -e "sweep\tmode\tbatch_size\tinput_len\toutput_len\tgamma\taccept_rate\tout_tok_s\tduration_s" > "$OUTPUT_FILE"

log "Starting sweep → $OUTPUT_FILE"
log "Model: $MODEL  Draft: $DRAFT_MODEL"
log "Sweeps: $SWEEPS"
log ""

# ==== 1. Batch size sweep ====
if echo "$SWEEPS" | grep -q "batch"; then
    log "=== Sweep 1: Batch size (input=512, output=512, gamma=5) ==="
    for batch in 1 4 8 16 32 64 128; do
        for mode in ar vanilla_sd cospec; do
            run_one "batch" "$mode" "$batch" 512 512 5 "-1"
        done
    done
fi

# ==== 2. Input length sweep ====
if echo "$SWEEPS" | grep -q "input_len"; then
    log "=== Sweep 2: Input length (batch=32, output=512, gamma=5) ==="
    for input_len in 128 256 512 1024 2048; do
        for mode in ar vanilla_sd cospec; do
            run_one "input_len" "$mode" 32 "$input_len" 512 5 "-1"
        done
    done
fi

# ==== 3. Output length sweep ====
if echo "$SWEEPS" | grep -q "output_len"; then
    log "=== Sweep 3: Output length (batch=32, input=512, gamma=5) ==="
    for output_len in 128 256 512 1024 2048; do
        for mode in ar vanilla_sd cospec; do
            run_one "output_len" "$mode" 32 512 "$output_len" 5 "-1"
        done
    done
fi

# ==== 4. Acceptance rate sweep (CoSpec only) ====
if echo "$SWEEPS" | grep -q "accept_rate"; then
    log "=== Sweep 4: Acceptance rate (batch=32, input=512, output=512, gamma=5, CoSpec only) ==="
    for rate in 0.9 0.8 0.7 0.6; do
        run_one "accept_rate" "cospec" 32 512 512 5 "$rate"
    done
fi

# ==== 5. Draft size (gamma) sweep ====
if echo "$SWEEPS" | grep -q "gamma"; then
    log "=== Sweep 5: Draft size/gamma (batch=32, input=512, output=512) ==="
    for gamma in 1 3 5 7; do
        for mode in ar vanilla_sd cospec; do
            # AR doesn't use gamma, only run once (at gamma=1)
            if [ "$mode" = "ar" ] && [ "$gamma" != "1" ]; then
                continue
            fi
            run_one "gamma" "$mode" 32 512 512 "$gamma" "-1"
        done
    done
fi

kill_server

log ""
log "=== Sweep complete ==="
log "Results: $OUTPUT_FILE"
log ""
column -t -s$'\t' "$OUTPUT_FILE"
