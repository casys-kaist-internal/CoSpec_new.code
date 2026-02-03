#!/bin/bash
# =============================================================================
# CoSpec Comprehensive Evaluation Script
# =============================================================================
# Runs AR, Vanilla SD, and CoSpec configurations across multiple request rates
# following OSDI/SOSP evaluation standards.
#
# Usage:
#   ./run_evaluation.sh                    # Full evaluation
#   ./run_evaluation.sh --quick            # Quick test (fewer rates)
#   ./run_evaluation.sh --experiment 1     # Run specific experiment only
#   ./run_evaluation.sh --config cospec    # Run specific config only
#
# Required: ShareGPT dataset, NVIDIA MPS (for CoSpec)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_SCRIPT="$PROJECT_ROOT/cospec/benchmark/bench_serving.py"

# =============================================================================
# Configuration
# =============================================================================

# Models (Primary: Llama-3.1-8B/1B)
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-meta-llama/Llama-3.1-1B}"

# Speculative decoding settings
GAMMA=5  # num_speculative_tokens
DEFAULT_SM_RATIO=0.7

# Server settings
PORT=${PORT:-8100}
GPU_MEM_UTIL=0.85

# Benchmark settings
WARMUP_DURATION=120  # 2 minutes warmup
BENCHMARK_DURATION=300  # 5 minutes measurement
NUM_PROMPTS=500  # Used for warmup and duration-based cycling

# Request rates to sweep (requests per second)
REQUEST_RATES_FULL=(0.5 1 2 3 4 5 6 7 8 10)
REQUEST_RATES_QUICK=(1 2 4 6 8)
REQUEST_RATES=("${REQUEST_RATES_FULL[@]}")

# SM ratios for ablation (Experiment 3)
SM_RATIOS=(0.5 0.6 0.7 0.8 0.9)

# Gamma values for ablation (Experiment 4)
GAMMA_VALUES=(3 5 7)

# Number of repetitions per configuration
NUM_REPEATS=3

# Dataset
DATASET="sharegpt"
SHAREGPT_PATH="${SHAREGPT_PATH:-$PROJECT_ROOT/ShareGPT_V3_unfiltered_cleaned_split.json}"

# Results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/evaluation_${TIMESTAMP}}"

# =============================================================================
# Parse Arguments
# =============================================================================

QUICK_MODE=false
EXPERIMENT=""
CONFIG_FILTER=""
SKIP_WARMUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            REQUEST_RATES=("${REQUEST_RATES_QUICK[@]}")
            NUM_REPEATS=1
            WARMUP_DURATION=60
            BENCHMARK_DURATION=120
            shift
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILTER="$2"
            shift 2
            ;;
        --skip-warmup)
            SKIP_WARMUP=true
            shift
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --target-model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --draft-model)
            DRAFT_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$RESULTS_DIR"

# Log configuration
cat > "$RESULTS_DIR/config.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "target_model": "$TARGET_MODEL",
    "draft_model": "$DRAFT_MODEL",
    "gamma": $GAMMA,
    "default_sm_ratio": $DEFAULT_SM_RATIO,
    "warmup_duration": $WARMUP_DURATION,
    "benchmark_duration": $BENCHMARK_DURATION,
    "request_rates": [$(IFS=,; echo "${REQUEST_RATES[*]}")],
    "num_repeats": $NUM_REPEATS,
    "dataset": "$DATASET",
    "quick_mode": $QUICK_MODE
}
EOF

echo "=== CoSpec Evaluation ==="
echo "Target: $TARGET_MODEL"
echo "Draft: $DRAFT_MODEL"
echo "Results: $RESULTS_DIR"
echo "Request rates: ${REQUEST_RATES[*]}"
echo "Repeats: $NUM_REPEATS"
echo ""

# Download ShareGPT if needed
if [ ! -f "$SHAREGPT_PATH" ]; then
    echo "Downloading ShareGPT dataset..."
    curl -sL 'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json' \
        -o "$SHAREGPT_PATH"
fi

# CSV header
CSV_FILE="$RESULTS_DIR/results.csv"
echo "config,gamma,sm_ratio,request_rate,repeat,completed,failed,total_time_s,request_throughput,input_tok_throughput,output_tok_throughput,mean_ttft_ms,p50_ttft_ms,p90_ttft_ms,p99_ttft_ms,mean_tpot_ms,p50_tpot_ms,p90_tpot_ms,p99_tpot_ms,mean_itl_ms,p50_itl_ms,p90_itl_ms,p99_itl_ms,mean_e2e_ms,p50_e2e_ms,p90_e2e_ms,p99_e2e_ms" > "$CSV_FILE"

# =============================================================================
# Helper Functions
# =============================================================================

cleanup_server() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping server (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        sleep 3
    fi
}

wait_for_server() {
    local max_wait=300  # 5 minutes
    local start_time=$(date +%s)
    echo "Waiting for server to be ready..."

    while true; do
        if curl -s "http://localhost:$PORT/v1/models" | grep -q "$TARGET_MODEL" 2>/dev/null; then
            echo "Server ready."
            return 0
        fi

        local elapsed=$(($(date +%s) - start_time))
        if [ $elapsed -gt $max_wait ]; then
            echo "ERROR: Server failed to start within ${max_wait}s"
            return 1
        fi

        sleep 5
    done
}

start_server_ar() {
    # AR baseline: no speculative decoding
    echo "Starting AR server..."
    VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port "$PORT" \
        --model "$TARGET_MODEL" \
        --seed 42 \
        --enforce-eager \
        --enable-chunked-prefill \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --disable-log-requests \
        > "$RESULTS_DIR/server_ar.log" 2>&1 &

    echo $!
}

start_server_vanilla_sd() {
    local gamma=$1
    # Vanilla SD: speculative decoding without CoSpec
    echo "Starting Vanilla SD server (gamma=$gamma)..."
    VLLM_USE_V1=0 COSPEC=0 python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port "$PORT" \
        --model "$TARGET_MODEL" \
        --speculative-model "$DRAFT_MODEL" \
        --num-speculative-tokens "$gamma" \
        --seed 42 \
        --enforce-eager \
        --enable-chunked-prefill \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --disable-log-requests \
        > "$RESULTS_DIR/server_vanilla_sd_gamma${gamma}.log" 2>&1 &

    echo $!
}

start_server_cospec() {
    local gamma=$1
    local sm_ratio=$2
    # CoSpec: speculative decoding with SM partitioning
    echo "Starting CoSpec server (gamma=$gamma, sm_ratio=$sm_ratio)..."

    # Ensure MPS is running
    if ! pgrep -f nvidia-cuda-mps-control > /dev/null 2>&1; then
        echo "Starting MPS..."
        bash "$PROJECT_ROOT/cospec/scripts/start_mps.sh" || true
        sleep 2
    fi

    VLLM_USE_V1=0 COSPEC=1 COSPEC_TARGET_SM_RATIO=$sm_ratio python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port "$PORT" \
        --model "$TARGET_MODEL" \
        --speculative-model "$DRAFT_MODEL" \
        --num-speculative-tokens "$gamma" \
        --seed 42 \
        --enforce-eager \
        --enable-chunked-prefill \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --disable-log-requests \
        > "$RESULTS_DIR/server_cospec_gamma${gamma}_sm${sm_ratio}.log" 2>&1 &

    echo $!
}

run_benchmark() {
    local config=$1
    local gamma=$2
    local sm_ratio=$3
    local request_rate=$4
    local repeat=$5
    local output_file="$RESULTS_DIR/${config}_gamma${gamma}_sm${sm_ratio}_rate${request_rate}_r${repeat}.json"

    echo "  Running: config=$config, gamma=$gamma, sm_ratio=$sm_ratio, rate=$request_rate, repeat=$repeat"

    python "$BENCH_SCRIPT" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --model "$TARGET_MODEL" \
        --dataset "$DATASET" \
        --dataset-path "$SHAREGPT_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --duration "$BENCHMARK_DURATION" \
        --request-rate "$request_rate" \
        --temperature 0.0 \
        --seed "$repeat" \
        --save-result \
        --result-dir "$RESULTS_DIR" \
        --result-filename "$(basename "$output_file")" \
        > "$RESULTS_DIR/${config}_gamma${gamma}_sm${sm_ratio}_rate${request_rate}_r${repeat}.log" 2>&1

    # Parse results and append to CSV
    if [ -f "$output_file" ]; then
        python3 -c "
import json
with open('$output_file') as f:
    r = json.load(f)
# Extract percentiles
def get_p(data, p):
    import numpy as np
    return np.percentile(data, p) * 1000 if data else 0

ttfts = [x for x in r.get('ttfts', []) if x > 0] if 'ttfts' in r else []
tpots = r.get('tpots', []) if 'tpots' in r else []
itls = r.get('itls', []) if 'itls' in r else []
e2es = r.get('e2es', []) if 'e2es' in r else []

# Use mean values from output, calculate percentiles if raw data available
row = [
    '$config',
    '$gamma',
    '$sm_ratio',
    '$request_rate',
    '$repeat',
    r.get('completed', 0),
    r.get('failed', 0),
    r.get('total_time_s', 0),
    r.get('request_throughput', 0),
    r.get('input_tok_throughput', 0),
    r.get('output_tok_throughput', 0),
    r.get('mean_ttft_ms', 0),
    get_p(ttfts, 50) if ttfts else r.get('mean_ttft_ms', 0),
    get_p(ttfts, 90) if ttfts else 0,
    get_p(ttfts, 99) if ttfts else 0,
    r.get('mean_tpot_ms', 0),
    get_p(tpots, 50) if tpots else r.get('mean_tpot_ms', 0),
    get_p(tpots, 90) if tpots else 0,
    get_p(tpots, 99) if tpots else 0,
    r.get('mean_itl_ms', 0),
    get_p(itls, 50) if itls else r.get('mean_itl_ms', 0),
    get_p(itls, 90) if itls else 0,
    get_p(itls, 99) if itls else 0,
    r.get('mean_e2e_ms', 0),
    get_p(e2es, 50) if e2es else r.get('mean_e2e_ms', 0),
    get_p(e2es, 90) if e2es else 0,
    get_p(e2es, 99) if e2es else 0,
]
print(','.join(str(x) for x in row))
" >> "$CSV_FILE"
    else
        echo "WARNING: Output file not found: $output_file"
    fi
}

run_warmup() {
    local config=$1
    echo "  Running warmup for $config..."
    python "$BENCH_SCRIPT" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --model "$TARGET_MODEL" \
        --dataset "$DATASET" \
        --dataset-path "$SHAREGPT_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --duration "$WARMUP_DURATION" \
        --request-rate 2.0 \
        --temperature 0.0 \
        > "$RESULTS_DIR/${config}_warmup.log" 2>&1
}

# =============================================================================
# Experiment 1: Main Latency-Throughput Comparison
# =============================================================================

run_experiment_1() {
    echo ""
    echo "=== Experiment 1: Main Latency-Throughput Comparison ==="
    echo ""

    # --- AR Baseline ---
    if [ -z "$CONFIG_FILTER" ] || [ "$CONFIG_FILTER" = "ar" ]; then
        echo "[AR Baseline]"
        SERVER_PID=$(start_server_ar)
        wait_for_server || { cleanup_server $SERVER_PID; return 1; }

        if [ "$SKIP_WARMUP" = false ]; then
            run_warmup "ar"
        fi

        for rate in "${REQUEST_RATES[@]}"; do
            for repeat in $(seq 1 $NUM_REPEATS); do
                run_benchmark "ar" 0 "1.0" "$rate" "$repeat"
            done
        done

        cleanup_server $SERVER_PID
    fi

    # --- Vanilla SD ---
    if [ -z "$CONFIG_FILTER" ] || [ "$CONFIG_FILTER" = "vanilla_sd" ]; then
        echo "[Vanilla SD (gamma=$GAMMA)]"
        SERVER_PID=$(start_server_vanilla_sd $GAMMA)
        wait_for_server || { cleanup_server $SERVER_PID; return 1; }

        if [ "$SKIP_WARMUP" = false ]; then
            run_warmup "vanilla_sd"
        fi

        for rate in "${REQUEST_RATES[@]}"; do
            for repeat in $(seq 1 $NUM_REPEATS); do
                run_benchmark "vanilla_sd" "$GAMMA" "1.0" "$rate" "$repeat"
            done
        done

        cleanup_server $SERVER_PID
    fi

    # --- CoSpec ---
    if [ -z "$CONFIG_FILTER" ] || [ "$CONFIG_FILTER" = "cospec" ]; then
        echo "[CoSpec (gamma=$GAMMA, sm_ratio=$DEFAULT_SM_RATIO)]"
        SERVER_PID=$(start_server_cospec $GAMMA $DEFAULT_SM_RATIO)
        wait_for_server || { cleanup_server $SERVER_PID; return 1; }

        if [ "$SKIP_WARMUP" = false ]; then
            run_warmup "cospec"
        fi

        for rate in "${REQUEST_RATES[@]}"; do
            for repeat in $(seq 1 $NUM_REPEATS); do
                run_benchmark "cospec" "$GAMMA" "$DEFAULT_SM_RATIO" "$rate" "$repeat"
            done
        done

        cleanup_server $SERVER_PID
    fi
}

# =============================================================================
# Experiment 3: SM Ratio Ablation
# =============================================================================

run_experiment_3() {
    echo ""
    echo "=== Experiment 3: SM Ratio Ablation ==="
    echo ""

    local ablation_rate=4  # Fixed moderate load

    for sm_ratio in "${SM_RATIOS[@]}"; do
        echo "[CoSpec SM ratio=$sm_ratio]"
        SERVER_PID=$(start_server_cospec $GAMMA $sm_ratio)
        wait_for_server || { cleanup_server $SERVER_PID; return 1; }

        if [ "$SKIP_WARMUP" = false ]; then
            run_warmup "cospec_sm${sm_ratio}"
        fi

        for repeat in $(seq 1 $NUM_REPEATS); do
            run_benchmark "cospec_ablation" "$GAMMA" "$sm_ratio" "$ablation_rate" "$repeat"
        done

        cleanup_server $SERVER_PID
    done
}

# =============================================================================
# Experiment 4: Gamma (Speculation Length) Ablation
# =============================================================================

run_experiment_4() {
    echo ""
    echo "=== Experiment 4: Gamma Ablation ==="
    echo ""

    local ablation_rate=4  # Fixed moderate load

    for gamma in "${GAMMA_VALUES[@]}"; do
        # Vanilla SD with this gamma
        echo "[Vanilla SD gamma=$gamma]"
        SERVER_PID=$(start_server_vanilla_sd $gamma)
        wait_for_server || { cleanup_server $SERVER_PID; return 1; }

        if [ "$SKIP_WARMUP" = false ]; then
            run_warmup "vanilla_sd_gamma${gamma}"
        fi

        for repeat in $(seq 1 $NUM_REPEATS); do
            run_benchmark "vanilla_sd_ablation" "$gamma" "1.0" "$ablation_rate" "$repeat"
        done

        cleanup_server $SERVER_PID

        # CoSpec with this gamma
        echo "[CoSpec gamma=$gamma]"
        SERVER_PID=$(start_server_cospec $gamma $DEFAULT_SM_RATIO)
        wait_for_server || { cleanup_server $SERVER_PID; return 1; }

        if [ "$SKIP_WARMUP" = false ]; then
            run_warmup "cospec_gamma${gamma}"
        fi

        for repeat in $(seq 1 $NUM_REPEATS); do
            run_benchmark "cospec_ablation" "$gamma" "$DEFAULT_SM_RATIO" "$ablation_rate" "$repeat"
        done

        cleanup_server $SERVER_PID
    done
}

# =============================================================================
# Main Execution
# =============================================================================

trap 'cleanup_server $SERVER_PID' EXIT

if [ -n "$EXPERIMENT" ]; then
    case "$EXPERIMENT" in
        1) run_experiment_1 ;;
        3) run_experiment_3 ;;
        4) run_experiment_4 ;;
        *)
            echo "Unknown experiment: $EXPERIMENT"
            echo "Available: 1 (main comparison), 3 (SM ratio ablation), 4 (gamma ablation)"
            exit 1
            ;;
    esac
else
    # Run all experiments
    run_experiment_1
    run_experiment_3
    run_experiment_4
fi

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $RESULTS_DIR"
echo "CSV: $CSV_FILE"
echo ""
echo "To generate plots, run:"
echo "  python $SCRIPT_DIR/plot_results.py --results-dir $RESULTS_DIR"
