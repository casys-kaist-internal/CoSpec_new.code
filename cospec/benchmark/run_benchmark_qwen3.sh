#!/bin/bash

# =============================================
# CoSpec Benchmark: Qwen3-8B + Qwen3-0.6B
# =============================================

ulimit -n 65535

# Benchmark Control
SKIP_BASELINE=false

# Model Configuration
export TARGET_MODEL="Qwen/Qwen3-8B"
export DRAFT_MODEL="Qwen/Qwen3-0.6B"
export TENSOR_PARALLEL_SIZE=1
export DRAFT_TENSOR_PARALLEL_SIZE=1

# Dataset Configuration
DATASETS=("math500")

# Request Rate Configuration (requests per second) for each dataset
MATH500_RATES=(2 4 6 8 10 12 14 16)
SHAREGPT_RATES=(1 2 3 4 5 6 7 8)

get_request_rates() {
    local dataset=$1
    case $dataset in
        "sharegpt") echo "${SHAREGPT_RATES[@]}" ;;
        "math500")  echo "${MATH500_RATES[@]}" ;;
    esac
}

# Speculative Configuration
BASELINE_SPEC_TOKENS=(0 1 3 5 7)
COSPEC_SPEC_TOKENS=7

# Temperature Configuration
TEMPERATURES=(0)

# Benchmark Configuration
export WARMUP_DURATION=1
export BENCHMARK_DURATION=10  # Duration in minutes

PORT=8100

# CoSpec Feature Configuration
declare -A COSPEC_CONFIGS=(
    ["baseline"]="export COSPEC=0; export COSPEC_DYNAMIC_COLOCATION=0"
    ["colocation"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=0"
    ["colocation_dynamic"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1"
)

# Define the configurations to run
declare -a CONFIG_ORDER=(
    "colocation"
    "colocation_dynamic"
)

# =============================================
# Directory Setup
# =============================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_qwen3_8b_0.6b_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

echo "config,spec_tokens,temperature,request_rate,dataset,tensor_parallel_size,draft_tensor_parallel_size,successful_requests,benchmark_duration,total_input_tokens,total_generated_tokens,request_throughput,output_token_throughput,total_token_throughput,mean_ttft,median_ttft,p99_ttft,mean_tpot,median_tpot,p99_tpot,mean_itl,median_itl,p99_itl,mean_e2el,median_e2el,p99_e2el,mean_token_latency,median_token_latency,p99_token_latency" > "$RESULTS_DIR/benchmark_results.csv"

# =============================================
# Helper Functions
# =============================================

start_server() {
    local config=$1
    local spec_tokens=$2

    > "$RESULTS_DIR/${config}_server.log"

    eval "${COSPEC_CONFIGS[$config]}"

    local CMD="python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port $PORT \
        --model $TARGET_MODEL \
        --seed 42 \
        -tp $TENSOR_PARALLEL_SIZE \
        --enable-chunked-prefill \
        --gpu_memory_utilization 0.80 \
        --disable-log-requests"

    if [ "$spec_tokens" -gt 0 ]; then
        CMD+=" --speculative_config '{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $spec_tokens, \"draft_tensor_parallel_size\": $DRAFT_TENSOR_PARALLEL_SIZE}'"
    fi

    eval "$CMD" > "$RESULTS_DIR/${config}_${spec_tokens}_server.log" 2>&1 &
    sleep 2

    local server_pid=$(pgrep -f "python -m vllm.entrypoints.openai.api_server.*--port $PORT")

    sleep 3

    if [ -z "$server_pid" ] || ! kill -0 $server_pid 2>/dev/null; then
        echo "Error: Server failed to start for config $config" >&2
        exit 1
    fi

    echo $server_pid
}

parse_benchmark_results() {
    local output_file=$1
    local results=()

    results+=($(grep "Successful requests:" "$output_file" | awk '{print $3}'))
    results+=($(grep "Benchmark duration (s):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Total input tokens:" "$output_file" | awk '{print $4}'))
    results+=($(grep "Total generated tokens:" "$output_file" | awk '{print $4}'))
    results+=($(grep "Request throughput (req/s):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Output token throughput (tok/s):" "$output_file" | awk '{print $5}'))
    results+=($(grep "Total Token throughput (tok/s):" "$output_file" | awk '{print $5}'))
    results+=($(grep "Mean TTFT (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Median TTFT (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "P99 TTFT (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Mean TPOT (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Median TPOT (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "P99 TPOT (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Mean ITL (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Median ITL (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "P99 ITL (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Mean E2EL (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Median E2EL (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "P99 E2EL (ms):" "$output_file" | awk '{print $4}'))
    results+=($(grep "Mean Token Latency (ms):" "$output_file" | awk '{print $5}'))
    results+=($(grep "Median Token Latency (ms):" "$output_file" | awk '{print $5}'))
    results+=($(grep "P99 Token Latency (ms):" "$output_file" | awk '{print $5}'))

    echo "${results[*]}"
}

run_warmup() {
    local config=$1
    local spec_tokens=$2
    local temperature=$3
    local request_rate=$4
    local dataset=$5

    echo "Running warmup: $config (Spec Tokens: $spec_tokens, Rate: $request_rate, Dataset: $dataset)"

    python benchmark_serving.py \
        --backend vllm \
        --port $PORT \
        --model $TARGET_MODEL \
        --dataset-name $dataset \
        --ignore-eos \
        --duration-minutes $WARMUP_DURATION \
        --request-rate $request_rate \
        --temperature $temperature > "$RESULTS_DIR/${config}_${spec_tokens}_${temperature}_${request_rate}_${dataset}_warmup.txt"
}

run_benchmark() {
    local config=$1
    local spec_tokens=$2
    local temperature=$3
    local request_rate=$4
    local dataset=$5

    echo "Running benchmark: $config (Spec Tokens: $spec_tokens, Temp: $temperature, Rate: $request_rate, Dataset: $dataset)"

    python benchmark_serving.py \
        --backend vllm \
        --port $PORT \
        --model $TARGET_MODEL \
        --dataset-name $dataset \
        --ignore-eos \
        --duration-minutes $BENCHMARK_DURATION \
        --request-rate $request_rate \
        --temperature $temperature > "$RESULTS_DIR/${config}_${spec_tokens}_${temperature}_${request_rate}_${dataset}_output.txt"

    local results=($(parse_benchmark_results "$RESULTS_DIR/${config}_${spec_tokens}_${temperature}_${request_rate}_${dataset}_output.txt"))
    echo "$config,$spec_tokens,$temperature,$request_rate,$dataset,$TENSOR_PARALLEL_SIZE,$DRAFT_TENSOR_PARALLEL_SIZE,${results[*]}" | tr ' ' ',' >> "$RESULTS_DIR/benchmark_results.csv"
}

# =============================================
# Main Execution
# =============================================

TOTAL_RUNS=0

# Baseline runs
if [ "$SKIP_BASELINE" = false ]; then
    for spec_tokens in "${BASELINE_SPEC_TOKENS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            read -ra rates <<< "$(get_request_rates "$dataset")"
            TOTAL_RUNS=$((TOTAL_RUNS + ${#TEMPERATURES[@]} * ${#rates[@]}))
        done
    done
fi

# CoSpec runs
for config in "${CONFIG_ORDER[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        read -ra rates <<< "$(get_request_rates "$dataset")"
        TOTAL_RUNS=$((TOTAL_RUNS + ${#TEMPERATURES[@]} * ${#rates[@]}))
    done
done

CURRENT_RUN=0

# Run baseline benchmarks
if [ "$SKIP_BASELINE" = false ]; then
    echo "Running baseline benchmarks..."
    echo "Total runs to complete: $TOTAL_RUNS"

    for spec_tokens in "${BASELINE_SPEC_TOKENS[@]}"; do
        server_pid=$(start_server "baseline" $spec_tokens)
        echo "Server PID: $server_pid"

        for dataset in "${DATASETS[@]}"; do
            for temperature in "${TEMPERATURES[@]}"; do
                read -ra rates <<< "$(get_request_rates "$dataset")"
                for request_rate in "${rates[@]}"; do
                    CURRENT_RUN=$((CURRENT_RUN + 1))
                    echo "[$CURRENT_RUN/$TOTAL_RUNS]"
                    run_benchmark "baseline" "$spec_tokens" "$temperature" "$request_rate" "$dataset"
                done
            done
        done

        kill $server_pid
        wait $server_pid 2>/dev/null
        sleep 5
    done
fi

# Run CoSpec configurations
echo "Running CoSpec configurations..."

for config in "${CONFIG_ORDER[@]}"; do
    echo "Running $config configuration..."

    for dataset in "${DATASETS[@]}"; do
        for temperature in "${TEMPERATURES[@]}"; do
            server_pid=$(start_server "$config" $COSPEC_SPEC_TOKENS)
            echo "Server PID: $server_pid"

            read -ra rates <<< "$(get_request_rates "$dataset")"
            for request_rate in "${rates[@]}"; do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                echo "[$CURRENT_RUN/$TOTAL_RUNS]"
                run_benchmark "$config" "$COSPEC_SPEC_TOKENS" "$temperature" "$request_rate" "$dataset"
            done

            kill $server_pid
            wait $server_pid 2>/dev/null
            sleep 5
        done
    done
done

echo "Benchmark results saved to $RESULTS_DIR/benchmark_results.csv"
