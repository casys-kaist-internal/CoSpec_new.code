#!/bin/bash

# =============================================
# Configuration
# =============================================

# nvidia-smi -c EXCLUSIVE_PROCESS
ulimit -n 65535

# Model Configuration
export TARGET_MODEL="facebook/opt-13b"
export DRAFT_MODEL="facebook/opt-350m"
export TENSOR_PARALLEL_SIZE=2
export DRAFT_TENSOR_PARALLEL_SIZE=2

# Dataset Configuration
DATASETS=("sharegpt" "openmath" "opencode" "math500")

# Request Rate Configuration (requests per second) for each dataset
SHAREGPT_RATES=(2 4 6 8 10 12)
OPENMATH_RATES=(1 2 3 4 5 6)
OPENCODE_RATES=(1 2 3 4 5 6)
MATH500_RATES=(2 4 6 8 10 12 14)

# Function to get request rates for a dataset
get_request_rates() {
    local dataset=$1
    case $dataset in
        "sharegpt")
            echo "${SHAREGPT_RATES[@]}"
            ;;
        "openmath")
            echo "${OPENMATH_RATES[@]}"
            ;;
        "opencode")
            echo "${OPENCODE_RATES[@]}"
            ;;
        "math500")
            echo "${MATH500_RATES[@]}"
            ;;
    esac
}

# Speculative Configuration
BASELINE_SPEC_TOKENS=(0 1 3 5 7)  # Different spec token values for baseline
COSPEC_SPEC_TOKENS=7

# Temperature Configuration
TEMPERATURES=(0)

# Benchmark Configuration
export WARMUP_DURATION=1
export BENCHMARK_DURATION=5  # Duration in minutes

PORT=8003

# CoSpec Feature Configuration
declare -A COSPEC_CONFIGS=(
    ["baseline"]="export COSPEC=0; export COSPEC_DYNAMIC_COLOCATION=0; export COSPEC_SELECTIVE_VALIDATION=0; export COSPEC_CONSOLIDATED_ATTENTION=0"
    ["colocation"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=0; export COSPEC_SELECTIVE_VALIDATION=0; export COSPEC_CONSOLIDATED_ATTENTION=0"
    ["colocation_selective"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=0; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_CONSOLIDATED_ATTENTION=0"
    ["colocation_selective_consolidated"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=0; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["full_cospec"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_CONSOLIDATED_ATTENTION=1"
)

# Set nvidia-smi EXCLUSIVE_PROCESS 
# sudo nvidia-smi -c EXCLUSIVE_PROCESS

# =============================================
# Directory Setup
# =============================================

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="cospec_benchmark_results_${TIMESTAMP}_${TARGET_MODEL}_${DRAFT_MODEL}_tp${TENSOR_PARALLEL_SIZE}_dtp${DRAFT_TENSOR_PARALLEL_SIZE}"
mkdir -p $RESULTS_DIR

# Create CSV header
echo "config,spec_tokens,temperature,request_rate,dataset,tensor_parallel_size,draft_tensor_parallel_size,successful_requests,benchmark_duration,total_input_tokens,total_generated_tokens,request_throughput,output_token_throughput,total_token_throughput,mean_ttft,median_ttft,p99_ttft,mean_tpot,median_tpot,p99_tpot,mean_itl,median_itl,p99_itl,mean_e2el,median_e2el,p99_e2el,mean_token_latency,median_token_latency,p99_token_latency" > "$RESULTS_DIR/benchmark_results.csv"

# =============================================
# Helper Functions
# =============================================

start_server() {
    local config=$1
    local spec_tokens=$2

    > "$RESULTS_DIR/${config}_server.log"
    
    # Set environment variables first
    eval "${COSPEC_CONFIGS[$config]}"
    
    # Base server command
    local CMD="python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port $PORT \
        --model $TARGET_MODEL \
        --seed 42 \
        -tp $TENSOR_PARALLEL_SIZE \
        --enable-chunked-prefill \
        --gpu_memory_utilization 0.80 \
        --disable-log-requests"

    # Add speculative config if spec_tokens > 0
    if [ "$spec_tokens" -gt 0 ]; then
        CMD+=" --speculative_config '{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $spec_tokens, \"draft_tensor_parallel_size\": $DRAFT_TENSOR_PARALLEL_SIZE}'"
    fi

    # Start server in background and redirect output
    eval "$CMD" > "$RESULTS_DIR/${config}_${spec_tokens}_server.log" 2>&1 &
    # Wait a moment for the process to start
    sleep 2
    
    # Get the actual Python process ID
    local server_pid=$(pgrep -f "python -m vllm.entrypoints.openai.api_server.*--port $PORT")
    
    # Wait a bit more to ensure server starts
    sleep 3
    
    # Check if server is running
    if [ -z "$server_pid" ] || ! kill -0 $server_pid 2>/dev/null; then
        echo "Error: Server failed to start for config $config" >&2
        exit 1
    fi
    
    echo $server_pid
}

parse_benchmark_results() {
    local output_file=$1
    local results=()
    
    # Extract metrics from benchmark output
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

    echo "Running warmup with configuration: $config (Spec Tokens: $spec_tokens, Temperature: $temperature, Request Rate: $request_rate, Dataset: $dataset)"

    python benchmark_serving.py \
        --backend vllm \
        --port $PORT \
        --model $TARGET_MODEL \
        --dataset-name $dataset \
        --ignore-eos \
        --duration-minutes $WARMUP_DURATION \
        --request-rate $request_rate \
        --temperature $temperature > "$RESULTS_DIR/${config}_${spec_tokens}_${temperature}_${request_rate}_${dataset}_output.txt"
}

run_benchmark() {
    local config=$1
    local spec_tokens=$2
    local temperature=$3
    local request_rate=$4
    local dataset=$5
    
    echo "Running benchmark with configuration: $config (Spec Tokens: $spec_tokens, Temperature: $temperature, Request Rate: $request_rate, Dataset: $dataset)"
    
    # Run benchmark with duration-based progress
    python benchmark_serving.py \
        --backend vllm \
        --port $PORT \
        --model $TARGET_MODEL \
        --dataset-name $dataset \
        --ignore-eos \
        --duration-minutes $BENCHMARK_DURATION \
        --request-rate $request_rate \
        --temperature $temperature > "$RESULTS_DIR/${config}_${spec_tokens}_${temperature}_${request_rate}_${dataset}_output.txt"
    
    # Parse and save results
    local results=($(parse_benchmark_results "$RESULTS_DIR/${config}_${spec_tokens}_${temperature}_${request_rate}_${dataset}_output.txt"))
    echo "$config,$spec_tokens,$temperature,$request_rate,$dataset,$TENSOR_PARALLEL_SIZE,$DRAFT_TENSOR_PARALLEL_SIZE,${results[*]}" | tr ' ' ',' >> "$RESULTS_DIR/benchmark_results.csv"
}

# =============================================
# Main Execution
# =============================================

# Define the order of configurations to run
declare -a CONFIG_ORDER=(
    "colocation"
    "colocation_selective"
    "colocation_selective_consolidated"
)

TOTAL_RUNS=0
# Baseline runs
for spec_tokens in "${BASELINE_SPEC_TOKENS[@]}"; do
    if [ "$spec_tokens" -eq 0 ]; then
        temperatures=(0)
    else
        temperatures=("${TEMPERATURES[@]}")
    fi
    
    # Calculate runs for each dataset with its specific request rates
    for dataset in "${DATASETS[@]}"; do
        read -ra rates <<< "$(get_request_rates "$dataset")"
        TOTAL_RUNS=$((TOTAL_RUNS + ${#temperatures[@]} * ${#rates[@]}))
    done
done

# CoSpec runs
for config in "${CONFIG_ORDER[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        read -ra rates <<< "$(get_request_rates "$dataset")"
        TOTAL_RUNS=$((TOTAL_RUNS + ${#TEMPERATURES[@]} * ${#rates[@]}))
    done
done

# Initialize run counter
CURRENT_RUN=0

# Run baseline benchmark with different spec tokens and request rates
echo "Running baseline benchmarks with different configurations..."
echo "Total runs to complete: $TOTAL_RUNS"

# for spec_tokens in "${BASELINE_SPEC_TOKENS[@]}"; do
#     # Start server for this spec_tokens configuration
#     server_pid=$(start_server "baseline" $spec_tokens)
#     echo "Server PID: $server_pid"

#     # run_warmup "baseline" "$spec_tokens" 0.5 8 "sharegpt"
    
#     # For spec_tokens=0, only run with temperature=0
#     temperatures=("${TEMPERATURES[@]}")
    
#     # Run all temperature and request rate combinations
#     for dataset in "${DATASETS[@]}"; do
#         for temperature in "${temperatures[@]}"; do
#             read -ra rates <<< "$(get_request_rates "$dataset")"
#             for request_rate in "${rates[@]}"; do
#                 CURRENT_RUN=$((CURRENT_RUN + 1))
#                 slack "[$CURRENT_RUN/$TOTAL_RUNS]"
#                 run_benchmark "baseline" "$spec_tokens" "$temperature" "$request_rate" "$dataset"
#             done
#         done
#     done
    
#     # Cleanup server after all request rates are done
#     kill $server_pid
#     wait $server_pid 2>/dev/null
#     sleep 5
# done

# Run CoSpec ablation studies
echo "Running CoSpec ablation studies..."

for config in "${CONFIG_ORDER[@]}"; do
    echo "Running $config configuration..."
    server_pid=$(start_server "$config" $COSPEC_SPEC_TOKENS)
    echo "Server PID: $server_pid"
    
    # Run all temperature and request rate combinations
    for dataset in "${DATASETS[@]}"; do
        for temperature in "${TEMPERATURES[@]}"; do
            read -ra rates <<< "$(get_request_rates "$dataset")"
            for request_rate in "${rates[@]}"; do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                slack "[$CURRENT_RUN/$TOTAL_RUNS]"
                run_benchmark "$config" "$COSPEC_SPEC_TOKENS" "$temperature" "$request_rate" "$dataset"
            done
        done
    done
    
    # Cleanup server after all request rates are done
    kill $server_pid
    wait $server_pid 2>/dev/null
    sleep 5
done

echo "Benchmark results have been saved to $RESULTS_DIR/benchmark_results.csv" 