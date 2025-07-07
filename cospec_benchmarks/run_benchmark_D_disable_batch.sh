#!/bin/bash

# =============================================
# Configuration
# =============================================

# Set EXCLUSIVE_PROCESS if possible. This makes sure that MPS is used.
# nvidia-smi -c EXCLUSIVE_PROCESS 
ulimit -n 65535

# Benchmark Control
SKIP_BASELINE=true  # Set to false to run baseline configurations

# Model Configuration
export TARGET_MODEL="facebook/opt-66b"
export DRAFT_MODEL="facebook/opt-1.3b"
export TENSOR_PARALLEL_SIZE=4
export DRAFT_TENSOR_PARALLEL_SIZE=4
export DOWNLOAD_DIR="/workspace"
export DISABLE_BY_BATCH_SIZE=64
export VLLM_ATTENTION_BACKEND="XFORMERS"

# Dataset Configuration
DATASETS=("opencodeinstruct")

# Dataset Configuration
RATES=(1 2 3 4 5)

# Request Rate Configuration (requests per second)
# MATH500_RATES=(2 4 6 8 10)
MATH500_RATES=(10 8 6 4 2)
# Speculative Configuration
BASELINE_SPEC_TOKENS=(0 1 3 5 7)  # Different spec token values for baseline
COSPEC_SPEC_TOKENS=7

# Benchmark Configuration
BENCHMARK_DURATION=10  # Duration in minutes

# Calculate number of prompts based on duration and request rates
declare -A NUM_PROMPTS
for rate in "${RATES[@]}"; do
    # Convert minutes to seconds and multiply by request rate
    NUM_PROMPTS[$rate]=$((BENCHMARK_DURATION * 60 * rate))
done

PORT=8100

# CoSpec Feature Configuration
declare -A COSPEC_CONFIGS=(
    ["disable_by_batch"]="export COSPEC=0; export COSPEC_DYNAMIC_COLOCATION=0; export COSPEC_SELECTIVE_VALIDATION=0; export COSPEC_CONSOLIDATED_ATTENTION=0"
    ["baseline"]="export COSPEC=0; export COSPEC_DYNAMIC_COLOCATION=0; export COSPEC_SELECTIVE_VALIDATION=0; export COSPEC_CONSOLIDATED_ATTENTION=0"
    ["colocation"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=0; export COSPEC_SELECTIVE_VALIDATION=0; export COSPEC_CONSOLIDATED_ATTENTION=0"
    ["colocation_dynamic"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=0; export COSPEC_CONSOLIDATED_ATTENTION=0"
    ["colocation_dynamic_selective"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=tile; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.3; export COSPEC_CONSOLIDATED_ATTENTION=0"
    ["full_cospec"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=tile; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.3; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["without_selective_validation"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=0; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_tile_0.1"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=tile; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.1; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_tile_0.3"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=tile; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.3; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_tile_0.5"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=tile; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.5; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_threshold_0.1"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=threshold; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.1; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_threshold_0.3"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=threshold; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.3; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_threshold_0.5"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=threshold; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.5; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_linear_0.1"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=linear; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.1; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_linear_0.3"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=linear; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.3; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_linear_0.5"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=linear; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.5; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_polynomial_0.1"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=polynomial; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.1; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_polynomial_0.3"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=polynomial; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.3; export COSPEC_CONSOLIDATED_ATTENTION=1"
    ["selective_validation_polynomial_0.5"]="export COSPEC=1; export COSPEC_DYNAMIC_COLOCATION=1; export COSPEC_SELECTIVE_VALIDATION=1; export COSPEC_SELECTIVE_VALIDATION_METHOD=polynomial; export COSPEC_SELECTIVE_VALIDATION_THRESHOLD=0.5; export COSPEC_CONSOLIDATED_ATTENTION=1"
)

# Define the configurations to run
declare -a CONFIG_ORDER=(
    "disable_by_batch"
)

# =============================================
# Directory Setup
# =============================================

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="D_disable_by_batch_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

# Create CSV header
echo "config,spec_tokens,request_rate,dataset,tensor_parallel_size,draft_tensor_parallel_size,successful_requests,benchmark_duration,total_input_tokens,total_generated_tokens,request_throughput,output_token_throughput,total_token_throughput,mean_ttft,median_ttft,p99_ttft,mean_tpot,median_tpot,p99_tpot,mean_itl,median_itl,p99_itl,mean_e2el,median_e2el,p99_e2el,mean_token_latency,median_token_latency,p99_token_latency" > "$RESULTS_DIR/benchmark_results.csv"

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

    # Add speculative config if spec_tokens > 0 with disable_by_batch_size
    if [ "$spec_tokens" -gt 0 ]; then
        CMD+=" --speculative_config '{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $spec_tokens, \"draft_tensor_parallel_size\": $DRAFT_TENSOR_PARALLEL_SIZE, \"disable_by_batch_size\": $DISABLE_BY_BATCH_SIZE}'"
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

run_benchmark() {
    local config=$1
    local spec_tokens=$2
    local request_rate=$3
    local dataset=$4
    
    echo "Running benchmark with configuration: $config (Spec Tokens: $spec_tokens, Request Rate: $request_rate, Dataset: $dataset)"
    echo "Number of prompts to process: ${NUM_PROMPTS[$request_rate]}"
    
    # Run benchmark with num-prompts based progress
    python benchmark_serving_disable_batch.py \
        --backend vllm \
        --port $PORT \
        --model $TARGET_MODEL \
        --dataset-name $dataset \
        --ignore-eos \
        --request-rate $request_rate \
        --num-prompts ${NUM_PROMPTS[$request_rate]} > "$RESULTS_DIR/${config}_${spec_tokens}_${request_rate}_${dataset}_output.txt"
    
    # Parse and save results
    local results=($(parse_benchmark_results "$RESULTS_DIR/${config}_${spec_tokens}_${request_rate}_${dataset}_output.txt"))
    echo "$config,$spec_tokens,$request_rate,$dataset,$TENSOR_PARALLEL_SIZE,$DRAFT_TENSOR_PARALLEL_SIZE,${results[*]}" | tr ' ' ',' >> "$RESULTS_DIR/benchmark_results.csv"
}

# =============================================
# Main Execution
# =============================================

TOTAL_RUNS=0
# Baseline runs
if [ "$SKIP_BASELINE" = false ]; then
    for spec_tokens in "${BASELINE_SPEC_TOKENS[@]}"; do
        # Calculate runs for each dataset
        for dataset in "${DATASETS[@]}"; do
            TOTAL_RUNS=$((TOTAL_RUNS + ${#RATES[@]}))
        done
    done
fi

# CoSpec runs
for config in "${CONFIG_ORDER[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        TOTAL_RUNS=$((TOTAL_RUNS + ${#RATES[@]}))
    done
done

# Initialize run counter
CURRENT_RUN=0

# Run baseline benchmark with different spec tokens and request rates
if [ "$SKIP_BASELINE" = false ]; then
    echo "Running baseline benchmarks with different configurations..."
    echo "Total runs to complete: $TOTAL_RUNS"

    for spec_tokens in "${BASELINE_SPEC_TOKENS[@]}"; do
        # Start server for this spec_tokens configuration
        server_pid=$(start_server "baseline" $spec_tokens)
        echo "Server PID: $server_pid"

        for dataset in "${DATASETS[@]}"; do
            for request_rate in "${RATES[@]}"; do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                slack "[$CURRENT_RUN/$TOTAL_RUNS]"
                run_benchmark "baseline" "$spec_tokens" "$request_rate" "$dataset"
            done
        done
        
        # Cleanup server after all request rates are done
        kill $server_pid
        wait $server_pid 2>/dev/null
        sleep 5
    done
fi

# Run CoSpec ablation studies
echo "Running CoSpec ablation studies..."

# Run all request rate combinations
for dataset in "${DATASETS[@]}"; do
    for config in "${CONFIG_ORDER[@]}"; do
        echo "Running $config configuration..."
        server_pid=$(start_server "$config" $COSPEC_SPEC_TOKENS)
        echo "Server PID: $server_pid"
        
        for request_rate in "${RATES[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            slack "[$CURRENT_RUN/$TOTAL_RUNS]"
            run_benchmark "$config" "$COSPEC_SPEC_TOKENS" "$request_rate" "$dataset"
        done
        
        kill $server_pid
        wait $server_pid 2>/dev/null
        sleep 5
    done
done

echo "Benchmark results have been saved to $RESULTS_DIR/benchmark_results.csv" 