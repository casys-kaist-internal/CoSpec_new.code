#!/bin/bash
MODEL="facebook/opt-6.7b"
DATASET="math500"
TEMPERATURE=0
PORT=8011

echo "Starting benchmark with configuration:"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Temperature: $TEMPERATURE"
echo "Port: $PORT"
echo "Request Rate Pattern: 4 req/s (60s) -> 10 req/s (60s) -> 4 req/s (60s)"

# Run the benchmark
python benchmark_serving_dynamic_request_rate.py \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --temperature "$TEMPERATURE" \
    --ignore-eos \
    --port "$PORT" \
    --dynamic-rates

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully"
else
    echo "Error: Benchmark failed"
    exit 1
fi
