#!/bin/bash
# MODEL="facebook/opt-6.7b"
# MODEL="facebook/opt-13b"
MODEL="huggyllama/llama-7b"
# DATASET="math500"
# DATASET="opencode"
DATASET="gsm8k"
REQUEST_RATE=20
TEMPERATURE=0
NUM_PROMPTS=1000
PORT=8011

echo "Starting benchmark with configuration:"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Request Rate: $REQUEST_RATE"
echo "Temperature: $TEMPERATURE"
echo "Duration: $DURATION minutes"
echo "Port: $PORT"

# Run the benchmark
python benchmark_serving.py \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --request-rate "$REQUEST_RATE" \
    --temperature "$TEMPERATURE" \
    --num-prompts "$NUM_PROMPTS" \
    --ignore-eos \
    --port "$PORT"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully"
else
    echo "Error: Benchmark failed"
    exit 1
fi
