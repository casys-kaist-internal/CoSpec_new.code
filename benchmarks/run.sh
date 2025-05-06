#!/bin/bash

# Configuration
export COSPEC=${COSPEC:-0} 
export PROFILE=${PROFILE:-0}
export AR=${AR:-0}
export TARGET_MODEL="facebook/opt-6.7b"
export DRAFT_MODEL="facebook/opt-125m"
export NUM_SPEC_TOKENS=7
export TENSOR_PARALLEL_SIZE=1

# Base command
CMD="python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model $TARGET_MODEL \
    --seed 42 \
    -tp $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization 0.85 \
    --disable-log-requests"

# Speculative config
if [ "$AR" -eq 0 ]; then
    CMD+=" --speculative_config '{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}'"
else
    # assert that CoSpec is disabled
    if [ "$COSPEC" -eq 1 ]; then
        echo "CoSpec is enabled but AR is set to 1"
        exit 1
    fi
fi

# Add profiling if enabled
if [ "$PROFILE" -eq 1 ]; then
    CMD="nsys profile -t cuda,osrt,nvtx \
        --gpu-metrics-device=all \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        $CMD"
fi

# Execute the command
eval $CMD
