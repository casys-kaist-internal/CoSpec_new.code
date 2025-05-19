#!/bin/bash

# Configuration
export COSPEC=${COSPEC:-0} 
export PROFILE=${PROFILE:-0}
export AR=${AR:-0}
export TARGET_MODEL="facebook/opt-6.7b"
# export TARGET_MODEL="facebook/opt-350m"
export DRAFT_MODEL="facebook/opt-125m"
# export TARGET_MODEL="meta-llama/Llama-2-7b-hf"
# export DRAFT_MODEL="JackFram/llama-160m"
export NUM_SPEC_TOKENS=7
export TENSOR_PARALLEL_SIZE=1
export DRAFT_TENSOR_PARALLEL_SIZE=1

# Base command
CMD="python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8001 \
    --model $TARGET_MODEL \
    --seed 42 \
    -tp $TENSOR_PARALLEL_SIZE \
    --enable-chunked-prefill \
    --gpu_memory_utilization 0.80 \
    --disable-log-requests"

    # --enable-chunked-prefill \
    # --enforce-eager \

# Speculative config
if [ "$AR" -eq 0 ]; then
    CMD+=" --speculative_config '{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"draft_tensor_parallel_size\": $DRAFT_TENSOR_PARALLEL_SIZE}'"
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
