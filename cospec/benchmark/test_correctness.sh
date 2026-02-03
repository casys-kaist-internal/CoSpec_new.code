#!/bin/bash
# Correctness test: CoSpec vs baseline (sequential, one server at a time)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="Qwen/Qwen3-8B"
DRAFT="Qwen/Qwen3-0.6B"
PORT=8100
RESULTS="/tmp/correctness_test"
mkdir -p "$RESULTS"

# Use default attention backend (FLASH_ATTN) for max performance
export CUDA_MPS_PIPE_DIRECTORY="${PROJECT_ROOT}/log/mps/nvidia-mps"

wait_for_model() {
    local port=$1
    echo "Waiting for model to load..."
    for i in $(seq 1 180); do
        if curl -s "http://localhost:$port/v1/models" | grep -q "$MODEL" 2>/dev/null; then
            echo "Server ready."
            return 0
        fi
        sleep 2
    done
    echo "Server failed to start"
    return 1
}

collect_outputs() {
    local port=$1
    local output_file=$2
    python3 -c "
import json, requests

prompts = [
    'The capital of France is',
    'def fibonacci(n):\n    if n <= 1:\n        return n\n    return',
    'In quantum mechanics, the Heisenberg uncertainty principle states that',
    'The quick brown fox jumps over the lazy dog. This sentence is famous because',
    'To make a classic margherita pizza, you need the following ingredients:',
    'The theory of general relativity, published by Albert Einstein in 1915,',
    'Write a Python function that reverses a string:\n\ndef reverse_string(s):',
    'The three laws of thermodynamics are:\n1.',
]

results = []
for i, prompt in enumerate(prompts):
    resp = requests.post(
        'http://127.0.0.1:$port/v1/completions',
        json={'model': '$MODEL', 'prompt': prompt, 'max_tokens': 100, 'temperature': 0},
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()['choices'][0]['text']
    results.append({'prompt': prompt, 'output': text})
    print(f'  [{i+1}/{len(prompts)}] {len(text)} chars')

with open('$output_file', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved {len(results)} outputs to $output_file')
"
}

# === Phase 1: CoSpec server ===
echo "=== Phase 1: CoSpec (speculative decoding) ==="
COSPEC=1 python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port $PORT \
    --model $MODEL --seed 42 --enforce-eager \
    --enable-chunked-prefill --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    --speculative-config "{\"model\": \"$DRAFT\", \"num_speculative_tokens\": 5}" &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

wait_for_model $PORT
echo "Collecting CoSpec outputs..."
collect_outputs $PORT "$RESULTS/cospec.json"

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
trap - EXIT
sleep 5

# === Phase 2: Baseline server ===
echo ""
echo "=== Phase 2: Baseline (no speculative decoding) ==="
COSPEC=0 python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port $PORT \
    --model $MODEL --seed 42 --enforce-eager \
    --enable-chunked-prefill --gpu-memory-utilization 0.85 \
    --disable-log-requests &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

wait_for_model $PORT
echo "Collecting baseline outputs..."
collect_outputs $PORT "$RESULTS/baseline.json"

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
trap - EXIT

# === Phase 3: Compare ===
echo ""
echo "=== Phase 3: Comparing outputs ==="
python3 -c "
import json

with open('$RESULTS/cospec.json') as f:
    cospec = json.load(f)
with open('$RESULTS/baseline.json') as f:
    baseline = json.load(f)

passed = 0
failed = 0
for i, (c, b) in enumerate(zip(cospec, baseline)):
    short = c['prompt'][:50].replace('\n', '\\\\n')
    if c['output'] == b['output']:
        print(f'  [{i+1}] PASS  \"{short}...\"')
        passed += 1
    else:
        print(f'  [{i+1}] FAIL  \"{short}...\"')
        # Find divergence
        co, bo = c['output'], b['output']
        for j in range(min(len(co), len(bo))):
            if co[j] != bo[j]:
                print(f'       Diverge at char {j}:')
                print(f'       CoSpec:   {repr(co[max(0,j-10):j+30])}')
                print(f'       Baseline: {repr(bo[max(0,j-10):j+30])}')
                break
        else:
            print(f'       Length: CoSpec={len(co)}, Baseline={len(bo)}')
        failed += 1

print(f'\n{\"=\"*50}')
print(f'Results: {passed} passed, {failed} failed out of {len(cospec)}')
print(f'{\"=\"*50}')
exit(1 if failed > 0 else 0)
"
