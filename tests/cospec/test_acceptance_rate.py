"""Compare acceptance rates between CoSpec and regular spec decoding.

This test verifies that CoSpec reports the same acceptance metrics as
regular speculative decoding when outputs are identical.

Run: VLLM_USE_V1=0 python tests/cospec/test_acceptance_rate.py
"""
import os
import sys
import subprocess
import json

os.environ["VLLM_USE_V1"] = "0"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "San Francisco is known for its",
    "Facebook was created in 2004 by",
    "Curious George is a",
    "Python 3.11 brings improvements to its",
]
MAX_TOKENS = 64
NUM_SPEC_TOKENS = 5
MODEL = "JackFram/llama-68m"

def run_subprocess(cospec: bool):
    script = f'''
import os
os.environ["COSPEC"] = "{1 if cospec else 0}"
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams
import torch
import json

llm = LLM(
    model="{MODEL}",
    speculative_config={{"model": "{MODEL}", "num_speculative_tokens": {NUM_SPEC_TOKENS}}},
    enforce_eager=True,
    max_num_seqs=8,
)

outputs = llm.generate({PROMPTS!r}, SamplingParams(temperature=0.0, max_tokens={MAX_TOKENS}))
torch.cuda.synchronize()

results = [out.outputs[0].text for out in outputs]

executor = llm.llm_engine.model_executor
worker = getattr(executor, "driver_worker", None)
actual = getattr(worker, "worker", worker)

accepted, draft = 0, 0
sampler = getattr(actual, "spec_decode_sampler", None)
if sampler:
    acc_val = sampler.num_accepted_tokens
    draft_val = sampler.num_draft_tokens
    accepted = int(acc_val.item() if hasattr(acc_val, "item") else acc_val)
    draft = int(draft_val.item() if hasattr(draft_val, "item") else draft_val)

print("RESULT:" + json.dumps({{"outputs": results, "accepted": accepted, "draft": draft}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr[-800:]}")
        return None

    # Print debug lines from stderr
    for line in result.stderr.split('\n'):
        if line.startswith('STEP ') or line.startswith('DRAFT_'):
            print(f"  [DEBUG] {line.strip()}")
    for line in result.stdout.split('\n'):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    return None

if __name__ == "__main__":
    print("=" * 70)
    print("ACCEPTANCE RATE COMPARISON: Regular SD vs CoSpec")
    print("=" * 70)

    print("\nRunning Regular SD...")
    regular = run_subprocess(False)

    print("Running CoSpec...")
    cospec = run_subprocess(True)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if not regular or not cospec:
        print("ERROR: One or both runs failed")
        sys.exit(1)

    # Check outputs
    if regular["outputs"] == cospec["outputs"]:
        print("\n✓ Outputs match exactly")
    else:
        print("\n✗ Outputs differ!")
        for i, (r, c) in enumerate(zip(regular["outputs"], cospec["outputs"])):
            if r != c:
                print(f"  Prompt {i}: '{r[:40]}...' vs '{c[:40]}...'")

    # Compare rates
    r_rate = regular["accepted"] / regular["draft"] if regular["draft"] > 0 else 0
    c_rate = cospec["accepted"] / cospec["draft"] if cospec["draft"] > 0 else 0

    print(f"\nRegular SD: {regular['accepted']:4d} / {regular['draft']:4d} = {r_rate:6.2%}")
    print(f"CoSpec:     {cospec['accepted']:4d} / {cospec['draft']:4d} = {c_rate:6.2%}")

    diff = abs(r_rate - c_rate)
    print(f"\nDifference: {diff:.2%}")

    if diff < 0.05:  # 5% tolerance
        print("\n" + "=" * 70)
        print("✓ PASS: Acceptance rates are within 5% tolerance")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ FAIL: Acceptance rates differ by more than 5%")
        print("  This indicates a bug in CoSpec metrics tracking.")
        print("  See CLAUDE.md P0 section for fix plan.")
        print("=" * 70)
        sys.exit(1)
