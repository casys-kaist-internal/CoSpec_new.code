"""Correctness test: CoSpec speculative decoding vs baseline AR.

Sends identical prompts to two servers (CoSpec and baseline) with greedy
decoding and compares outputs token-by-token. With temperature=0, outputs
must be identical.

Usage:
    # Start both servers first, then:
    python3 test_correctness.py --model Qwen/Qwen3-8B \
        --cospec-port 8100 --baseline-port 8200
"""

import argparse
import json
import requests
import sys
from typing import Optional

TEST_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "In quantum mechanics, the Heisenberg uncertainty principle states that",
    "The quick brown fox jumps over the lazy dog. This sentence is famous because",
    "To make a classic margherita pizza, you need the following ingredients:",
    "The theory of general relativity, published by Albert Einstein in 1915,",
    "Write a Python function that reverses a string:\n\ndef reverse_string(s):",
    "The three laws of thermodynamics are:\n1.",
]


def complete(url: str, model: str, prompt: str, max_tokens: int = 100) -> str:
    """Send a non-streaming completion request and return the generated text."""
    resp = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["text"]


def check_server(url: str, label: str) -> bool:
    """Check if a server is reachable."""
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        print(f"  {label} server at {url} is not reachable")
        return False


def main():
    parser = argparse.ArgumentParser(description="CoSpec correctness test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cospec-port", type=int, default=8100)
    parser.add_argument("--baseline-port", type=int, default=8200)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--prompts", type=str, default=None,
                        help="JSON file with list of prompt strings")
    args = parser.parse_args()

    cospec_url = f"http://{args.host}:{args.cospec_port}"
    baseline_url = f"http://{args.host}:{args.baseline_port}"

    print("Checking servers...")
    if not check_server(cospec_url, "CoSpec") or \
       not check_server(baseline_url, "Baseline"):
        sys.exit(1)
    print("Both servers ready.\n")

    prompts = TEST_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)

    passed = 0
    failed = 0
    errors = 0

    for i, prompt in enumerate(prompts):
        short = prompt[:60].replace("\n", "\\n")
        print(f"[{i+1}/{len(prompts)}] \"{short}...\"")

        try:
            cospec_out = complete(cospec_url, args.model, prompt, args.max_tokens)
            baseline_out = complete(baseline_url, args.model, prompt, args.max_tokens)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            continue

        if cospec_out == baseline_out:
            print(f"  PASS ({len(cospec_out)} chars)")
            passed += 1
        else:
            print(f"  FAIL")
            # Find first divergence point
            for j in range(min(len(cospec_out), len(baseline_out))):
                if cospec_out[j] != baseline_out[j]:
                    print(f"  Diverge at char {j}:")
                    print(f"    CoSpec:   ...{repr(cospec_out[max(0,j-20):j+30])}")
                    print(f"    Baseline: ...{repr(baseline_out[max(0,j-20):j+30])}")
                    break
            else:
                print(f"  Length mismatch: CoSpec={len(cospec_out)}, Baseline={len(baseline_out)}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {errors} errors")
    print(f"{'='*50}")
    sys.exit(1 if failed > 0 or errors > 0 else 0)


if __name__ == "__main__":
    main()
