#!/usr/bin/env python3
"""Quick API server test - run directly to verify CoSpec works with HTTP requests.

Usage: python3 tests/cospec/run_api_test.py
"""

import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL = "JackFram/llama-68m"
PORT = 18234
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


def start_server():
    env = os.environ.copy()
    env.update({"COSPEC": "1", "VLLM_USE_V1": "0"})

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--model", MODEL,
        "--speculative-config",
        json.dumps({"model": MODEL, "num_speculative_tokens": 5}),
        "--seed", "42",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.7",
        "--disable-frontend-multiprocessing",
        "--disable-log-requests",
    ]

    print(f"Starting server: {' '.join(cmd[:6])}...")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_server(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def send_request(prompt, max_tokens=32):
    resp = requests.post(
        f"http://localhost:{PORT}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 42,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def kill_server(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.kill()
        proc.wait(timeout=5)


def main():
    proc = start_server()
    try:
        print("Waiting for server to start...")
        if not wait_for_server():
            output = proc.stdout.read().decode()[-3000:]
            print(f"FAIL: Server did not start.\nOutput:\n{output}")
            sys.exit(1)
        print("Server ready!")

        # Test 1: Sequential requests
        print("\n=== Test 1: Sequential requests ===")
        for i, prompt in enumerate(PROMPTS[:4]):
            result = send_request(prompt, max_tokens=16)
            text = result["choices"][0]["text"]
            print(f"  [{i}] {prompt!r} -> {text!r}")
            assert len(text) > 0, f"Empty response for {prompt!r}"
        print("PASS: Sequential requests")

        # Test 2: Concurrent requests (8 at once)
        print("\n=== Test 2: Concurrent requests (8) ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(8):
                prompt = PROMPTS[i % len(PROMPTS)]
                f = executor.submit(send_request, prompt, 32)
                futures.append((i, prompt, f))

            errors = []
            for i, prompt, f in futures:
                try:
                    result = f.result(timeout=120)
                    text = result["choices"][0]["text"]
                    print(f"  [{i}] {prompt!r} -> {text[:50]!r}...")
                except Exception as e:
                    errors.append((i, prompt, str(e)))
                    print(f"  [{i}] {prompt!r} -> ERROR: {e}")

        if errors:
            print(f"FAIL: {len(errors)} errors")
            sys.exit(1)
        print("PASS: Concurrent requests")

        # Test 3: Staggered requests (mixed prefill/decode batches)
        print("\n=== Test 3: Staggered requests ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(8):
                prompt = PROMPTS[i % len(PROMPTS)]
                delay = 0.0 if i < 4 else 1.0
                def do_req(p=prompt, d=delay):
                    time.sleep(d)
                    return send_request(p, 32)
                f = executor.submit(do_req)
                futures.append((i, prompt, f))

            errors = []
            for i, prompt, f in futures:
                try:
                    result = f.result(timeout=120)
                    text = result["choices"][0]["text"]
                    print(f"  [{i}] {prompt!r} -> {text[:50]!r}...")
                except Exception as e:
                    errors.append((i, prompt, str(e)))
                    print(f"  [{i}] {prompt!r} -> ERROR: {e}")

        if errors:
            print(f"FAIL: {len(errors)} errors")
            sys.exit(1)
        print("PASS: Staggered requests")

        print("\n=== ALL TESTS PASSED ===")

    finally:
        print("\nShutting down server...")
        kill_server(proc)
        print("Done.")


if __name__ == "__main__":
    main()
