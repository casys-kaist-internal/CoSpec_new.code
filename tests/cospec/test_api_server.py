# SPDX-License-Identifier: Apache-2.0
"""CoSpec API server tests with concurrent requests.

These tests launch the OpenAI-compatible API server with CoSpec enabled and
send concurrent requests, which exercises the AsyncLLMEngine code path
(including _process_model_outputs, output remapping, and the two-queue
pipeline with dynamic batch compositions).

This is the same code path used by server.sh + client.sh.

NOTE: Requires NVIDIA MPS. Start with: bash cospec/scripts/start_mps.sh
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import requests

# Default test model (small for fast testing)
DEFAULT_MODEL = "JackFram/llama-68m"

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


def _check_mps_available():
    """Check if MPS is available."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nvidia-cuda-mps-control"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    mps_paths = [
        "/tmp/nvidia-mps/control",
        os.path.join(os.path.dirname(__file__),
                     "../../log/mps/nvidia-mps/control"),
    ]
    mps_pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if mps_pipe_dir:
        mps_paths.insert(0, os.path.join(mps_pipe_dir, "control"))

    for path in mps_paths:
        if os.path.exists(path):
            return True
    return False


def _wait_for_server(port: int, timeout: float = 120.0) -> bool:
    """Wait for the server to become ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def _start_server(model: str, draft_model: str, port: int,
                  num_speculative_tokens: int = 5,
                  enable_chunked_prefill: bool = False) -> subprocess.Popen:
    """Start the CoSpec API server in a subprocess."""
    env = os.environ.copy()
    env.update({
        "COSPEC": "1",
        "VLLM_USE_V1": "0",
    })

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", model,
        "--speculative-config",
        json.dumps({
            "model": draft_model,
            "num_speculative_tokens": num_speculative_tokens,
        }),
        "--seed", "42",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.7",
        "--disable-frontend-multiprocessing",
        "--disable-log-requests",
    ]
    if enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def _kill_server(proc: subprocess.Popen):
    """Kill the server process and all children."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _send_completion_request(port: int, model: str, prompt: str,
                             max_tokens: int = 16) -> dict:
    """Send a single completion request to the server."""
    resp = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 42,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# Skip all tests if MPS is not running
pytestmark = pytest.mark.skipif(
    not _check_mps_available(),
    reason="CoSpec requires NVIDIA MPS. Start with: bash cospec/scripts/start_mps.sh"
)


class TestCoSpecAPIServer:
    """Tests that exercise the CoSpec API server with concurrent requests."""

    @pytest.fixture(autouse=True)
    def _setup_port(self):
        """Assign a unique port for each test."""
        # Use a high port to avoid conflicts
        self.port = 18234

    def test_sequential_requests(self):
        """Test sequential requests through the API server.

        This exercises the AsyncLLMEngine code path with single requests,
        ensuring basic server functionality works with CoSpec.
        """
        model = DEFAULT_MODEL
        proc = _start_server(model, model, self.port)
        try:
            assert _wait_for_server(self.port), \
                f"Server did not start. Output:\n{proc.stdout.read().decode()[:2000]}"

            for prompt in PROMPTS[:4]:
                result = _send_completion_request(
                    self.port, model, prompt, max_tokens=16)
                text = result["choices"][0]["text"]
                assert len(text) > 0, f"Empty response for prompt: {prompt!r}"
        finally:
            _kill_server(proc)

    def test_concurrent_requests(self):
        """Test concurrent requests through the API server.

        This is the key test that matches server.sh + client.sh behavior:
        multiple requests arriving simultaneously, creating mixed batches
        with sequences in different pipeline phases.
        """
        import concurrent.futures

        model = DEFAULT_MODEL
        proc = _start_server(model, model, self.port)
        try:
            assert _wait_for_server(self.port), \
                f"Server did not start. Output:\n{proc.stdout.read().decode()[:2000]}"

            # Send all requests concurrently
            num_prompts = 8
            max_tokens = 32

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_prompts) as executor:
                futures = []
                for i in range(num_prompts):
                    prompt = PROMPTS[i % len(PROMPTS)]
                    f = executor.submit(
                        _send_completion_request,
                        self.port, model, prompt, max_tokens)
                    futures.append((i, prompt, f))

                # Collect results
                results = []
                errors = []
                for i, prompt, f in futures:
                    try:
                        result = f.result(timeout=120)
                        text = result["choices"][0]["text"]
                        results.append((i, prompt, text))
                    except Exception as e:
                        errors.append((i, prompt, str(e)))

            assert not errors, \
                f"Failed requests:\n" + "\n".join(
                    f"  [{i}] {p!r}: {e}" for i, p, e in errors)
            assert len(results) == num_prompts, \
                f"Expected {num_prompts} results, got {len(results)}"

            # All responses should be non-empty
            for i, prompt, text in results:
                assert len(text) > 0, \
                    f"Empty response for prompt [{i}]: {prompt!r}"
        finally:
            _kill_server(proc)

    def test_staggered_requests(self):
        """Test staggered request arrivals.

        Sends requests with small delays to create diverse batch compositions
        (some prefilling while others are decoding), exercising the two-queue
        pipeline's handling of dynamic batch changes.
        """
        import concurrent.futures

        model = DEFAULT_MODEL
        proc = _start_server(model, model, self.port)
        try:
            assert _wait_for_server(self.port), \
                f"Server did not start. Output:\n{proc.stdout.read().decode()[:2000]}"

            num_prompts = 8
            max_tokens = 32
            results = []
            errors = []

            def send_with_delay(idx, prompt, delay):
                time.sleep(delay)
                return _send_completion_request(
                    self.port, model, prompt, max_tokens)

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_prompts) as executor:
                futures = []
                for i in range(num_prompts):
                    prompt = PROMPTS[i % len(PROMPTS)]
                    # Stagger: first 4 immediately, next 4 after 1 second
                    delay = 0.0 if i < 4 else 1.0
                    f = executor.submit(send_with_delay, i, prompt, delay)
                    futures.append((i, prompt, f))

                for i, prompt, f in futures:
                    try:
                        result = f.result(timeout=120)
                        text = result["choices"][0]["text"]
                        results.append((i, prompt, text))
                    except Exception as e:
                        errors.append((i, prompt, str(e)))

            assert not errors, \
                f"Failed requests:\n" + "\n".join(
                    f"  [{i}] {p!r}: {e}" for i, p, e in errors)
            assert len(results) == num_prompts
        finally:
            _kill_server(proc)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
