# SPDX-License-Identifier: Apache-2.0
"""CoSpec E2E tests comparing CoSpec output with AR (autoregressive) baseline.

These tests verify that CoSpec produces identical outputs to standard AR decoding
when using greedy sampling (temperature=0).

NOTE: These tests require NVIDIA MPS to be running for SM partitioning.
Start MPS with: bash cospec/scripts/start_mps.sh

NOTE: Each test runs AR and CoSpec in separate subprocesses to avoid GPU memory
leaks between runs (a known vLLM limitation).
"""

import os
import sys
import json
import subprocess
import pytest
from itertools import cycle


PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "San Francisco is know for its",
    "Facebook was created in 2004 by",
    "Curious George is a",
    "Python 3.11 brings improvements to its",
]

# Default test model (small for fast testing)
DEFAULT_MODEL = "JackFram/llama-68m"


def _check_mps_available():
    """Check if MPS is available by looking for the MPS control daemon or socket."""
    # Check for MPS process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nvidia-cuda-mps-control"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    # Check for MPS control socket
    mps_paths = [
        "/tmp/nvidia-mps/control",
        os.path.join(os.path.dirname(__file__), "../../log/mps/nvidia-mps/control"),
    ]
    mps_pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if mps_pipe_dir:
        mps_paths.insert(0, os.path.join(mps_pipe_dir, "control"))

    for path in mps_paths:
        if os.path.exists(path):
            return True
    return False


def _run_in_subprocess(script: str, env: dict = None) -> str:
    """Run a Python script in a subprocess and return its stdout."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed:\n{result.stderr}")

    return result.stdout


def _extract_json_output(stdout: str) -> list:
    """Extract JSON output from subprocess stdout.

    vLLM logs to stdout, so we need to find the JSON line which starts with '[['.
    """
    for line in stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('[[') and line.endswith(']]'):
            return json.loads(line)
    # Fallback to last line
    return json.loads(stdout.strip().split('\n')[-1])


def run_ar_baseline(model: str, prompts: list, max_tokens: int, seed: int) -> list:
    """Run autoregressive baseline in subprocess."""
    script = f'''
import json
from vllm import LLM, SamplingParams

prompts = {prompts!r}
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens={max_tokens},
    seed={seed},
)

llm = LLM(
    model="{model}",
    enforce_eager=True,
)
outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
results = [(out.outputs[0].text, list(out.outputs[0].token_ids)) for out in outputs]
print(json.dumps(results))
'''
    # Run without COSPEC
    env = {"COSPEC": "0", "VLLM_USE_V1": "0"}
    stdout = _run_in_subprocess(script, env)
    return _extract_json_output(stdout)


def run_cospec(model: str, draft_model: str, prompts: list, max_tokens: int,
               seed: int, num_speculative_tokens: int = 5,
               enable_chunked_prefill: bool = False) -> list:
    """Run CoSpec in subprocess."""
    script = f'''
import json
from vllm import LLM, SamplingParams

# Cleanup CoSpec IPC handles
try:
    from vllm.cospec import cleanup_cospec_resources
    cleanup_cospec_resources()
except Exception:
    pass

prompts = {prompts!r}
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens={max_tokens},
    seed={seed},
)

llm = LLM(
    model="{model}",
    enforce_eager=True,
    speculative_config={{
        "model": "{draft_model}",
        "num_speculative_tokens": {num_speculative_tokens},
    }},
    enable_chunked_prefill={enable_chunked_prefill},
)
outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
results = [(out.outputs[0].text, list(out.outputs[0].token_ids)) for out in outputs]
print(json.dumps(results))
'''
    env = {"COSPEC": "1", "VLLM_USE_V1": "0"}
    stdout = _run_in_subprocess(script, env)
    return _extract_json_output(stdout)


def compare_outputs(ar_outputs, cospec_outputs, prompts):
    """Compare AR baseline outputs with CoSpec outputs."""
    assert len(ar_outputs) == len(cospec_outputs), \
        f"Output count mismatch: AR={len(ar_outputs)}, CoSpec={len(cospec_outputs)}"

    mismatches = []
    for i, (ar_out, cospec_out, prompt) in enumerate(zip(ar_outputs, cospec_outputs, prompts)):
        ar_text, ar_tokens = ar_out
        cospec_text, cospec_tokens = cospec_out

        if ar_tokens != cospec_tokens:
            mismatches.append({
                "index": i,
                "prompt": prompt,
                "ar_text": ar_text,
                "cospec_text": cospec_text,
                "ar_tokens": ar_tokens,
                "cospec_tokens": cospec_tokens,
            })

    if mismatches:
        msg = f"Found {len(mismatches)} mismatches:\n"
        for m in mismatches:
            msg += f"\n[{m['index']}] Prompt: {m['prompt']!r}\n"
            msg += f"    AR:     {m['ar_text']!r}\n"
            msg += f"    CoSpec: {m['cospec_text']!r}\n"
        pytest.fail(msg)


# Skip all CoSpec tests if MPS is not running
pytestmark = pytest.mark.skipif(
    not _check_mps_available(),
    reason="CoSpec requires NVIDIA MPS. Start with: bash cospec/scripts/start_mps.sh"
)


class TestCoSpecE2E:
    """E2E tests comparing CoSpec with AR baseline."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("max_tokens", [16, 32])
    def test_cospec_vs_ar_greedy(self, batch_size: int, max_tokens: int):
        """Test CoSpec produces identical output to AR baseline with greedy decoding."""
        model = DEFAULT_MODEL
        draft_model = DEFAULT_MODEL

        prompts = [prompt for prompt, _ in zip(cycle(PROMPTS), range(batch_size))]

        # Run AR baseline (in subprocess)
        ar_outputs = run_ar_baseline(model, prompts, max_tokens, seed=42)

        # Run CoSpec (in subprocess)
        cospec_outputs = run_cospec(model, draft_model, prompts, max_tokens, seed=42)

        # Compare
        compare_outputs(ar_outputs, cospec_outputs, prompts)

    @pytest.mark.parametrize("batch_size", [8])
    @pytest.mark.parametrize("max_tokens", [32])
    def test_cospec_vs_ar_chunked_prefill(self, batch_size: int, max_tokens: int):
        """Test CoSpec with chunked prefill produces identical output to AR baseline."""
        model = DEFAULT_MODEL
        draft_model = DEFAULT_MODEL

        prompts = [prompt for prompt, _ in zip(cycle(PROMPTS), range(batch_size))]

        # Run AR baseline
        ar_outputs = run_ar_baseline(model, prompts, max_tokens, seed=42)

        # Run CoSpec with chunked prefill
        cospec_outputs = run_cospec(
            model, draft_model, prompts, max_tokens, seed=42,
            enable_chunked_prefill=True
        )

        # Compare
        compare_outputs(ar_outputs, cospec_outputs, prompts)

    @pytest.mark.parametrize("num_speculative_tokens", [1, 3, 5, 7])
    def test_cospec_different_gamma(self, num_speculative_tokens: int):
        """Test CoSpec with different speculation lengths (gamma)."""
        model = DEFAULT_MODEL
        draft_model = DEFAULT_MODEL
        batch_size = 4
        max_tokens = 32

        prompts = [prompt for prompt, _ in zip(cycle(PROMPTS), range(batch_size))]

        # Run AR baseline
        ar_outputs = run_ar_baseline(model, prompts, max_tokens, seed=42)

        # Run CoSpec with different gamma
        cospec_outputs = run_cospec(
            model, draft_model, prompts, max_tokens, seed=42,
            num_speculative_tokens=num_speculative_tokens
        )

        # Compare
        compare_outputs(ar_outputs, cospec_outputs, prompts)


class TestCoSpecStress:
    """Stress tests for CoSpec."""

    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("max_tokens", [64])
    def test_cospec_larger_batch(self, batch_size: int, max_tokens: int):
        """Test CoSpec with larger batch size."""
        model = DEFAULT_MODEL
        draft_model = DEFAULT_MODEL

        prompts = [prompt for prompt, _ in zip(cycle(PROMPTS), range(batch_size))]

        ar_outputs = run_ar_baseline(model, prompts, max_tokens, seed=42)
        cospec_outputs = run_cospec(model, draft_model, prompts, max_tokens, seed=42)
        compare_outputs(ar_outputs, cospec_outputs, prompts)


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"] + sys.argv[1:])
