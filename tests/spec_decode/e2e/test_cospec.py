# SPDX-License-Identifier: Apache-2.0
"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs.

NOTE: These tests require NVIDIA MPS to be running for SM partitioning.
Start MPS with: bash cospec/scripts/start_mps.sh
"""

import os
import pytest

from .conftest import run_equality_correctness_test_with_env

MAIN_MODEL = "JackFram/llama-68m"


def _check_mps_available():
    """Check if MPS is available by looking for the MPS control daemon."""
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nvidia-cuda-mps-control"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def init_cospec():
    # Cleanup previous CoSpec IPC handles
    try:
        from vllm.cospec import cleanup_cospec_resources
        cleanup_cospec_resources()
    except Exception as e:
        print("CoSpec IPC cleanup failed: %s" % str(e))


# Skip all CoSpec tests if MPS is not running
pytestmark = pytest.mark.skipif(
    not _check_mps_available(),
    reason="CoSpec requires NVIDIA MPS. Start with: bash cospec/scripts/start_mps.sh"
)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "enforce_eager": True,
        "model_name": "JackFram/llama-68m",
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Identical models.
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 5,
            },
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cospec(vllm_runner, common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, test_llm_kwargs,
                                batch_size: int, output_len: int, seed: int):
    init_cospec()
    env_vars = {
        "COSPEC": "1",
    }
    run_equality_correctness_test_with_env(vllm_runner,
                                            common_llm_kwargs,
                                            per_test_common_llm_kwargs,
                                            baseline_llm_kwargs,
                                            test_llm_kwargs,
                                            batch_size,
                                            max_output_len=output_len,
                                            seed=seed,
                                            temperature=0.0,
                                            env_vars=env_vars)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "enforce_eager": True,
        "model_name": "JackFram/llama-68m",
        "enable_chunked_prefill": True,
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 5,
            },
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cospec_chunked_prefill(
        vllm_runner, common_llm_kwargs,
        per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs,
        batch_size: int, output_len: int, seed: int):
    init_cospec()
    env_vars = {
        "COSPEC": "1",
    }
    run_equality_correctness_test_with_env(vllm_runner,
                                            common_llm_kwargs,
                                            per_test_common_llm_kwargs,
                                            baseline_llm_kwargs,
                                            test_llm_kwargs,
                                            batch_size,
                                            max_output_len=output_len,
                                            seed=seed,
                                            temperature=0.0,
                                            env_vars=env_vars)
