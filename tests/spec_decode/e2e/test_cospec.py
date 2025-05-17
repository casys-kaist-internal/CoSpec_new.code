# SPDX-License-Identifier: Apache-2.0
"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs.
"""

import pytest
import os

from .conftest import run_equality_correctness_test_with_env

MAIN_MODEL = "JackFram/llama-68m"

def init_cospec():
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    # Cleanup previous shared memory files on server start
    try:
        import glob
        shm_files = glob.glob('/dev/shm/cospec*')
        for f in shm_files:
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except Exception as e:
                print("Failed to remove %s: %s", f, str(e))
        print("Cleaned up %d shared memory files from previous runs", len(shm_files))
    except Exception as e:
        print("Shared memory cleanup failed: %s", str(e))

@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "enforce_eager": False,
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
def test_spec_decode_consolidated_attention(vllm_runner, common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, test_llm_kwargs,
                                batch_size: int, output_len: int, seed: int):
    init_cospec()
    env_vars = {"COSPEC_CONSOLIDATED_ATTENTION": "1"}
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
        "enforce_eager": False,
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
def test_spec_decode_selective_validation(vllm_runner, common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, test_llm_kwargs,
                                batch_size: int, output_len: int, seed: int):
    init_cospec()
    env_vars = {
        "COSPEC": "1",
        "COSPEC_SELECTIVE_VALIDATION_CORRECTNESS_TEST": "1"
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
        "enforce_eager": False,
        "model_name": "JackFram/llama-68m",
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Identical models.
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 7,
            },
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("output_len", [32, 64])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_selective_validation_consolidated_attention(vllm_runner, common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, test_llm_kwargs,
                                batch_size: int, output_len: int, seed: int):
    init_cospec()
    env_vars = {
        "COSPEC": "1",
        "COSPEC_SELECTIVE_VALIDATION_CORRECTNESS_TEST": "1",
        "COSPEC_CONSOLIDATED_ATTENTION": "1"
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
