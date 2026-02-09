# SPDX-License-Identifier: Apache-2.0
"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs.
"""

import glob
import os

import pytest

from .conftest import run_equality_correctness_test_with_env

MAIN_MODEL = "JackFram/llama-68m"


@pytest.fixture(autouse=True)
def cospec_env(monkeypatch):
    """Set VLLM_ATTENTION_BACKEND and clean up shared memory files."""
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "XFORMERS")

    # Cleanup shared memory files before each test
    for f in glob.glob('/tmp/cospec*'):
        try:
            if os.path.isfile(f):
                os.remove(f)
        except OSError:
            pass

    yield

    # Cleanup shared memory files after each test
    for f in glob.glob('/tmp/cospec*'):
        try:
            if os.path.isfile(f):
                os.remove(f)
        except OSError:
            pass


# ── Existing tests ──────────────────────────────────────────────────────

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
    env_vars = {
        "COSPEC_CORRECTNESS_TEST": "1",
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
def test_spec_decode_cospec_selective_validation(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs,
        batch_size: int, output_len: int, seed: int):
    env_vars = {
        "COSPEC_CORRECTNESS_TEST": "1",
        "COSPEC": "1",
        "COSPEC_SELECTIVE_VALIDATION": "1",
        "COSPEC_SELECTIVE_VALIDATION_METHOD": "random",
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
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 5,
            },
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4,
            "max_num_seqs": 4,
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_chunked_prefill_selective_validation(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs,
        batch_size: int, output_len: int, seed: int):
    env_vars = {
        "COSPEC_CORRECTNESS_TEST": "1",
        "COSPEC": "1",
        "COSPEC_SELECTIVE_VALIDATION": "1",
        "COSPEC_SELECTIVE_VALIDATION_METHOD": "random",
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


# ── New tests ───────────────────────────────────────────────────────────

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
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 3,
            },
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cospec_gamma3(vllm_runner, common_llm_kwargs,
                                   per_test_common_llm_kwargs,
                                   baseline_llm_kwargs, test_llm_kwargs,
                                   batch_size: int, output_len: int,
                                   seed: int):
    """Test with gamma=3 (optimal from profiling)."""
    env_vars = {
        "COSPEC_CORRECTNESS_TEST": "1",
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
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("output_len", [64])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cospec_batch1(vllm_runner, common_llm_kwargs,
                                   per_test_common_llm_kwargs,
                                   baseline_llm_kwargs, test_llm_kwargs,
                                   batch_size: int, output_len: int,
                                   seed: int):
    """Test batch=1 — primary latency use case."""
    env_vars = {
        "COSPEC_CORRECTNESS_TEST": "1",
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
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("output_len", [128])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cospec_longer_output(vllm_runner, common_llm_kwargs,
                                          per_test_common_llm_kwargs,
                                          baseline_llm_kwargs,
                                          test_llm_kwargs,
                                          batch_size: int, output_len: int,
                                          seed: int):
    """Test longer output — catches KV cache drift over more decode steps."""
    env_vars = {
        "COSPEC_CORRECTNESS_TEST": "1",
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
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 5,
            },
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4,
            "max_num_seqs": 4,
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cospec_chunked_prefill(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs,
        batch_size: int, output_len: int, seed: int):
    """Test chunked prefill without selective validation."""
    env_vars = {
        "COSPEC_CORRECTNESS_TEST": "1",
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
        "enforce_eager": False,
        "model_name": "JackFram/llama-68m",
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
def test_spec_decode_cospec_cuda_graphs(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs,
        batch_size: int, output_len: int, seed: int):
    """Test CUDA graph record/replay path (enforce_eager=False)."""
    env_vars = {
        "COSPEC_CORRECTNESS_TEST": "1",
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
