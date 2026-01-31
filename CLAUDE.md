# Code Review Notes

## Base Commit
Base commit hash: bbde4701f825fd9f2607a8d5e7e87bc0f0b014c4

## Changes Made

### Removed Consolidated Attention
- Reverted `csrc/` directory to base commit (removed consolidated attention kernels)
- Deleted: `csrc/attention/consolidated_attention_kernels.cuh`
- Deleted: `csrc/attention/consolidated_paged_attention_v1.cu`
- Deleted: `csrc/attention/consolidated_paged_attention_v2.cu`
- Reverted: `csrc/attention/attention_kernels.cuh`, `attention_utils.cuh`, `paged_attention_v1.cu`, `paged_attention_v2.cu`, `ops.h`, `torch_bindings.cpp`

### Reverted Attention Python Files
- Reverted `vllm/attention/ops/paged_attn.py` to base commit
- Reverted `vllm/attention/backends/xformers.py` to base commit

### Removed xformers Environment Variable
- Removed `os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"` from `vllm/entrypoints/openai/api_server.py`

### Removed Consolidated Attention Test Files
- Deleted: `tests/kernels/attention/run_consolidated_attention.py`
- Deleted: `tests/kernels/attention/test_consolidated_attention.py`
- Deleted: `run_attention_test.py`
- Deleted: `cospec_benchmarks/run_consolidated_attention.py`
- Removed test functions from `tests/spec_decode/e2e/test_cospec.py`:
  - `test_spec_decode_consolidated_attention`
  - `test_spec_decode_selective_validation_consolidated_attention`
  - `test_spec_decode_chunked_prefill_selective_validation_consolidated_attention`

### Removed Consolidated Attention from vllm Core
- Removed `consolidated_paged_attention_v1` and `consolidated_paged_attention_v2` functions from `vllm/_custom_ops.py`
- Removed `COSPEC_CONSOLIDATED_ATTENTION` env var from `vllm/envs.py`
- Removed `consolidated_lens_tensor` parameter from:
  - `vllm/sequence.py` (ExecuteModelRequest)
  - `vllm/worker/worker_base.py`
  - `vllm/worker/model_runner_base.py`
  - `vllm/worker/model_runner.py`
  - `vllm/attention/backends/abstract.py`
  - `vllm/attention/backends/utils.py`
  - `vllm/spec_decode/target_model_runner.py`
  - `vllm/spec_decode/batch_expansion.py`
  - `vllm/spec_decode/spec_decode_worker.py`
