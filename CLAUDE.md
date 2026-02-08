# CoSpec — Collaborative Speculative Decoding for vLLM

Base vLLM commit: `bbde4701f825fd9f2607a8d5e7e87bc0f0b014c4`

## Architecture

CoSpec adds three features on top of vLLM's speculative decoding:

1. **Colocation** — Draft and target models share the same GPU via CUDA MPS, enabling overlapped execution.
2. **Dynamic Colocation** — Runtime switching between co-located and non-co-located modes based on predicted speedup ratios from polynomial regression models.
3. **Tiled Selective Validation** — ML-based prediction of token acceptance probability. Low-confidence tokens skip verification, reducing target model batch size.

### Key Files

**CoSpec core** (`vllm/cospec/`):
- `cospec_manager.py` — Central orchestrator: profiling, selective validation, colocation mode, shared memory locks
- `profiler.py` — Unified profiler wrapping colocation and tiling profilers
- `colocation_profiler.py` — Measures colocation vs non-colocation latency, trains polynomial regression models
- `tiling_profiler.py` — Profiles target model latency vs token count, trains regression models for batch-size-aware scheduling
- `selective_validator.py` — Polynomial regression predicting acceptance probability; strategies: `tile`, `linear`, `threshold`, `random`
- `shm.py` — UltraDict-based shared memory for inter-process communication

**vLLM integration**:
- `vllm/spec_decode/spec_decode_worker.py` — Main integration point; initializes `CospecManager`, calls selective validation
- `vllm/spec_decode/batch_expansion.py` — Batch expansion for scoring
- `vllm/entrypoints/openai/api_server.py` — CoSpec server entry point with dual engine clients
- `vllm/entrypoints/openai/serving_completion_cospec.py` — CoSpec completion serving with dynamic colocation, profiling

**Tests**:
- `tests/spec_decode/e2e/test_cospec.py` — E2E tests (basic, selective validation, chunked prefill)

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `COSPEC` | bool | `False` | Enable CoSpec mode |
| `COSPEC_DYNAMIC_COLOCATION` | bool | `False` | Enable dynamic colocation switching |
| `COSPEC_SELECTIVE_VALIDATION` | bool | `False` | Enable selective validation |
| `COSPEC_SELECTIVE_VALIDATION_METHOD` | str | `"tile"` | Method: `tile`, `linear`, `threshold`, `random` |
| `COSPEC_SELECTIVE_VALIDATION_THRESHOLD` | float | `0.5` | Confidence threshold for validation |
| `COSPEC_CORRECTNESS_TEST` | bool | `False` | Correctness testing mode |

## Docker Environment

- Container: `cospec-vllm`
- Bind mount: host `/mnt/sdb/sjchoi/CoSpec_new.code` = container `/workspace`
- Files are always in sync (no `docker cp` needed)
- Must set `CUDA_VISIBLE_DEVICES=0` (multi-GPU host, MPS restricts to GPU 0)
- MPS must be running: `nvidia-cuda-mps-control -d`

## Build

```bash
docker exec -w /workspace cospec-vllm pip install -e .
```

## Run Tests

```bash
docker exec -w /workspace -e VLLM_USE_V1=0 -e CUDA_VISIBLE_DEVICES=0 \
  cospec-vllm python3 -m pytest tests/spec_decode/e2e/test_cospec.py -v --timeout=300
```

## Run Server

```bash
# CoSpec mode (requires two engine processes)
COSPEC=1 VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
  --model <target_model> --speculative-model <draft_model> \
  --num-speculative-tokens 5

# With selective validation
COSPEC=1 COSPEC_SELECTIVE_VALIDATION=1 COSPEC_SELECTIVE_VALIDATION_METHOD=tile \
  VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server ...

# With dynamic colocation
COSPEC=1 COSPEC_DYNAMIC_COLOCATION=1 VLLM_USE_V1=0 \
  python -m vllm.entrypoints.openai.api_server ...
```

## Directory Structure

```
vllm/cospec/           # CoSpec core modules
vllm/spec_decode/      # Speculative decoding (modified for CoSpec)
vllm/entrypoints/      # API server with CoSpec serving
cospec/
  data/                # Datasets (ShareGPT)
  backup/              # Old benchmark scripts and results
tests/spec_decode/e2e/ # E2E tests including test_cospec.py
```

## Requirements

- `VLLM_USE_V1=0` (CoSpec requires V0 engine)
- CUDA MPS enabled for colocation
- XFORMERS attention backend for tests
