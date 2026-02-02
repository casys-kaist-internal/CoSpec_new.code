# CoSpec — Co-located Speculative Decoding for vLLM

## What Is CoSpec

CoSpec runs **two vLLM speculative decoding engine instances on the same GPU simultaneously** (colocation). It overlaps target and draft model execution between the two engines, coordinates them via file locks and shared memory, and applies two optimizations: **dynamic colocation switching** and **selective validation**.

This is a fork of vLLM. The base commit is `bbde4701f825fd9f2607a8d5e7e87bc0f0b014c4`.

## Architecture Overview

```
Engine A (primary=True)          Engine B (primary=False)
    |                                 |
    |-- draft_start() ---------->    |-- target_start() -------->
    |   [draft model on GPU]         |   [target model on GPU]
    |   early exit signal <----------|-- target_finish()
    |-- draft_finish()               |
    |-- target_start() -------->     |-- draft_start() --------->
    |   [target model on GPU]        |   [draft model on GPU]
    |-- target_finish() ---------->  |   early exit signal
    |                                |-- draft_finish()
```

Two engines alternate GPU access: while one runs its target model, the other runs its draft model. File locks enforce exclusivity; shared memory passes signals.

## Key Files

### CoSpec Core (`vllm/cospec/`)

| File | Purpose |
|------|---------|
| `cospec_manager.py` | Central coordinator. Manages locks, shared memory signals, profiler, and selective validator. Called from model runners and spec decode worker. |
| `shm.py` | Inter-process shared memory via `UltraDict` (`/dev/shm`). Used for lock signaling and early-exit flags between the two engine processes. |
| `profiler.py` | Top-level profiler facade. Delegates to `ColocationProfiler` and `TilingProfiler`. |
| `colocation_profiler.py` | Profiles colocation vs non-colocation latency. Trains regression models to predict speedup ratio. Contains `CustomColocationModel` and `CustomNonColocationModel`. |
| `tiling_profiler.py` | Profiles target model latency vs token count. Captures the GPU tiling effect (staircase latency at multiples of 8). Trains linear and polynomial regression models. |
| `selective_validator.py` | Predicts per-token acceptance probability from draft model confidence. Generates masks to skip verifying low-probability tokens. |
| `utils.py` | Shared `remove_outliers()` (IQR method) used by both profilers. |

### Modified vLLM Files

| File | CoSpec Changes |
|------|----------------|
| `vllm/envs.py` (lines 738-754) | Defines 6 `COSPEC_*` environment variables |
| `vllm/worker/model_runner.py` (lines 1781-1843) | Calls `cospec_manager.target_start()/finish()` or `draft_start()/finish()` around model execution based on `is_target` flag |
| `vllm/spec_decode/spec_decode_worker.py` | Creates `CospecManager`, wires it to both workers. Integrates selective validation into `_run_speculative_decoding_step`. Early exit checks in `_run_non_driver_rank`. |
| `vllm/spec_decode/multi_step_worker.py` (lines 119-121) | Early exit check: `cospec_manager.check_early_exit_draft()` breaks draft loop |
| `vllm/spec_decode/batch_expansion.py` | Uses `reshape_and_pad` for variable-length proposals (from selective validation) |
| `vllm/spec_decode/top1_proposer.py` | `_adjust_proposal_lens` for early-exited draft sequences |
| `vllm/entrypoints/openai/api_server.py` (line 1462) | `run_server_cospec()` entry point when `COSPEC=1` |
| `vllm/entrypoints/openai/serving_completion_cospec.py` | Orchestrates two engines, profiling, request routing, and dynamic colocation switching |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSPEC` | `0` | Master switch. Enables two-engine colocation. |
| `COSPEC_DYNAMIC_COLOCATION` | `0` | Enables runtime switching between colocation and non-colocation modes based on batch size. |
| `COSPEC_SELECTIVE_VALIDATION` | `0` | Enables selective validation to reduce target model verification workload. |
| `COSPEC_SELECTIVE_VALIDATION_METHOD` | `"tile"` | Mask generation method: `tile`, `linear`, `polynomial`, `threshold`, or `random`. |
| `COSPEC_SELECTIVE_VALIDATION_THRESHOLD` | `0.5` | Cumulative acceptance probability threshold for filtering tokens. |
| `COSPEC_CORRECTNESS_TEST` | `0` | Test mode. When set, original proposals are NOT restored after selective validation (used to verify correctness with reduced proposals). |

## Three Core Techniques

### 1. GPU Colocation with Lock-Based Scheduling

**Where:** `cospec_manager.py`, `model_runner.py`

- Two engine processes (distinguished by `is_primary`) share one GPU.
- `target_start()`/`target_finish()` and `draft_start()`/`draft_finish()` use `fcntl.flock(LOCK_EX)` file locks for GPU exclusivity.
- The driver rank (rank 0) acquires the lock first and publishes its group via shared memory. Non-driver ranks busy-wait until their group is signaled, then acquire the lock.
- **Early exit:** `target_finish()` sets `early_exit_{other_engine} = True` in shared memory. The other engine's draft loop checks `check_early_exit_draft()` and breaks if set. This prevents wasted draft computation.
- Multi-rank coordination: the driver propagates early-exit decisions to non-driver ranks via per-rank shared memory keys.

### 2. Dynamic Colocation

**Where:** `colocation_profiler.py`, `serving_completion_cospec.py`

Colocation helps at low batch sizes (overlapped compute) but hurts at high batch sizes (GPU contention).

**Latency models** (fit via least-squares):
- **Non-colocation:** `T(B, γ) = γ·(α₀ + α₁B + α₂B²) + δ₀ + δ₁Nₜ + δ₂Nₜ²` where `Nₜ = B·(γ+1)`
- **Colocation:** `T(B, γ) = 2·(β₀ + β₁Nₛ + β₂Nₛ²)·(1 + φ₁B + φ₂B²)` where `Nₛ = (B/2)·(γ+1)`

**Profiling** (at startup):
- Batch sizes: `range(8, 129, 8)`, speculative tokens: `range(1, 8)`
- 15 iterations each (5 warmup, 10 recorded), with IQR outlier removal

**Runtime switching** (background task every 0.5s):
- Tracks batch size via EMA (α=0.5)
- Queries `predict_colocation_speedup_ratio(batch_size_ema, num_spec_tokens_ema)`
- Hysteresis: 5 consecutive predictions in the same direction before switching
- Minimum dwell time: 30 seconds between mode switches
- Switching to non-colocation adjusts speculative window to selective validator's EMA; switching to colocation restores it to 7

### 3. Selective Validation

**Where:** `selective_validator.py`, `tiling_profiler.py`

Instead of verifying all K draft tokens, predict which will be rejected and skip them.

**Acceptance predictor:**
1. Collects 50K samples of `(unscaled_temp_probs, min(target_prob/draft_prob, 1))`
2. Trains degree-1 polynomial regression (linear): `P(accept) ≈ f(draft_confidence)`
3. Validates on 20K additional samples (AUROC, ECE)

**Mask generation** (the `tile` method — the default and best):
1. Predict per-token acceptance probability → compute cumulative product along each sequence
2. Flatten all tokens across batch, sort by cumulative acceptance probability (descending)
3. Look up target model latency from tiling profiler for each possible token count
4. Compute `expected_throughput[i] = cumsum(sorted_probs[0:i]) / latency[i]`
5. Find optimal cutoff maximizing expected throughput (starting from where probability drops below threshold)
6. This is a greedy knapsack optimization accounting for the GPU's non-linear tiling latency

**Integration with scoring** (`spec_decode_worker.py:_run_speculative_decoding_step`):
1. Clone original proposals
2. Apply selective validation (modifies proposals in-place, reducing token count)
3. Score the filtered proposals via target model (less work)
4. Restore original proposals for accurate history training
5. Run acceptance/rejection on the scored results

**Tiling effect:** GPU latency is a staircase function of token count because CUDA kernels tile work into blocks (typically multiples of 8). The tiling profiler measures this empirically and precomputes latencies up to 4096 tokens.

## Test Files

| File | What It Tests |
|------|---------------|
| `tests/spec_decode/e2e/test_cospec.py` | Three e2e tests using `JackFram/llama-68m` as both target and draft: basic CoSpec, selective validation (random method), and chunked prefill + selective validation. All use `COSPEC_CORRECTNESS_TEST=1`. |

## Build & Run

- This is a vLLM fork; standard vLLM build (`pip install -e .`).
- Requires `UltraDict` for shared memory (`pip install UltraDict`).
- CoSpec server: set `COSPEC=1` and run the normal `vllm.entrypoints.openai.api_server` — it auto-detects and calls `run_server_cospec()`.
- Tests: `pytest tests/spec_decode/e2e/test_cospec.py` (requires GPU).

## Code Review Notes

### Removed Features (relative to earlier development branches)
- **Consolidated Attention** kernels have been removed. The `csrc/` directory is reverted to the base commit. All related Python code, env vars, and tests are deleted.
- See git history for details. The base commit is `bbde4701f825fd9f2607a8d5e7e87bc0f0b014c4`.
