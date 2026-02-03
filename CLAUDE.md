# CoSpec v2 — Specialized Workers with Shared KV Cache for vLLM

## What Is CoSpec

CoSpec runs a **target process** and a **draft process** on the same GPU via MPS (Multi-Process Service). The target process owns the scheduler and orchestrator; the draft process is a lightweight RPC worker. SM (Streaming Multiprocessor) partitioning via `libsmctrl` enables concurrent execution with zero-cost mode switching between three modes: **AR**, **Vanilla SD**, and **Colocated SD**.

This is a fork of vLLM. The base commit is `bbde4701f825fd9f2607a8d5e7e87bc0f0b014c4`.

## Architecture Overview

```
API Entrypoint (thin, single queue)
        |
  Target Process (owns scheduler + orchestrator)
    ├── Target model weights
    ├── Shared KV cache (allocated here, exported via CUDA IPC)
    ├── Shared logit buffer (allocated here, exported via CUDA IPC)
    ├── CostModel: decide(B, α, S) → (mode, γ, r)
    ├── SMController (libsmctrl)
    └── Orchestrator (replaces SpecDecodeWorker + serving_completion_cospec)
        |
  Draft Process (RPC worker, receives commands)
    ├── Draft model weights
    ├── Shared KV cache (opened via CUDA IPC)
    ├── Shared logit buffer (opened via CUDA IPC)
    └── SMController (libsmctrl)
```

Both processes on same GPU via MPS.

## Three Modes (per-step decision, zero-cost switching)

| Mode | Target SMs | Draft SMs | Flow |
|------|-----------|-----------|------|
| AR | 100% | idle | target.forward(B) |
| Vanilla SD | 100% sequential | 100% sequential | draft.propose(B,γ) → target.verify(B) |
| Colocated SD | r% | (1-r)% | draft.propose ∥ target.verify (concurrent, two-queue pipelined) |

## Key Files

### CoSpec Core (`vllm/cospec/`)

| File | Purpose |
|------|---------|
| `cospec_manager.py` | Central coordinator. Creates SMController, manages shared memory signals. `cleanup_cospec_resources()` removes stale IPC handles from `/dev/shm` and `/tmp`. |
| `sm_controller.py` | ctypes wrapper around `cospec/csrc/build/libsmctrl.so`. Provides `set_partition(stream, ratio)` and `set_full_gpu(stream)` for SM partitioning. |
| `shared_kv_cache.py` | `SharedKVCacheAllocator`: Target allocates KV tensors, exports CUDA IPC handles to `/dev/shm`. Draft opens handles. |
| `shared_logit_buffer.py` | `SharedLogitBuffer`: Pre-allocated GPU buffer `[max_batch, max_spec_tokens, vocab_size]`. Draft writes logits, target reads for verification. |
| `worker_rpc.py` | `DraftWorkerRPC` (client in target process) and `DraftWorkerServer` (in draft process). Commands over `multiprocessing.Pipe`, large data via shared GPU memory. |
| `cost_model.py` | `CostModel`: Analytical model with `decide(B, α, S) → (Mode, γ, r)`. **Currently hardcoded to always return COLOCATED_SD** (latency formulas are placeholders). |
| `orchestrator.py` | `CoSpecOrchestrator`: Two-queue colocated SD pipelining. Sequences alternate between `draft_queue` and `verify_queue` for true concurrent execution. |

### Modified vLLM Files

| File | CoSpec Changes |
|------|----------------|
| `vllm/envs.py` | Defines `COSPEC` environment variable (master switch) |
| `vllm/worker/cache_engine.py` | `shared_mode` parameter in `_allocate_kv_cache()`: `"owner"` (allocate + export IPC), `"client"` (open IPC), `None` (legacy) |
| `vllm/worker/model_runner.py` | Uses `sm_controller.set_partition()` / `set_full_gpu()` around model execution |
| `vllm/spec_decode/spec_decode_worker.py` | Creates `CospecManager` with SM controller, wires to workers. Handles two-queue partial outputs via `_cospec_skip`. Calls `orchestrator.remove_sequence()` on finished requests. |
| `vllm/engine/llm_engine.py` | Handles CoSpec empty outputs (no-op bootstrap steps) and partial outputs (`_cospec_skip` indices for draft-phase sequences). |
| `vllm/entrypoints/openai/api_server.py` | `run_server_cospec()` entry point. Calls `cleanup_cospec_resources()` on startup. |
| `vllm/config.py` | `SpeculativeConfig.is_primary` field |
| `vllm/engine/arg_utils.py` | `is_primary` parameter threading |

### Native Code (`cospec/csrc/`)

| File | Purpose |
|------|---------|
| `src/libsmctrl_core.c` | Core SM control via CUDA driver debug hooks |
| `src/libsmctrl.h` | Header: `set_stream_mask()`, `get_tpc_info()`, `make_mask()` |
| `src/libsmctrl_validator.cu` | CUDA kernel to verify SM mask configuration |
| `build/libsmctrl.so` | Pre-built shared library |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSPEC` | `0` | Master switch. Enables CoSpec with SM partitioning and MPS. |

**Important**: Do NOT set `VLLM_ATTENTION_BACKEND`. Let vLLM auto-select the optimal backend (FLASH_ATTN). Never force XFORMERS — it is slower.

## Colocated SD Data Flow — Two-Queue Model

```
Orchestrator._step_colocated_sd():
  0. Move pending_pool → draft_queue (load-balanced entries from last step)
  1. Split scheduler batch B into:
     - draft_seqs:   in draft_queue (need proposals)
     - verify_seqs:  in verify_queue (have proposals, need verification)
     - prefill_seqs: prompt sequences
     - new seqs:     just finished prefill → load-balance into draft or pending

  2. Bootstrap (verify_queue empty):
     - Draft proposes draft_seqs → move to verify_queue → return []

  3. Concurrent phase (verify_queue non-empty):
     a. sm_controller.set_partition(target=r, draft=1-r)
     b. draft_rpc.propose_async(draft_seqs)  ‖  target.score(verify_seqs)
     c. Barrier: torch.cuda.synchronize()

  4. Verify verify_seqs proposals → output (seq_ids stored on orchestrator.last_output_seq_ids)

  5. Rotate queues:
     - verified seqs → draft_queue
     - drafted seqs  → verify_queue
```

### Sequence Lifecycle
```
Prefill → [load balance] → draft_queue (or pending 1 step)
  → [proposed] → verify_queue → [verified, output emitted] → draft_queue → ...
```

### Engine Integration for Partial Outputs
- Empty output (`[]`): engine skips processing, reschedules immediately
- Partial output (`_cospec_skip` on SpecDecodeWorker instance, read by engine via `_get_cospec_skip()`): engine skips draft-phase sequences, processes only verify-phase sequences. Output remapped with placeholders for skipped indices.
- `remove_sequence(seq_id)`: called from `_track_finished_requests()` to clean up queue state for finished/preempted sequences

## Implementation Progress (v2 wiring)

### DONE — Code written and tested (22 tests pass, 8/8 correctness)
1. **`shared_logit_buffer.py`**: Fixed double `_share_cuda_()` bug (line 68/72).
2. **`cost_model.py`**: `decide()` hardcoded to always return `COLOCATED_SD` mode.
3. **`worker_rpc.py`**: Added `propose_async()` / `propose_collect()` to `DraftWorkerRPC`. Wired `DraftWorkerServer._handle_propose()` to call `self._worker.get_spec_proposals()` and return CPU tensors over pipe.
4. **`orchestrator.py`**: Two-queue colocated SD with load-balanced entry:
   - `_draft_queue`, `_verify_queue`, `_pending_pool` replace `_prev_batch`/`_prev_proposals`
   - `_slice_proposal()`: extract single-sequence from batched proposals
   - `_merge_proposals()`: combine individual proposals back into batch
   - `_run_prefills_only()`: handle prefills when verify_queue is empty
   - `flush()`: drain verify_queue on shutdown
   - `remove_sequence(seq_id)`: clean up finished/preempted sequences
   - `_step_colocated_sd()`: full two-queue concurrent pipeline
   - All wired methods: `_score_proposals()`, `_verify()`, `_create_output()`, `_sync_draft_prefills()`
5. **`spec_decode_worker.py`**:
   - Added `self.orchestrator` and `self._draft_process` fields to `__init__`.
   - `_init_cospec_draft_process()` spawns draft as `mp.Process` (spawn context) with its own CUDA context for true MPS concurrency. Called from `initialize_cache()` (not `init_device()`) so KV cache handles and model weights are ready.
   - Top-level `_draft_process_entry()`: initializes CUDA, distributed env (separate port), loads model via CUDA IPC (`SharedMemoryModelLoader`), imports shared KV cache (`shared_mode="client"`), runs `DraftWorkerServer.serve()`.
   - `_assert_mps_running()` check when `COSPEC=1` — raises RuntimeError with instructions if MPS daemon not detected.
   - Ready/error handshake via pipe: child sends `("READY", pid)` or `("ERROR", traceback)` before parent proceeds.
   - `__del__` terminates/kills draft process with timeout, calls `cleanup_cospec_resources()`.
   - Saves `_draft_worker_kwargs` in `create_worker()` for pickling to the draft process.
   - Clears `static_forward_context` in draft process to avoid duplicate attention layer names.
   - `execute_model()`: delegates to `orchestrator.step()`, reads `orchestrator.last_output_seq_ids` to compute `_cospec_skip` for partial outputs.
   - `_track_finished_requests()`: calls `orchestrator.remove_sequence()` for finished seq_ids.
6. **`worker.py`**: `_init_cache_engine()` checks `cospec_shared_mode` attribute first ("owner"/"client"), falls back to "owner" when `cospec_manager` is present.
7. **`shared_kv_cache.py`**: Fixed `_import_from_handles()` to create CUDA tensor before `set_()` (was creating CPU tensor, causing device mismatch in cross-process IPC).
8. **`llm_engine.py`**: Handles empty CoSpec output (returns early) and `_cospec_skip` via `_get_cospec_skip()` (reads from worker through executor chain, skips draft-phase sequences, remaps outputs with placeholders).
9. **`cospec_manager.py`**: `cleanup_cospec_resources()` removes stale `/dev/shm/cospec_*` and `/tmp/cospec_*`. Called automatically in `CospecManager.__init__()`. `SharedMemory.cleanup()` for explicit teardown.

### TODO — Organized by Priority

#### P1: Optimization — Performance improvements
1. **Cost model latency formulas**: `cost_model.py` `decide()` is hardcoded to always return `COLOCATED_SD`. The skeleton latency formulas (`_latency_ar`, `_latency_vanilla_sd`, `_latency_colocated_sd`) need coefficients from profiling to enable actual per-step mode selection.
2. **SM ratio tuning**: Hardcoded `default_sm_ratio=0.7`. Should be profiled per model pair.
3. **`cospec_manager.target_sm_ratio` feedback from orchestrator**: Currently hardcoded 1.0. The orchestrator controls SM partitions directly via `sm_controller` but doesn't update `cospec_manager.target_sm_ratio`, which `model_runner.py` reads. Only matters for the non-orchestrator code path.
4. **CUDA graph support**: E2E tests use `enforce_eager=True`. Need to verify CoSpec works with CUDA graphs enabled for max performance.

#### P2: Robustness — Edge cases and cleanup
5. **Process cleanup on shutdown**: `SpecDecodeWorker.__del__` calls `orchestrator.shutdown()` but this can race with interpreter shutdown (logging error on closed file). Need a proper shutdown hook earlier in the lifecycle (e.g., in engine shutdown).
6. **`SharedMemoryModelLoader._inprocess_state_dicts` cleanup**: Class-level dict caches state_dicts forever. Should be cleaned up when the engine is destroyed to free GPU memory for the shared weight references.
7. **Shared memory instance_id collision**: `shared_kv_cache.py` and `shared_logit_buffer.py` use `instance_id="default"`. Multiple CoSpec instances on same machine will collide. Should use PID or UUID.
8. **E2E test coverage**: Two E2E tests with identical models (`JackFram/llama-68m` for both target and draft): one without chunked prefill, one with. Need tests with: (a) different target/draft models, (b) larger batch sizes, (c) greedy vs sampling. Note: running both E2E tests in the same pytest process fails due to GPU memory not being freed between tests (pre-existing issue); run them in separate processes.
9. **UltraDict cleanup at exit**: `FileNotFoundError: '/cospec_shared_3'` during interpreter shutdown. Non-fatal but noisy. Need to unlink shared memory before `cleanup_cospec_resources()` deletes it.

### RECENTLY FIXED
7. **`sm_controller.py`**: Fixed segfault when `set_stream_mask` was called with the default CUDA stream (handle=0). Now falls back to `set_global_mask` for default stream. Added graceful `PermissionError` handling when MPS is not available (logs warning once, then silently skips).
8. **`spec_decode_worker.py`**: Added `__del__` cleanup to shut down the orchestrator and draft thread on garbage collection.
9. **`test_cost_model.py`**: Updated `test_ar_mode_has_zero_gamma` → `test_hardcoded_colocated_mode` to match the hardcoded always-colocated behavior.
10. **`test_sm_controller.py`**: Added `PermissionError` handling for tests that need MPS privileges. Added `test_set_partition_explicit_stream` for non-default stream testing.
11. **`loader.py` (SharedMemoryModelLoader)**: Fixed `CUDA error: invalid resource handle` when target and draft models are in the same process. Added in-process state_dict caching (`_inprocess_state_dicts` class variable) to avoid CUDA IPC for intra-process sharing. Draft model now loads from cached state_dict in ~6ms instead of failing.
12. **`orchestrator.py`**: Replaced `_prev_batch`/`_prev_proposals` pipeline with two-queue model (`_draft_queue`, `_verify_queue`, `_pending_pool`). Load-balanced entry for new decode sequences. `flush()` drains pipeline on shutdown. `remove_sequence()` for cleanup.
13. **All tests passing**: 22 tests pass (20 unit + 2 E2E). Run E2E tests in separate processes to avoid GPU memory leak. **Important**: Set `VLLM_USE_V1=0` for spec decode tests (V1 engine doesn't support spec decoding): `docker exec -w /workspace/vllm cospec-vllm python3 -m pytest tests/cospec/ -v && docker exec -w /workspace/vllm -e VLLM_USE_V1=0 cospec-vllm python3 -m pytest tests/spec_decode/e2e/test_cospec.py::test_spec_decode_cospec -v && docker exec -w /workspace/vllm -e VLLM_USE_V1=0 cospec-vllm python3 -m pytest tests/spec_decode/e2e/test_cospec.py::test_spec_decode_cospec_chunked_prefill -v`
14. **Chunked prefill support in orchestrator** (was P0): `_create_output()` now uses real `proposal_lens` from draft proposals instead of hardcoding gamma. Added `_sync_draft_prefills()` to sync draft KV cache for prefill sequences after target scoring (mirrors `spec_decode_worker.py:921-936`). Added `EXECUTE_PREFILL` RPC command in `worker_rpc.py`. Added E2E test with `enable_chunked_prefill=True`.
15. **IPC cleanup**: `cleanup_cospec_resources()` in `cospec_manager.py` removes stale `/dev/shm/cospec_*` and `/tmp/cospec_*`. Called from `api_server.py` on startup, `spec_decode_worker.__del__`, and test `init_cospec()`.
16. **Removed forced XFORMERS backend**: All CoSpec scripts and tests now use the default attention backend (FLASH_ATTN) for max performance.
17. **Fixed user-specific paths**: `run_docker.sh` uses `$HOME` instead of hardcoded user path for HF cache.
18. **Fixed `proposal_probs` None bug**: `_materialize_probs()` reads probs from `SharedLogitBuffer` before `_slice_proposal()`, preventing `None` probs in merged proposals.
19. **Fixed SamplerOutput attribute error**: `SamplerOutput` is a `msgspec.Struct` that doesn't allow arbitrary attributes. Moved `_cospec_seq_ids` to `orchestrator.last_output_seq_ids` and `_cospec_skip` to `SpecDecodeWorker._cospec_skip`. Engine reads skip via `_get_cospec_skip()` method.
20. **Fixed stale IPC handles on startup**: `CospecManager.__init__()` now calls `cleanup_cospec_resources()` to remove stale handles before model loading begins.
21. **Correctness verified**: 8/8 prompts match baseline output exactly (greedy decoding, `JackFram/llama-68m`, batch_size=8, max_tokens=32). Two-queue pipeline produces identical results to non-CoSpec baseline.

### Known Hardcoded Values
- `cost_model.py`: Always returns `COLOCATED_SD` mode, `sm_ratio=0.7`, `gamma=5`
- `cost_model.py`: EMA coefficients `alpha=0.8`, `ema_weight=0.3`, `batch_ema_weight=0.5` (untuned)
- `cost_model.py`: Latency formula coefficients are placeholders (never profiled)
- `shared_kv_cache.py` / `shared_logit_buffer.py`: `instance_id="default"` (collision risk)
- `cospec_manager.py`: `target_sm_ratio=1.0` never updated by orchestrator

### Skeleton Methods Status
- ~~`CoSpecOrchestrator._score_proposals()`~~ → WIRED
- ~~`CoSpecOrchestrator._verify()`~~ → WIRED
- ~~`CoSpecOrchestrator._create_output()`~~ → WIRED
- ~~`DraftWorkerServer._handle_propose()`~~ → WIRED
- `CostModel._latency_ar/vanilla_sd/colocated_sd()` → NOT NEEDED (hardcoded always-colocated)

## Test Files

| File | What It Tests |
|------|---------------|
| `tests/spec_decode/e2e/test_cospec.py` | E2E tests using `JackFram/llama-68m` as both target and draft |
| `tests/cospec/test_cost_model.py` | Unit tests for cost model decision logic |
| `tests/cospec/test_sm_controller.py` | Unit + GPU integration tests for SM controller |
| `tests/cospec/test_shared_kv_cache.py` | Tests for shared KV cache allocation/cleanup |
| `tests/cospec/test_worker_rpc.py` | Tests for RPC pipe communication |

## Build & Run

- This is a vLLM fork; standard vLLM build (`pip install -e .`).
- Build libsmctrl: `cd cospec/csrc && mkdir -p build && cd build && cmake .. && make`
- Requires `UltraDict` for shared memory (`pip install UltraDict`).
- CoSpec server: set `COSPEC=1` and run the normal `vllm.entrypoints.openai.api_server`.
- Tests: `pytest tests/cospec/` (unit tests) or `pytest tests/spec_decode/e2e/test_cospec.py` (requires GPU).
- Run in Docker: `docker exec -w /workspace/vllm cospec-vllm python3 -m pytest tests/cospec/`
- **Do NOT set `VLLM_ATTENTION_BACKEND`** — let vLLM auto-select FLASH_ATTN for best performance.

## Code Review Notes

### Removed Features (v1 → v2 migration)
- **Lock-based GPU scheduling** (`fcntl.flock`) replaced by SM partitioning via `libsmctrl`
- **Dual-engine architecture** (two full vLLM engines) replaced by single target engine + lightweight draft RPC worker
- **ColocationProfiler** regression models replaced by analytical `CostModel`
- **Dynamic colocation switching** (EMA + hysteresis) replaced by per-step cost model decisions
- **`serving_completion_cospec.py`** dual-engine routing removed; single engine with orchestrator
- **Selective validation** removed (can be re-added to orchestrator if needed)
- **Consolidated Attention** kernels removed in earlier cleanup
- The base commit is `bbde4701f825fd9f2607a8d5e7e87bc0f0b014c4`.
