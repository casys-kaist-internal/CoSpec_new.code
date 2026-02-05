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
| `__init__.py` | Module exports and `cleanup_cospec_resources()` for removing stale IPC handles. |
| `sm_controller.py` | `SMController`: ctypes wrapper for libsmctrl SM partitioning. `CospecManager`: creates controller, holds config. |
| `orchestrator.py` | `CoSpecOrchestrator`: Two-queue colocated SD pipelining. `Mode` enum. Always uses colocated mode with SM ratio 0.7. |
| `worker_rpc.py` | `DraftWorkerRPC` (client) and `DraftWorkerServer` (draft process). Commands over pipe, large data via shared GPU memory. |
| `shared_memory.py` | `SharedKVCache` and `SharedLogitBuffer`: CUDA IPC for sharing GPU tensors between target and draft processes. |

### Modified vLLM Files

| File | CoSpec Changes |
|------|----------------|
| `vllm/envs.py` | Defines `COSPEC` environment variable (master switch). `VLLM_USE_V1` defaults to `1` (non-CoSpec users get V1). |
| `vllm/worker/cache_engine.py` | `shared_mode` parameter in `_allocate_kv_cache()`: `"owner"` (allocate + export IPC), `"client"` (open IPC), `None` (legacy) |
| `vllm/worker/model_runner.py` | Standard vLLM model runner. SM partitioning is managed by the orchestrator, not model_runner. |
| `vllm/spec_decode/spec_decode_worker.py` | Creates `CospecManager` with SM controller, wires to workers. Handles two-queue partial outputs via `_cospec_skip`. Calls `orchestrator.remove_sequence()` on finished requests. |
| `vllm/engine/llm_engine.py` | Handles CoSpec empty outputs (no-op bootstrap steps) and partial outputs (`_cospec_skip` indices for draft-phase sequences). |
| `vllm/entrypoints/openai/api_server.py` | Engine selection: `COSPEC=1` → in-process AsyncLLMEngine, else V1/V0 branching. Calls `cleanup_cospec_resources()` on startup. |
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
| `VLLM_USE_V1` | `0` | Use V1 engine. This CoSpec fork defaults to V0 since V1 doesn't support speculative decoding. |

**Important notes**:
- Do NOT set `VLLM_ATTENTION_BACKEND`. Let vLLM auto-select the optimal backend (FLASH_ATTN). Never force XFORMERS — it is slower.
- **Set `VLLM_USE_V1=0` when using CoSpec or speculative decoding** — V1 engine doesn't support spec decode yet.
- **MPS is required**: CoSpec will fail immediately if NVIDIA MPS is not running. Start MPS with `bash cospec/scripts/start_mps.sh`.

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

### DONE — Code written and tested (26 tests pass: 14 unit + 12 E2E)
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
8. **`llm_engine.py`**: Handles empty CoSpec output (returns early when `outputs=[]` during bootstrap) and `_cospec_skip` via `_get_cospec_skip()` (reads from worker through executor chain, skips draft-phase sequences, remaps outputs with placeholders).
9. **`cospec_manager.py`**: `cleanup_cospec_resources()` removes stale `/dev/shm/cospec_*` and `/tmp/cospec_*`. Called automatically in `CospecManager.__init__()`. `SharedMemory.cleanup()` for explicit teardown.

### TODO — Organized by Priority

#### P1: Optimization — Performance improvements
1. **Cost model latency formulas**: `cost_model.py` `decide()` is hardcoded to always return `COLOCATED_SD`. The skeleton latency formulas (`_latency_ar`, `_latency_vanilla_sd`, `_latency_colocated_sd`) need coefficients from profiling to enable actual per-step mode selection.
2. **SM ratio tuning**: Hardcoded `default_sm_ratio=0.7`. Should be profiled per model pair.
3. **`cospec_manager.target_sm_ratio` feedback from orchestrator**: Currently hardcoded 1.0. The orchestrator controls SM partitions directly via `sm_controller` but doesn't update `cospec_manager.target_sm_ratio`, which `model_runner.py` reads. Only matters for the non-orchestrator code path.
4. **CUDA graph support**: E2E tests use `enforce_eager=True`. Need to verify CoSpec works with CUDA graphs enabled for max performance.

#### P2: Robustness — Edge cases and cleanup
5. ~~**Process cleanup on shutdown**~~: FIXED — Added `atexit` handler for reliable cleanup. `DraftWorkerRPC.shutdown()` now waits for acknowledgment.
6. **`SharedMemoryModelLoader._inprocess_state_dicts` cleanup**: Class-level dict caches state_dicts forever. Should be cleaned up when the engine is destroyed to free GPU memory for the shared weight references.
7. **Shared memory instance_id collision**: `shared_kv_cache.py` and `shared_logit_buffer.py` use `instance_id="default"`. Multiple CoSpec instances on same machine will collide. Should use PID or UUID.
8. **E2E test coverage**: Two E2E tests with identical models (`JackFram/llama-68m` for both target and draft): one without chunked prefill, one with. Need tests with: (a) different target/draft models, (b) larger batch sizes, (c) greedy vs sampling. Note: running both E2E tests in the same pytest process fails due to GPU memory not being freed between tests (pre-existing issue); run them in separate processes.
9. **UltraDict cleanup at exit**: `FileNotFoundError: '/cospec_shared_3'` during interpreter shutdown. Non-fatal but noisy. Need to unlink shared memory before `cleanup_cospec_resources()` deletes it.

### RECENTLY FIXED
7. **`sm_controller.py`**: Fixed segfault when `set_stream_mask` was called with the default CUDA stream (handle=0). Now falls back to `set_global_mask` for default stream. Added graceful `PermissionError` handling when MPS is not available (logs warning once, then silently skips). **Fixed ctypes restype bug**: `libsmctrl_set_global_mask` is a void function, but ctypes defaults to `c_int` return type. In thread pool workers (like AsyncLLMEngine uses), this caused garbage values to be interpreted as error code 1 (EPERM). Fixed by setting `restype = None` and `argtypes` for all libsmctrl functions.
8. **`spec_decode_worker.py`**: Added `__del__` cleanup to shut down the orchestrator and draft thread on garbage collection.
9. **`test_cost_model.py`**: Updated `test_ar_mode_has_zero_gamma` → `test_hardcoded_colocated_mode` to match the hardcoded always-colocated behavior.
10. **`test_sm_controller.py`**: Added `PermissionError` handling for tests that need MPS privileges. Added `test_set_partition_explicit_stream` for non-default stream testing.
11. **`loader.py` (SharedMemoryModelLoader)**: Fixed `CUDA error: invalid resource handle` when target and draft models are in the same process. Added in-process state_dict caching (`_inprocess_state_dicts` class variable) to avoid CUDA IPC for intra-process sharing. Draft model now loads from cached state_dict in ~6ms instead of failing.
12. **`orchestrator.py`**: Replaced `_prev_batch`/`_prev_proposals` pipeline with two-queue model (`_draft_queue`, `_verify_queue`, `_pending_pool`). Load-balanced entry for new decode sequences. `flush()` drains pipeline on shutdown. `remove_sequence()` for cleanup.
13. **All tests passing**: 26 tests pass (14 unit + 12 E2E). Run all tests: `docker exec -w /workspace/vllm -e VLLM_USE_V1=0 cospec-vllm python3 -m pytest tests/cospec/ -v`
14. **Chunked prefill support in orchestrator** (was P0): `_create_output()` now uses real `proposal_lens` from draft proposals instead of hardcoding gamma. Added `_sync_draft_prefills()` to sync draft KV cache for prefill sequences after target scoring (mirrors `spec_decode_worker.py:921-936`). Added `EXECUTE_PREFILL` RPC command in `worker_rpc.py`. Added E2E test with `enable_chunked_prefill=True`.
15. **IPC cleanup**: `cleanup_cospec_resources()` in `cospec_manager.py` removes stale `/dev/shm/cospec_*` and `/tmp/cospec_*`. Called from `api_server.py` on startup, `spec_decode_worker.__del__`, and test `init_cospec()`.
16. **Removed forced XFORMERS backend**: All CoSpec scripts and tests now use the default attention backend (FLASH_ATTN) for max performance.
17. **Fixed user-specific paths**: `run_docker.sh` uses `$HOME` instead of hardcoded user path for HF cache.
18. **Fixed `proposal_probs` None bug**: `_materialize_probs()` reads probs from `SharedLogitBuffer` before `_slice_proposal()`, preventing `None` probs in merged proposals.
19. **Fixed SamplerOutput attribute error**: `SamplerOutput` is a `msgspec.Struct` that doesn't allow arbitrary attributes. Moved `_cospec_seq_ids` to `orchestrator.last_output_seq_ids` and `_cospec_skip` to `SpecDecodeWorker._cospec_skip`. Engine reads skip via `_get_cospec_skip()` method.
20. **Fixed stale IPC handles on startup**: `CospecManager.__init__()` now calls `cleanup_cospec_resources()` to remove stale handles before model loading begins.
21. **Correctness verified**: 8/8 prompts match baseline output exactly (greedy decoding, `JackFram/llama-68m`, batch_size=8, max_tokens=32). Two-queue pipeline produces identical results to non-CoSpec baseline.
22. **Removed legacy code from model_runner.py**: Removed unused `cospec_manager` and `is_target` parameters from `execute_model()`. SM partitioning is now managed entirely by the orchestrator in CoSpec v2, not by model_runner.
23. **Restored V1/V0 engine branching in api_server.py**: Fixed accidental removal of V1 AsyncLLM and V0 in-process paths. Non-CoSpec users now get proper V1/V0 engine selection. CoSpec still uses in-process AsyncLLMEngine for MPS compatibility.
24. **`VLLM_USE_V1` default is `0`**: This CoSpec fork defaults to V0 engine since V1 doesn't support speculative decoding yet.
25. **Process cleanup with atexit handler**: Added `atexit` handler in `spec_decode_worker.py` for reliable cleanup at interpreter exit (more reliable than `__del__`). Refactored cleanup into shared `_cleanup_cospec()` method.
26. **RPC shutdown acknowledgment**: `DraftWorkerRPC.shutdown()` now waits for draft process to acknowledge SHUTDOWN command before closing connection, preventing race conditions.
27. **`max_spec_tokens` fix**: Removed hardcoded fallback of 5. Now reads from `speculative_config.num_speculative_tokens` and raises `ValueError` if not found. Note: `proposer_worker.max_proposal_len` is max SEQUENCE length, not speculative tokens - do not confuse them.
28. **E2E test JSON parsing**: Fixed `test_e2e.py` to extract JSON output from among vLLM log lines (vLLM logs to stdout, not stderr).
29. **Output remapping for CoSpec partial outputs**: Fixed `llm_engine.py` output remapping bug. The outputs are in `prefills + decodes` order but the remapping assumed original scheduler order. Now uses separate `prefill_idx` and `decode_idx` counters to correctly map outputs to sequence groups when prefills and decodes are interleaved in the scheduler batch. This fixes `AssertionError` in `multi_step.py:121` where `parent_seq_id` didn't match `seq_id`.

### Known Hardcoded Values
- `cost_model.py`: Always returns `COLOCATED_SD` mode, `target_sm_ratio=0.7`
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
| `tests/cospec/test_e2e.py` | E2E correctness tests (12 tests): greedy decoding, chunked prefill, different gamma values, batch sizes |
| `tests/cospec/test_cost_model.py` | Mode enum values |
| `tests/cospec/test_sm_controller.py` | Unit + GPU integration tests for SM controller |
| `tests/cospec/test_shared_kv_cache.py` | Tests for SharedKVCache allocation/cleanup |
| `tests/cospec/test_worker_rpc.py` | Tests for RPC pipe communication |

## Build & Run

- This is a vLLM fork; standard vLLM build (`pip install -e .`).
- Build libsmctrl: `cd cospec/csrc && mkdir -p build && cd build && cmake .. && make`
- **Start MPS before running**: `bash cospec/scripts/start_mps.sh`
- CoSpec server: set `COSPEC=1` and run the normal `vllm.entrypoints.openai.api_server`.
- Tests: `pytest tests/cospec/` (14 unit + 12 E2E tests, requires GPU for E2E).
- Run in Docker: `docker exec -w /workspace/vllm -e VLLM_USE_V1=0 cospec-vllm python3 -m pytest tests/cospec/ -v`
- **Do NOT set `VLLM_ATTENTION_BACKEND`** — let vLLM auto-select FLASH_ATTN for best performance.
- **MPS must be running** — CoSpec fails immediately without MPS. Tests skip automatically if MPS is not detected.

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
