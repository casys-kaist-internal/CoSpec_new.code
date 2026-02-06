# CoSpec v2 — Colocated Speculative Decoding for vLLM

## What Is CoSpec

CoSpec runs a **target process** and a **draft process** on the same GPU via MPS (Multi-Process Service). The target process owns the scheduler and orchestrator; the draft process is a lightweight RPC worker. SM (Streaming Multiprocessor) partitioning via `libsmctrl` enables concurrent execution with zero-cost mode switching between three modes: **AR**, **Vanilla SD**, and **Colocated SD**.

This is a fork of vLLM. The base commit is `bbde4701f825fd9f2607a8d5e7e87bc0f0b014c4`.

## Architecture Overview

```
API Entrypoint (thin, single queue)
        |
  Target Process (owns scheduler + orchestrator)
    ├── Target model weights
    ├── Target KV cache
    ├── Shared logit buffer (allocated here, exported via CUDA IPC)
    ├── CostModel: decide(B, α, S) → (mode, γ, r)
    ├── SMController (libsmctrl)
    └── Orchestrator (replaces SpecDecodeWorker + serving_completion_cospec)
        |
  Draft Process (RPC worker, receives commands)
    ├── Draft model weights
    ├── Draft KV cache (separate from target)
    ├── Shared logit buffer (opened via CUDA IPC)
    └── SMController (libsmctrl)
```

Both processes on same GPU via MPS.
Note: Target and draft have SEPARATE KV caches (different model architectures).

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
| `orchestrator.py` | `CoSpecOrchestrator`: Two-queue colocated SD pipelining with load balancing, per-step logging, and timing. `Mode` enum. Always uses colocated mode with SM ratio 0.7. |
| `draft_process.py` | Draft process lifecycle: `init_draft_process()`, `cleanup_cospec()`, `assert_mps_running()`, `_draft_process_entry()`. Spawns draft as `mp.Process` with its own CUDA context for MPS concurrency. |
| `worker_rpc.py` | `DraftWorkerRPC` (client) and `DraftWorkerServer` (draft process). Commands over pipe, large data via shared GPU memory. |
| `shared_memory.py` | `SharedLogitBuffer`: CUDA IPC for sharing draft logits between target and draft processes. |
| `metadata.py` | `CoSpecOutputMetadata`: Dataclass bundling per-step output metadata (seq_ids, num_prefills, skip_indices). |

### Modified vLLM Files

| File | CoSpec Changes |
|------|----------------|
| `vllm/envs.py` | Defines `COSPEC` environment variable (master switch). `VLLM_USE_V1` defaults to `1` (non-CoSpec users get V1). |
| `vllm/worker/model_runner.py` | Standard vLLM model runner. SM partitioning is managed by the orchestrator, not model_runner. |
| `vllm/spec_decode/spec_decode_worker.py` | Creates `CospecManager` with SM controller, wires to workers. Stores per-step `CoSpecOutputMetadata` for engine. Calls `orchestrator.remove_sequence()` on finished requests. |
| `vllm/engine/llm_engine.py` | Handles CoSpec empty outputs (no-op bootstrap steps) and partial outputs via `CoSpecOutputMetadata` for seq_id-based remapping. |
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
| `COSPEC_LOG` | `0` | Per-step orchestrator logging (mode, batch composition, acceptance rate, timing). Set `COSPEC_LOG=1` to enable. |
| `COSPEC_PROFILE` | `0` | Enable PyTorch profiler. Exports Chrome trace to `COSPEC_PROFILE_OUTPUT`. SM partitioning is disabled during profiling (CUPTI/libsmctrl conflict). |
| `COSPEC_PROFILE_SKIP` | `20` | Number of warmup steps to skip before profiling. |
| `COSPEC_PROFILE_STEPS` | `100` | Number of steps to profile after warmup. |
| `COSPEC_PROFILE_OUTPUT` | `/workspace/cospec_trace.json` | Output path for Chrome trace JSON. Open in `chrome://tracing` or Perfetto UI. |
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

### DONE — Code written and tested (10 unit + 14 E2E tests = 24 total)
1. **`shared_logit_buffer.py`**: Fixed double `_share_cuda_()` bug (line 68/72).
2. **`cost_model.py`**: `decide()` hardcoded to always return `COLOCATED_SD` mode.
3. **`worker_rpc.py`**: Added `propose_async()` / `propose_collect()` to `DraftWorkerRPC`. Wired `DraftWorkerServer._handle_propose()` to call `self._worker.get_spec_proposals()` and return CPU tensors over pipe.
4. **`orchestrator.py`**: Two-queue colocated SD with load-balanced entry:
   - `_draft_queue`, `_verify_queue`, `_pending_pool` replace `_prev_batch`/`_prev_proposals`
   - `_slice_proposal()`: extract single-sequence from batched proposals
   - `_merge_proposals()`: combine individual proposals back into batch
   - `_run_prefills_only()`: handle prefills when verify_queue is empty
   - `flush()`: drain verify_queue on shutdown
   - `remove_sequence(seq_id)`: clean up finished/preempted sequences from all queues (draft, verify, pending)
   - `_step_colocated_sd()`: full two-queue concurrent pipeline
   - All wired methods: `_score_proposals()`, `_verify()`, `_create_output()`, `_sync_draft_prefills()`
   - **Per-step logging** (`_log_step()`): logs mode (AR/SD/CoSpec), batch composition (P/D/V + pend=N), per-step acceptance rate (delta from cumulative sampler counters), and timing (draft/target/prefill/total in ms). In CoSpec concurrent mode, draft time is not measured (overlaps with target); gap between target and total shows if draft was bottleneck.
   - **Load balancing**: New decode sequences are split between `draft_seqs` (drafted now) and `_pending_pool` (deferred to next step) by comparing `len(draft_seqs) + len(pending_pool)` vs `len(verify_seqs)`. Pending pool is promoted to draft_queue at start of each step.
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
4. ~~**CUDA graph support**~~: CoSpec works with CUDA graphs. Do NOT use `enforce_eager` — CUDA graphs are enabled by default and provide better performance.

#### P2: Robustness — Edge cases and cleanup
5. ~~**Process cleanup on shutdown**~~: FIXED — Added `atexit` handler for reliable cleanup. `DraftWorkerRPC.shutdown()` now waits for acknowledgment.
6. **`SharedMemoryModelLoader._inprocess_state_dicts` cleanup**: Class-level dict caches state_dicts forever. Should be cleaned up when the engine is destroyed to free GPU memory for the shared weight references.
7. **Shared memory instance_id collision**: `shared_kv_cache.py` and `shared_logit_buffer.py` use `instance_id="default"`. Multiple CoSpec instances on same machine will collide. Should use PID or UUID.
8. **E2E test coverage**: ~~Need tests with different target/draft models~~ DONE — added `test_acceptance_rate_different_models` with Qwen3-8B/Qwen3-0.6B. Still could add: (a) greedy vs sampling tests, (b) longer sequence tests. Note: running E2E tests in subprocesses (each test spawns its own vLLM) to avoid GPU memory leaks between tests.
9. **UltraDict cleanup at exit**: `FileNotFoundError: '/cospec_shared_3'` during interpreter shutdown. Non-fatal but noisy. Need to unlink shared memory before `cleanup_cospec_resources()` deletes it.

### RECENTLY FIXED
40. **Fixed acceptance rate metrics discrepancy** (was P0): Two issues fixed:
   - **Buggy counters**: Orchestrator's `_update_stats()` counted ALL rows of `accepted_token_ids` including prefill dummy rows. Fix: removed `_update_stats()` and orchestrator counters; `get_stats()` now reads directly from `self.sdw.spec_decode_sampler`.
   - **Draft KV cache desync from bonus tokens**: After verification, the target gives a "bonus token" to accepted sequences. The draft model's KV cache was missing this token's entry, causing wrong proposals and ~13% rejection rate (87.44% acceptance vs expected 100% with same model). Fix: pass `seq_ids_with_bonus_token` from orchestrator to draft worker via RPC in both the concurrent phase (`propose_async`) and bootstrap phase (`propose`). The draft's `multi_step_worker._expand_execute_model_request()` then handles the bonus token expansion correctly (same mechanism as regular SD). Also syncs draft KV cache for prefills via `execute_prefill` RPC in `spec_decode_worker.py`. Test now shows 0% difference between CoSpec and regular SD (440/440 = 100% for both).
7. **`sm_controller.py`**: Fixed segfault when `set_stream_mask` was called with the default CUDA stream (handle=0). Now falls back to `set_global_mask` for default stream. Added graceful `PermissionError` handling when MPS is not available (logs warning once, then silently skips). **Fixed ctypes restype bug**: `libsmctrl_set_global_mask` is a void function, but ctypes defaults to `c_int` return type. In thread pool workers (like AsyncLLMEngine uses), this caused garbage values to be interpreted as error code 1 (EPERM). Fixed by setting `restype = None` and `argtypes` for all libsmctrl functions.
8. **`spec_decode_worker.py`**: Added `__del__` cleanup to shut down the orchestrator and draft thread on garbage collection.
9. **`test_cost_model.py`**: Updated `test_ar_mode_has_zero_gamma` → `test_hardcoded_colocated_mode` to match the hardcoded always-colocated behavior.
10. **`test_sm_controller.py`**: Added `PermissionError` handling for tests that need MPS privileges. Added `test_set_partition_explicit_stream` for non-default stream testing.
11. **`loader.py` (SharedMemoryModelLoader)**: Fixed `CUDA error: invalid resource handle` when target and draft models are in the same process. Added in-process state_dict caching (`_inprocess_state_dicts` class variable) to avoid CUDA IPC for intra-process sharing. Draft model now loads from cached state_dict in ~6ms instead of failing.
12. **`orchestrator.py`**: Replaced `_prev_batch`/`_prev_proposals` pipeline with two-queue model (`_draft_queue`, `_verify_queue`, `_pending_pool`). Load-balanced entry for new decode sequences. `flush()` drains pipeline on shutdown. `remove_sequence()` for cleanup.
13. **All tests passing**: 24 tests pass (10 unit + 14 E2E). Run all tests: `docker exec -w /workspace -e VLLM_USE_V1=0 -e CUDA_VISIBLE_DEVICES=0 cospec-vllm python3 -m pytest tests/cospec/ -v`
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
30. **Disabled async output processing for CoSpec**: In async mode, `_advance_to_next_step()` uses `zip()` which stops early with partial outputs, causing prefills to not get their `num_computed_tokens` updated (leading to scheduler assertion errors). Fixed in `scheduler.py` by disabling async output processing when `COSPEC=1`.
31. **Fixed prefill proposal mismatch**: When prefills are in the target batch for scoring/verification, `_verify_tokens` expected proposals to match the batch size. But `merged_proposals` only had entries for `verify_seqs`. Added `_prepend_prefill_proposals()` in `orchestrator.py` to add dummy proposal entries (with `proposal_len=0`) for prefills.
32. **Removed incorrect KV cache sharing**: Target and draft models have SEPARATE KV caches (different model architectures, different layer counts, different head sizes). Removed `cospec_shared_mode = "owner"` from target worker and `cospec_shared_mode = "client"` from draft process. Each model now allocates its own KV cache normally. Draft KV cache is populated via `execute_prefill` RPC calls. This was the root cause of near-zero acceptance rate (0.05%) — draft was reading garbage data from wrong-shaped KV tensors.
33. **Fixed SharedLogitBuffer.read_logits() call signature**: The orchestrator was passing batch_size and num_tokens as arguments, but `read_logits()` takes no arguments (reads metadata from buffer). Fixed to call `read_logits()` with no args and unpack returned tuple.
34. ~~**Fixed _strip_sgm_for_draft token separation**~~: REMOVED - `_strip_sgm_for_draft` was deleted as dead code. SequenceGroupMetadata is now passed directly to draft worker without stripping.
35. **Fixed bootstrap output restructuring**: `scorer_worker.execute_model()` returns `[SamplerOutput(outputs=[all_prefills])]` (one SamplerOutput with all outputs), but llm_engine remapping expects one SamplerOutput per prefill. Added restructuring in `_bootstrap_step` to split into per-prefill SamplerOutputs.
36. **Added defensive output remapping**: The two-queue pipeline can have seq_id mismatches between `last_output_seq_ids` and actual outputs. Added defensive code in `llm_engine.py` to: (a) detect count mismatches, (b) extract seq_ids from actual output parent_seq_ids, (c) log warnings instead of crashing. This improves stability though root cause (queue state sync) is not fully fixed.
37. **Reset orchestrator state each step**: Added `last_output_seq_ids = None` and `last_output_num_prefills = 0` at start of each `step()` to prevent stale values from previous steps causing remapping errors.
38. **Force multi-output path for CoSpec**: Changed `has_multiple_outputs` condition to include CoSpec even with single output, ensuring seq_id-based remapping is always used when `cospec_seq_ids` is set.
39. **Code cleanup**: Removed dead code: `SharedKVCache` class (target/draft have separate KV caches), `_strip_sgm_for_draft` function (unnecessary stripping), `shared_mode` parameter from `CacheEngine`. Added `CoSpecOutputMetadata` dataclass to bundle per-step metadata atomically. Simplified llm_engine.py output remapping (~190 → ~90 lines).
40. **Per-step logging in orchestrator**: Added `_log_step()` method with per-step summary line showing: mode (AR/SD/CoSpec), batch composition (P=prefills D=draft V=verify, optional pend=N), per-step acceptance rate as delta from cumulative rejection sampler counters, and timing in ms. In CoSpec concurrent mode, draft time is unmeasurable (overlaps with target), so only target and total times are shown — the gap indicates if draft was the bottleneck.
41. **Load balancing for two-queue pipeline**: Implemented `_pending_pool` for load-balanced entry of new decode sequences. New sequences are split between `draft_seqs` (drafted this step) and `_pending_pool` (deferred to next step) by comparing queue sizes. Pending pool promoted to draft_queue at start of each step. Dramatically improved queue balance during burst arrivals (D=38 V=35 vs old D=13 V=1).
42. **Acceptance rate tests**: Added `TestCoSpecAcceptanceRate` with two tests: (a) `test_acceptance_rate_same_model` — same model for target/draft, verifies 100% acceptance and exact output match; (b) `test_acceptance_rate_different_models` — Qwen3-8B target + Qwen3-0.6B draft, verifies acceptance rates within 5% tolerance. Note: with different models, outputs may diverge slightly between regular SD and CoSpec due to floating-point non-determinism in batched GPU operations (this affects regular SD vs AR too, not CoSpec-specific).
43. **Draft process refactored to `draft_process.py`**: Extracted draft process management from `spec_decode_worker.py` into `vllm/cospec/draft_process.py`. Functions: `init_draft_process()`, `cleanup_cospec()`, `assert_mps_running()`, `_draft_process_entry()`. Cleaner separation of concerns.
44. **GPU memory utilization**: Lowered `server.sh` default from 0.85 to 0.80 to avoid draft worker OOM with large batches (40+ seqs). Added `PYTHONUNBUFFERED=1` for immediate log output.
45. **PyTorch profiler integration**: Added `COSPEC_PROFILE=1` env var to enable `torch.profiler` with CPU+CUDA activities. Profiler starts in `orchestrator.__init__()` (before first `set_global_mask` call) so CUPTI registers its callback subscriber first. Uses `torch.profiler.schedule()` with configurable warmup/active steps. Exports Chrome trace JSON via `on_trace_ready` callback. Modified `libsmctrl_core.c` to gracefully handle CUPTI conflict: `setup_sm_control_callback()` now warns and returns instead of aborting when `cuSubscribeLaunchCallback` fails (error 999). `set_global_mask`/`set_next_mask` become no-ops when callback setup failed (checked via `sm_control_setup_ok` flag). SM partitioning is disabled during profiling, but two-queue pipeline is fully captured.

### Known Hardcoded Values
- `cost_model.py`: Always returns `COLOCATED_SD` mode, `target_sm_ratio=0.7`
- `cost_model.py`: EMA coefficients `alpha=0.8`, `ema_weight=0.3`, `batch_ema_weight=0.5` (untuned)
- `cost_model.py`: Latency formula coefficients are placeholders (never profiled)
- `shared_logit_buffer.py`: `instance_id="default"` (collision risk with multiple instances)
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
| `tests/cospec/test_e2e.py` | E2E correctness tests (14 tests): greedy decoding, chunked prefill, different gamma values, batch sizes, acceptance rate with same model (100% match) and different models (Qwen3-8B/Qwen3-0.6B, within 5% tolerance) |
| `tests/cospec/test_sm_controller.py` | Unit + GPU integration tests for SM controller (7 tests) |
| `tests/cospec/test_worker_rpc.py` | Tests for RPC pipe communication (3 tests) |
| `tests/cospec/test_acceptance_rate.py` | Standalone acceptance rate comparison script (not run by pytest, use `python3 tests/cospec/test_acceptance_rate.py`) |

## Build & Run

- This is a vLLM fork; standard vLLM build (`pip install -e .`).
- Build libsmctrl: `cd cospec/csrc && mkdir -p build && cd build && cmake .. && make`
- **Start MPS before running**: `bash cospec/scripts/start_mps.sh`
- **Do NOT set `VLLM_ATTENTION_BACKEND`** — let vLLM auto-select FLASH_ATTN for best performance.
- **MPS must be running** — CoSpec fails immediately without MPS. Tests skip automatically if MPS is not detected.
- All scripts are written for the container environment — no `docker exec` wrappers inside scripts.

### Running in Docker

**Start server** (terminal 1):
```bash
docker exec -it -w /workspace cospec-vllm bash cospec/scripts/server.sh
```

**Run benchmark client** (terminal 2):
```bash
docker exec -it -w /workspace cospec-vllm bash cospec/scripts/client.sh
```

**Run tests**:
```bash
docker exec -w /workspace -e VLLM_USE_V1=0 -e CUDA_VISIBLE_DEVICES=0 cospec-vllm python3 -m pytest tests/cospec/ -v --timeout=300
```

## Profiling

### PyTorch Profiler (Recommended)

Set `COSPEC_PROFILE=1` to enable. The profiler captures CPU ops and CUDA kernels from the **target process only** (draft process runs in a separate MPS context, invisible to this profiler).

```bash
docker exec -it -w /workspace -e COSPEC_PROFILE=1 -e COSPEC_PROFILE_SKIP=10 -e COSPEC_PROFILE_STEPS=50 cospec-vllm bash cospec/scripts/server.sh
# In another terminal: bash cospec/scripts/client.sh
# Trace auto-saved to /workspace/cospec_trace.json
```

Open the trace in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev/).

**How it works**: The profiler starts in `orchestrator.__init__()` (before the first `set_global_mask` call) so CUPTI registers its callback subscriber first. When libsmctrl later tries to register its own callback, it gets error 999 and gracefully degrades — SM partitioning becomes a no-op during profiling. The two-queue pipeline structure is still fully profiled.

**CUPTI / libsmctrl conflict**: Both CUPTI (used by torch.profiler and nsys) and libsmctrl use `cuSubscribeLaunchCallback` — only one subscriber is allowed. Whichever registers first wins. The C code in `libsmctrl_core.c` has graceful degradation: if callback subscription fails, `set_global_mask` becomes a no-op instead of aborting.

**Limitations**:
- Only the target process is profiled (draft process has its own MPS CUDA context)
- SM partitioning is disabled during profiling (acceptable tradeoff)
- nsys profiling also conflicts with libsmctrl for the same reason

### Profiling Results (Qwen3-8B + Qwen3-0.6B, RTX A6000)

- **GEMM dominates**: 90.8% of GPU time is matrix multiplications (decode-phase, weight-bound)
- **Attention is cheap**: 0.35% of GPU time (short context decode, flash attention split-KV)
- **Target GPU utilization**: 32.9% overall, 55-75% during verify steps
- **Two-queue pattern visible**: odd steps have ~1,046 kernels (verify), even steps have 0 kernels (target idles while draft proposes)
- **Main bottlenecks**: `cudaStreamSynchronize` (78% of CUDA runtime = draft/target barrier), ~35ms idle per draft-only step, `aten::index_put_` spikes to 354ms (KV cache sync stalls)

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
