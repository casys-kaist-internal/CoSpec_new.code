"""CoSpec v2 Orchestrator - Simplified.

Two-queue pipelining for concurrent draft + target execution.
Reuses original spec_decode_worker methods for verification and output.
"""

import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.profiler import record_function

import vllm.envs as envs
from vllm.cospec.sm_controller import SMController
from vllm.cospec.worker_rpc import DraftWorkerRPC
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
from vllm.spec_decode.interfaces import SpeculativeProposals

logger = init_logger(__name__)


class CoSpecOrchestrator:
    """Two-queue pipelining orchestrator for concurrent draft + target.

    Pipeline model:
      Step N:   Draft proposes batch A  ||  Target verifies batch B
      Step N+1: Draft proposes batch B  ||  Target verifies batch A

    New decode sequences are load-balanced between draft and pending_pool
    based on current queue sizes to keep D ≈ V each step.

    Reuses original spec_decode_worker methods for verification/output.
    """

    def __init__(
        self,
        spec_decode_worker: Any,
        draft_rpc: DraftWorkerRPC,
        sm_controller: SMController,
        target_sm_ratio: float = 0.7,
        shared_logit_buffer: Any = None,
    ):
        self.sdw = spec_decode_worker  # Access to original methods
        self.draft_rpc = draft_rpc
        self.sm_controller = sm_controller
        self.target_sm_ratio = target_sm_ratio
        self.shared_logit_buffer = shared_logit_buffer

        # SM partitioning: disabled by default (MPS natural sharing is better
        # for memory-bound decode). Set COSPEC_SM_PARTITION=1 to enable.
        self._use_sm_partition = bool(
            int(os.getenv("COSPEC_SM_PARTITION", "0")))

        # Two-queue state
        self._draft_queue: Dict[int, SequenceGroupMetadata] = {}
        # Verify state: batched proposals + seq_id -> row index mapping
        self._verify_proposals: Optional[SpeculativeProposals] = None
        self._verify_indices: Dict[int, int] = {}
        # Pending pool: new decode seqs deferred to next step for balancing
        self._pending_pool: Dict[int, SequenceGroupMetadata] = {}

        # Acceptance rate override: COSPEC_ACCEPT_RATE=0.7 forces ~70%.
        # Default -1.0 means use natural rate (no override).
        self._target_accept_rate = envs.COSPEC_ACCEPT_RATE
        if self._target_accept_rate >= 0:
            logger.info("CoSpec acceptance rate override: %.2f",
                        self._target_accept_rate)

        # Per-step logging: controlled by COSPEC_LOG=1 env var.
        # Disabled by default to avoid GPU->CPU sync from sampler .item().
        self._do_log = envs.COSPEC_LOG

        # Stats
        self._step_count = 0
        # Track sampler counters to compute per-step acceptance
        self._prev_accepted_tokens = 0
        self._prev_draft_tokens = 0

        # Output metadata for engine
        self.last_output_seq_ids: Optional[List[int]] = None
        self.last_output_num_prefills: int = 0

        # PyTorch profiler (COSPEC_PROFILE=1)
        # Must start BEFORE first set_global_mask call so CUPTI subscribes
        # first. libsmctrl gracefully degrades (SM partitioning becomes no-op).
        self._profiler: Optional[torch.profiler.profile] = None
        self._profile_active = False
        if envs.COSPEC_PROFILE:
            self._profile_skip = int(os.getenv("COSPEC_PROFILE_SKIP", "20"))
            self._profile_steps = int(os.getenv("COSPEC_PROFILE_STEPS", "100"))
            self._profile_output = os.getenv(
                "COSPEC_PROFILE_OUTPUT", "/workspace/cospec_trace.json")

            def _on_trace_ready(prof):
                prof.export_chrome_trace(self._profile_output)
                logger.info("CoSpec profiler: trace saved to %s",
                            self._profile_output)

            logger.info("CoSpec profiler enabled: skip=%d steps, "
                        "profile=%d steps, output=%s",
                        self._profile_skip, self._profile_steps,
                        self._profile_output)
            # Start profiler immediately so CUPTI registers before libsmctrl.
            # Schedule: warmup steps are profiled but discarded (CUPTI warmup),
            # then active steps are recorded and exported via on_trace_ready.
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=self._profile_skip,
                    active=self._profile_steps,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=False,
                on_trace_ready=_on_trace_ready,
            )
            self._profiler.__enter__()
            self._profile_active = True
            logger.info("CoSpec profiler: CUPTI registered "
                        "(SM partitioning disabled during profiling)")

    def _maybe_step_profiler(self) -> None:
        """Advance profiler schedule each orchestrator step."""
        if not self._profile_active:
            return
        self._profiler.step()
        # After all active steps, schedule triggers on_trace_ready then
        # enters 'wait' phase. Stop the profiler to release resources.
        total = self._profile_skip + self._profile_steps
        if self._step_count >= total:
            self._profiler.__exit__(None, None, None)
            self._profile_active = False
            self._profiler = None

    def step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        num_lookahead_slots: int,
    ) -> List[Any]:
        """Execute one pipelined step."""
        # Reset output metadata at start of each step to avoid stale values
        self.last_output_seq_ids = None
        self.last_output_num_prefills = 0

        if not seq_group_metadata_list:
            return []

        self._step_count += 1
        gamma = num_lookahead_slots or self.sdw.max_spec_tokens
        stream = torch.cuda.current_stream()

        with record_function("cospec::step"):
            # Promote pending_pool → draft_queue before splitting
            self._draft_queue.update(self._pending_pool)
            self._pending_pool.clear()

            # Split batch into: prefill, verify (have proposals), draft (need proposals)
            with record_function("cospec::split_batch"):
                prefill_seqs, verify_seqs, verify_row_indices, draft_seqs = (
                    self._split_batch(seq_group_metadata_list))


            # === Bootstrap: no verify_seqs yet ===
            if not verify_seqs:
                result = self._bootstrap_step(
                    prefill_seqs, draft_seqs, gamma, stream)
                self._maybe_step_profiler()
                return result

            # === Concurrent phase: draft || verify ===
            _do_timing = self._do_log
            if _do_timing:
                t_step_start = time.monotonic()

            # SM partitioning: controlled by COSPEC_SM_PARTITION env var.
            # Default off (MPS natural sharing better for memory-bound decode).
            if self._use_sm_partition:
                self.sm_controller.set_partition(stream, self.target_sm_ratio)
                self.draft_rpc.set_partition_async(
                    1.0 - self.target_sm_ratio)

            # Save bonus token seq_ids from the PREVIOUS verification step.
            # draft_seqs are sequences that were verified last step, so the
            # current _seq_with_bonus_token_in_last_step has their bonus info.
            # Must read before _run_verification() which overwrites the set.
            bonus_ids = set(self.sdw._seq_with_bonus_token_in_last_step)

            # Start async draft proposals (with bonus token info)
            if draft_seqs:
                with record_function("cospec::propose_async_send"):
                    self.draft_rpc.propose_async(
                        draft_seqs, gamma,
                        seq_ids_with_bonus_token=bonus_ids)

            # Run target scoring + verification (reuse original methods)
            # This updates _seq_with_bonus_token_in_last_step for the NEXT step.
            if _do_timing:
                t_target_start = time.monotonic()
            with record_function("cospec::run_verification"):
                output = self._run_verification(
                    prefill_seqs, verify_seqs, verify_row_indices, gamma)
            if _do_timing:
                t_target_end = time.monotonic()

            # Collect draft results
            new_proposals = None
            if draft_seqs:
                with record_function("cospec::propose_collect_recv"):
                    new_proposals = self._collect_proposals(
                        self.draft_rpc.propose_collect(),
                        batch_size=len(draft_seqs),
                        num_spec_tokens=gamma)

            # Sync draft KV cache for prefills (fire-and-forget: FIFO
            # ordering ensures draft processes this before next propose)
            if prefill_seqs:
                with record_function("cospec::execute_prefill_rpc"):
                    self.draft_rpc.execute_prefill_async(prefill_seqs)

            # Rotate queues
            with record_function("cospec::rotate_queues"):
                self._rotate_queues(verify_seqs, draft_seqs, new_proposals)

            if _do_timing:
                t_step_end = time.monotonic()
                self._log_step(
                    "CoSpec", len(prefill_seqs), len(draft_seqs),
                    len(verify_seqs),
                    t_target_ms=(t_target_end - t_target_start) * 1000,
                    t_total_ms=(t_step_end - t_step_start) * 1000)

            # Set output metadata
            self.last_output_seq_ids = [
                self._get_seq_id(s) for s in (prefill_seqs + verify_seqs)]
            self.last_output_num_prefills = len(prefill_seqs)

            self._maybe_step_profiler()
            return output

    def _split_batch(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[SequenceGroupMetadata], List[SequenceGroupMetadata],
               List[int], List[SequenceGroupMetadata]]:
        """Split scheduler batch into prefill/verify/draft groups.

        New decode sequences (not in any queue) are load-balanced:
        compare draft_queue size vs verify_queue + pending_pool size.
        If draft side is larger, defer to pending_pool (drafted next step).
        Otherwise, add to draft_seqs (drafted this step).

        Returns:
            prefill_seqs: Prompt sequences.
            verify_seqs: Sequences with proposals ready for verification.
            verify_row_indices: Row indices into _verify_proposals for each
                verify_seq (same order/length as verify_seqs).
            draft_seqs: Sequences that need new draft proposals.
        """
        prefill_seqs: List[SequenceGroupMetadata] = []
        verify_seqs: List[SequenceGroupMetadata] = []
        verify_row_indices: List[int] = []
        draft_seqs: List[SequenceGroupMetadata] = []
        new_decode_seqs: List[SequenceGroupMetadata] = []

        for sgm in seq_group_metadata_list:
            sid = self._get_seq_id(sgm)

            if sgm.is_prompt:
                prefill_seqs.append(sgm)
            elif sid in self._verify_indices:
                verify_row_indices.append(self._verify_indices.pop(sid))
                verify_seqs.append(sgm)
            elif sid in self._draft_queue:
                self._draft_queue.pop(sid)
                draft_seqs.append(sgm)
            else:
                # New decode sequence - collect for load balancing
                new_decode_seqs.append(sgm)

        # Load-balance new decode sequences between draft (now) and
        # pending (next step) to keep the pipeline balanced.
        #
        # After this step, the pipeline state becomes:
        #   verify_queue = draft_seqs (just drafted → need verification)
        #   draft_queue  = pending_pool (promoted next step) +
        #                  verify_seqs (rotated back after verification)
        #
        # For next-step balance: len(draft_seqs) ≈ len(pending) + len(verify_seqs)
        # This ensures both queues have sequences, enabling true concurrent
        # draft || verify execution instead of alternating phases.
        for sgm in new_decode_seqs:
            future_verify = len(draft_seqs)
            future_draft = len(self._pending_pool) + len(verify_seqs)
            if future_verify <= future_draft:
                draft_seqs.append(sgm)
            else:
                sid = self._get_seq_id(sgm)
                self._pending_pool[sid] = sgm

        return prefill_seqs, verify_seqs, verify_row_indices, draft_seqs

    def _bootstrap_step(
        self,
        prefill_seqs: List[SequenceGroupMetadata],
        draft_seqs: List[SequenceGroupMetadata],
        gamma: int,
        stream: torch.cuda.Stream,
    ) -> List[Any]:
        """Bootstrap: draft proposals, run prefills, no verification yet.

        When both draft_seqs and prefill_seqs exist, overlaps draft
        propose (async) with target prefill for better GPU utilization.
        """
        with record_function("cospec::bootstrap_step"):
            _do_timing = self._do_log
            if _do_timing:
                t_step_start = time.monotonic()

            self.sm_controller.set_full_gpu(stream)
            self.draft_rpc.set_full_gpu_async()

            t_draft_ms = 0.0
            t_prefill_ms = 0.0

            # When both draft and prefill exist, overlap them:
            # send async propose, run target prefill, collect propose
            if draft_seqs and prefill_seqs:
                if _do_timing:
                    t0 = time.monotonic()
                bonus_ids = set(self.sdw._seq_with_bonus_token_in_last_step)

                # Fire async propose to draft (runs on separate MPS ctx)
                with record_function("cospec::bootstrap_propose_async"):
                    self.draft_rpc.propose_async(
                        draft_seqs, gamma,
                        seq_ids_with_bonus_token=bonus_ids)

                # Run target prefill concurrently
                with record_function("cospec::bootstrap_prefill"):
                    execute_req = ExecuteModelRequest(
                        seq_group_metadata_list=prefill_seqs,
                        num_lookahead_slots=0,
                    )
                    output = self.sdw.scorer_worker.execute_model(
                        execute_req)
                if _do_timing:
                    t_prefill_ms = (time.monotonic() - t0) * 1000

                # Collect draft proposals
                with record_function("cospec::bootstrap_propose_collect"):
                    proposals_dict = self.draft_rpc.propose_collect()
                self._verify_proposals = self._collect_proposals(
                    proposals_dict,
                    batch_size=len(draft_seqs),
                    num_spec_tokens=gamma)
                self._verify_indices = {
                    self._get_seq_id(sgm): i
                    for i, sgm in enumerate(draft_seqs)
                }
                if _do_timing:
                    t_draft_ms = (time.monotonic() - t0) * 1000

                # Sync draft KV cache for prefills (fire-and-forget)
                self.draft_rpc.execute_prefill_async(prefill_seqs)

                # Restructure output for engine
                if (output and len(output) == 1
                        and len(output[0].outputs) > 1):
                    from vllm.model_executor.layers.sampler import (
                        SamplerOutput)
                    restructured = []
                    for seq_output in output[0].outputs:
                        restructured.append(
                            SamplerOutput(outputs=[seq_output]))
                    output = restructured

                self.last_output_seq_ids = [
                    self._get_seq_id(s) for s in prefill_seqs]
                self.last_output_num_prefills = len(prefill_seqs)

                if _do_timing:
                    t_total_ms = (time.monotonic() - t_step_start) * 1000
                    self._log_step("SD", len(prefill_seqs),
                                   len(draft_seqs), 0,
                                   t_draft_ms=t_draft_ms,
                                   t_prefill_ms=t_prefill_ms,
                                   t_total_ms=t_total_ms)
                return output

            # Draft-only: blocking propose
            if draft_seqs:
                if _do_timing:
                    t0 = time.monotonic()
                bonus_ids = set(self.sdw._seq_with_bonus_token_in_last_step)
                with record_function("cospec::bootstrap_propose"):
                    proposals_dict = self.draft_rpc.propose(
                        draft_seqs, gamma,
                        seq_ids_with_bonus_token=bonus_ids)
                self._verify_proposals = self._collect_proposals(
                    proposals_dict,
                    batch_size=len(draft_seqs),
                    num_spec_tokens=gamma)
                self._verify_indices = {
                    self._get_seq_id(sgm): i
                    for i, sgm in enumerate(draft_seqs)
                }
                if _do_timing:
                    t_draft_ms = (time.monotonic() - t0) * 1000

            # Prefill-only: run through target
            if prefill_seqs and not draft_seqs:
                if _do_timing:
                    t0 = time.monotonic()
                with record_function("cospec::bootstrap_prefill"):
                    execute_req = ExecuteModelRequest(
                        seq_group_metadata_list=prefill_seqs,
                        num_lookahead_slots=0,
                    )
                    output = self.sdw.scorer_worker.execute_model(
                        execute_req)
                    self.draft_rpc.execute_prefill_async(prefill_seqs)
                if _do_timing:
                    t_prefill_ms = (time.monotonic() - t0) * 1000

                if (output and len(output) == 1
                        and len(output[0].outputs) > 1):
                    from vllm.model_executor.layers.sampler import (
                        SamplerOutput)
                    restructured = []
                    for seq_output in output[0].outputs:
                        restructured.append(
                            SamplerOutput(outputs=[seq_output]))
                    output = restructured

                self.last_output_seq_ids = [
                    self._get_seq_id(s) for s in prefill_seqs]
                self.last_output_num_prefills = len(prefill_seqs)

                if _do_timing:
                    t_total_ms = (time.monotonic() - t_step_start) * 1000
                    self._log_step("AR", len(prefill_seqs), 0, 0,
                                   t_prefill_ms=t_prefill_ms,
                                   t_total_ms=t_total_ms)
                return output

            # Pure draft bootstrap or empty
            if _do_timing:
                t_total_ms = (time.monotonic() - t_step_start) * 1000
                self._log_step("SD", 0, len(draft_seqs), 0,
                               t_draft_ms=t_draft_ms,
                               t_total_ms=t_total_ms)

            self.last_output_seq_ids = None
            self.last_output_num_prefills = 0
            return []

    def _run_verification(
        self,
        prefill_seqs: List[SequenceGroupMetadata],
        verify_seqs: List[SequenceGroupMetadata],
        verify_row_indices: List[int],
        gamma: int,
    ) -> List[Any]:
        """Run target scoring and verification using original spec_decode_worker methods."""
        # Combine prefills + verify_seqs for target batch
        target_batch = prefill_seqs + verify_seqs

        # Build proposals: select rows from batched proposals, prepend prefill dummies
        with record_function("cospec::build_verify_proposals"):
            proposals = self._build_verify_proposals(
                verify_row_indices, len(prefill_seqs), gamma)

        # Build execute request
        execute_req = ExecuteModelRequest(
            seq_group_metadata_list=target_batch,
            num_lookahead_slots=gamma,
        )

        # Score proposals using original scorer
        with record_function("cospec::score_proposals"):
            proposal_scores = self.sdw.scorer.score_proposals(
                execute_req, proposals)

        # Verify using original method
        with record_function("cospec::verify_tokens"):
            accepted_token_ids, target_logprobs = self.sdw._verify_tokens(
                target_batch, proposal_scores, proposals, gamma)

        # Override acceptance rate if configured
        if self._target_accept_rate >= 0:
            accepted_token_ids = self._apply_accept_rate_override(
                accepted_token_ids, gamma)

        # Create output using original method
        with record_function("cospec::create_output"):
            proposal_lens = proposals.proposal_lens
            if isinstance(proposal_lens, torch.Tensor):
                proposal_lens_list = proposal_lens.tolist()
            else:
                proposal_lens_list = list(proposal_lens)

            return self.sdw._create_output_sampler_list(
                target_batch,
                accepted_token_ids,
                target_logprobs=target_logprobs,
                prompt_logprobs=(proposal_scores.prompt_logprobs
                                if not self.sdw._disable_logprobs
                                else None),
                k=gamma,
                stage_times=(0.0, 0.0, 0.0),
                proposal_lens_list=proposal_lens_list,
            )

    def _build_verify_proposals(
        self,
        row_indices: List[int],
        num_prefills: int,
        gamma: int,
    ) -> SpeculativeProposals:
        """Build verification proposals from batched storage using index_select.

        Selects the requested rows from _verify_proposals and prepends
        dummy entries for prefills (proposal_len=0).
        """
        device = self.sdw.device

        if not row_indices:
            # Only prefills
            return SpeculativeProposals(
                proposal_token_ids=torch.zeros(
                    (num_prefills, gamma), dtype=torch.long, device=device),
                proposal_probs=torch.zeros(
                    (num_prefills, gamma, self.sdw._vocab_size),
                    device=device),
                proposal_lens=torch.zeros(
                    num_prefills, dtype=torch.long, device=device),
                no_proposals=True,
            )

        # Select rows from batched proposals
        idx = torch.tensor(row_indices, dtype=torch.long, device=device)
        src = self._verify_proposals
        token_ids = torch.index_select(src.proposal_token_ids, 0, idx)
        lens = torch.index_select(src.proposal_lens, 0, idx)
        probs = (torch.index_select(src.proposal_probs, 0, idx)
                 if src.proposal_probs is not None else None)

        # Prepend dummy entries for prefills
        if num_prefills > 0:
            dummy_tokens = torch.zeros(
                (num_prefills, gamma), dtype=torch.long, device=device)
            dummy_lens = torch.zeros(
                num_prefills, dtype=torch.long, device=device)
            token_ids = torch.cat([dummy_tokens, token_ids], dim=0)
            lens = torch.cat([dummy_lens, lens], dim=0)
            if probs is not None:
                dummy_probs = torch.zeros(
                    (num_prefills, gamma, probs.shape[-1]), device=device)
                probs = torch.cat([dummy_probs, probs], dim=0)

        return SpeculativeProposals(
            proposal_token_ids=token_ids,
            proposal_probs=probs,
            proposal_lens=lens,
            no_proposals=False,
        )

    def _collect_proposals(
        self, proposals_dict: Dict[str, Any],
        batch_size: int = 0,
        num_spec_tokens: int = 0,
    ) -> SpeculativeProposals:
        """Convert RPC response dict to SpeculativeProposals.

        Args:
            proposals_dict: Response from draft worker RPC.
            batch_size: Known batch size for direct buffer read (avoids
                GPU→CPU sync from .item()). Falls back to metadata read
                if 0.
            num_spec_tokens: Known gamma for direct buffer read.
        """
        device = self.sdw.device

        # Read probs from shared buffer if available
        if (proposals_dict.get("probs_in_shared_buffer")
                and self.shared_logit_buffer):
            with record_function("cospec::collect_read_shared_buffer"):
                if batch_size > 0 and num_spec_tokens > 0:
                    proposal_probs = (
                        self.shared_logit_buffer.read_logits_direct(
                            batch_size, num_spec_tokens))
                else:
                    proposal_probs, _, _ = (
                        self.shared_logit_buffer.read_logits())
        else:
            proposal_probs = proposals_dict["proposal_probs"]
            if proposal_probs is not None:
                proposal_probs = proposal_probs.to(device)

        with record_function("cospec::collect_to_device"):
            proposal_token_ids = proposals_dict[
                "proposal_token_ids"].to(device)
            proposal_lens = proposals_dict["proposal_lens"]
            if isinstance(proposal_lens, torch.Tensor):
                proposal_lens = proposal_lens.to(device)
            else:
                proposal_lens = torch.tensor(
                    proposal_lens, device=device)

        return SpeculativeProposals(
            proposal_token_ids=proposal_token_ids,
            proposal_probs=proposal_probs,
            proposal_lens=proposal_lens,
            no_proposals=proposals_dict.get("no_proposals", False),
        )

    def _rotate_queues(
        self,
        verify_seqs: List[SequenceGroupMetadata],
        draft_seqs: List[SequenceGroupMetadata],
        new_proposals: Optional[SpeculativeProposals],
    ) -> None:
        """Rotate queues after step completion.

        Also rebalances if rotation would leave verify_queue empty while
        draft_queue is large: moves half of draft_queue to pending_pool
        so the next step can achieve concurrent execution after one
        bootstrap step instead of alternating indefinitely.
        """
        # Verified sequences -> draft queue (need new proposals)
        for sgm in verify_seqs:
            sid = self._get_seq_id(sgm)
            self._draft_queue[sid] = sgm

        # Drafted sequences -> verify queue (have proposals as batch)
        if draft_seqs and new_proposals is not None:
            self._verify_proposals = new_proposals
            self._verify_indices = {
                self._get_seq_id(sgm): i
                for i, sgm in enumerate(draft_seqs)
            }
        elif not draft_seqs and len(self._draft_queue) > 1:
            # No seqs were drafted → verify_queue will be empty next step.
            # Rebalance: move half of draft_queue to pending_pool so that
            # after bootstrap, both queues have sequences for concurrent
            # execution.
            items = list(self._draft_queue.items())
            half = len(items) // 2
            self._draft_queue = dict(items[:half])
            for sid, sgm in items[half:]:
                self._pending_pool[sid] = sgm

    def remove_sequence(self, seq_id: int) -> None:
        """Remove finished sequence from queues."""
        self._draft_queue.pop(seq_id, None)
        self._verify_indices.pop(seq_id, None)
        self._pending_pool.pop(seq_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get acceptance stats from the rejection sampler.

        The sampler accumulates correct counters because _verify_tokens()
        filters out non-spec sequences (prefills) via
        split_batch_by_proposal_len() before calling the sampler.
        This matches regular SD exactly.
        """
        sampler = self.sdw.spec_decode_sampler
        accepted = 0
        draft = 0
        if sampler is not None:
            acc_val = sampler.num_accepted_tokens
            draft_val = sampler.num_draft_tokens
            accepted = int(
                acc_val.item() if hasattr(acc_val, 'item') else acc_val)
            draft = int(
                draft_val if isinstance(draft_val, int) else draft_val)
        rate = accepted / draft if draft > 0 else 0.0
        return {
            "step_count": self._step_count,
            "acceptance_rate": rate,
            "accepted_tokens": accepted,
            "total_spec_tokens": draft,
        }

    def shutdown(self) -> None:
        """Shutdown orchestrator and draft worker."""
        logger.info("CoSpecOrchestrator shutting down. Stats: %s",
                     self.get_stats())
        try:
            self.draft_rpc.shutdown()
        except Exception:
            pass

    def _log_step(
        self,
        mode: str,
        n_prefill: int,
        n_draft: int,
        n_verify: int,
        t_draft_ms: float = 0.0,
        t_target_ms: float = 0.0,
        t_prefill_ms: float = 0.0,
        t_total_ms: float = 0.0,
    ) -> None:
        """Log a single summary line for this step."""
        if not self._do_log:
            return

        # Per-step acceptance from sampler delta
        sampler = self.sdw.spec_decode_sampler
        cur_acc = 0
        cur_draft = 0
        step_acc = 0
        step_draft = 0
        if sampler is not None:
            acc_val = sampler.num_accepted_tokens
            draft_val = sampler.num_draft_tokens
            cur_acc = int(
                acc_val.item() if hasattr(acc_val, 'item') else acc_val)
            cur_draft = int(
                draft_val if isinstance(draft_val, int) else draft_val)
            step_acc = cur_acc - self._prev_accepted_tokens
            step_draft = cur_draft - self._prev_draft_tokens
            self._prev_accepted_tokens = cur_acc
            self._prev_draft_tokens = cur_draft

        # Batch composition string
        n_pend = len(self._pending_pool)
        if n_pend > 0:
            batch_str = f"P={n_prefill} D={n_draft} V={n_verify} pend={n_pend}"
        else:
            batch_str = f"P={n_prefill} D={n_draft} V={n_verify}"

        # Build timing string (only include non-zero phases)
        timing_parts = []
        if t_draft_ms > 0:
            timing_parts.append(f"draft={t_draft_ms:.1f}ms")
        if t_target_ms > 0:
            timing_parts.append(f"target={t_target_ms:.1f}ms")
        if t_prefill_ms > 0:
            timing_parts.append(f"prefill={t_prefill_ms:.1f}ms")
        timing = " ".join(timing_parts)

        # Build acceptance string (only for steps that verify)
        if step_draft > 0:
            cum_rate = cur_acc / cur_draft * 100 if cur_draft > 0 else 0.0
            accept_str = (f"accept: {step_acc}/{step_draft}="
                          f"{step_acc / step_draft * 100:.0f}% "
                          f"(cum {cum_rate:.1f}%) | ")
        else:
            accept_str = ""

        logger.info(
            "[CoSpec step=%d] %-6s | %s | %s%s| total=%.1fms",
            self._step_count, mode, batch_str,
            accept_str, timing, t_total_ms,
        )

    def _apply_accept_rate_override(
        self,
        accepted_token_ids: torch.Tensor,
        gamma: int,
    ) -> torch.Tensor:
        """Override accepted_token_ids to force a target acceptance rate.

        For each sequence, at each draft position (1..gamma-1):
        - If currently accepted, randomly reject with prob (1 - rate)
        - Cascading: once rejected, all subsequent positions become -1
        - Position 0 is NEVER rejected (it always has a valid token from
          either draft acceptance or target resampling — setting it to -1
          would cause IndexError in _create_output_sampler_list)
        - Bonus token (position gamma) is set to -1 if rejection happened
          (bonus only valid when ALL draft tokens accepted)
        """
        batch_size = accepted_token_ids.shape[0]
        rate = self._target_accept_rate
        result = accepted_token_ids.clone()

        for b in range(batch_size):
            for k in range(gamma):
                if result[b, k].item() == -1:
                    break  # already rejected, cascade
                if torch.rand(1).item() > rate:
                    # Keep token at position k (it's a valid accepted token),
                    # but reject all subsequent positions
                    if k + 1 < gamma:
                        result[b, k + 1:gamma] = -1
                    # Bonus token (position gamma) is only valid if ALL
                    # draft tokens accepted; since we just rejected, clear it
                    result[b, gamma] = -1
                    break
        return result

    @staticmethod
    def _get_seq_id(sgm: SequenceGroupMetadata) -> int:
        """Get sequence ID from metadata."""
        return next(iter(sgm.seq_data.keys()))
