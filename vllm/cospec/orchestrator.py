"""CoSpec v2 Orchestrator - Simplified.

Two-queue pipelining for concurrent draft + target execution.
Reuses original spec_decode_worker methods for verification and output.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

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

        # Two-queue state
        self._draft_queue: Dict[int, SequenceGroupMetadata] = {}
        # Verify state: batched proposals + seq_id -> row index mapping
        self._verify_proposals: Optional[SpeculativeProposals] = None
        self._verify_indices: Dict[int, int] = {}
        # Pending pool: new decode seqs deferred to next step for balancing
        self._pending_pool: Dict[int, SequenceGroupMetadata] = {}

        # Stats
        self._step_count = 0
        # Track sampler counters to compute per-step acceptance
        self._prev_accepted_tokens = 0
        self._prev_draft_tokens = 0

        # Output metadata for engine
        self.last_output_seq_ids: Optional[List[int]] = None
        self.last_output_num_prefills: int = 0

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

        # Promote pending_pool → draft_queue before splitting
        self._draft_queue.update(self._pending_pool)
        self._pending_pool.clear()

        # Split batch into: prefill, verify (have proposals), draft (need proposals)
        prefill_seqs, verify_seqs, verify_row_indices, draft_seqs = (
            self._split_batch(seq_group_metadata_list))

        # === Bootstrap: no verify_seqs yet ===
        if not verify_seqs:
            return self._bootstrap_step(prefill_seqs, draft_seqs, gamma, stream)

        # === Concurrent phase: draft || verify ===
        t_step_start = time.monotonic()

        self.sm_controller.set_partition(stream, self.target_sm_ratio)
        self.draft_rpc.set_partition(1.0 - self.target_sm_ratio)

        # Save bonus token seq_ids from the PREVIOUS verification step.
        # draft_seqs are sequences that were verified last step, so the
        # current _seq_with_bonus_token_in_last_step has their bonus info.
        # Must read before _run_verification() which overwrites the set.
        bonus_ids = set(self.sdw._seq_with_bonus_token_in_last_step)

        # Start async draft proposals (with bonus token info)
        if draft_seqs:
            self.draft_rpc.propose_async(
                draft_seqs, gamma,
                seq_ids_with_bonus_token=bonus_ids)

        # Run target scoring + verification (reuse original methods)
        # This updates _seq_with_bonus_token_in_last_step for the NEXT step.
        t_target_start = time.monotonic()
        output = self._run_verification(
            prefill_seqs, verify_seqs, verify_row_indices, gamma)
        t_target_end = time.monotonic()

        # Collect draft results
        new_proposals = None
        if draft_seqs:
            new_proposals = self._collect_proposals(
                self.draft_rpc.propose_collect())

        # Sync draft KV cache for prefills
        if prefill_seqs:
            self.draft_rpc.execute_prefill(prefill_seqs)

        # Rotate queues
        self._rotate_queues(verify_seqs, draft_seqs, new_proposals)

        t_step_end = time.monotonic()

        n_prefill = len(prefill_seqs)
        n_verify = len(verify_seqs)
        n_draft = len(draft_seqs)

        t_target_ms = (t_target_end - t_target_start) * 1000
        t_total_ms = (t_step_end - t_step_start) * 1000

        self._log_step("CoSpec", n_prefill, n_draft, n_verify,
                       t_target_ms=t_target_ms, t_total_ms=t_total_ms)

        # Set output metadata
        self.last_output_seq_ids = [
            self._get_seq_id(s) for s in (prefill_seqs + verify_seqs)]
        self.last_output_num_prefills = n_prefill

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
        # pending (next step) to keep D ≈ V.
        # draft side = draft_seqs (will be drafted this step)
        # verify side = verify_seqs (will be verified this step) +
        #               pending_pool (will be drafted next step)
        for sgm in new_decode_seqs:
            draft_side = len(draft_seqs) + len(self._pending_pool)
            verify_side = len(verify_seqs)
            if draft_side <= verify_side:
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
        """Bootstrap: draft proposals, run prefills, no verification yet."""
        t_step_start = time.monotonic()

        self.sm_controller.set_full_gpu(stream)
        self.draft_rpc.set_full_gpu()

        t_draft_ms = 0.0
        t_prefill_ms = 0.0

        # Draft proposals for draft_seqs
        if draft_seqs:
            # Pass bonus token info from previous verification (if any).
            # In bootstrap, previously-verified sequences still need bonus
            # token handling for correct draft KV cache state.
            t0 = time.monotonic()
            bonus_ids = set(self.sdw._seq_with_bonus_token_in_last_step)
            proposals_dict = self.draft_rpc.propose(
                draft_seqs, gamma,
                seq_ids_with_bonus_token=bonus_ids)
            # Store as batch — no slicing needed
            self._verify_proposals = self._collect_proposals(proposals_dict)
            self._verify_indices = {
                self._get_seq_id(sgm): i
                for i, sgm in enumerate(draft_seqs)
            }
            t_draft_ms = (time.monotonic() - t0) * 1000

        # Run prefills through target
        if prefill_seqs:
            t0 = time.monotonic()
            execute_req = ExecuteModelRequest(
                seq_group_metadata_list=prefill_seqs,
                num_lookahead_slots=0,
            )
            output = self.sdw.scorer_worker.execute_model(execute_req)
            # Sync draft KV cache
            self.draft_rpc.execute_prefill(prefill_seqs)
            t_prefill_ms = (time.monotonic() - t0) * 1000

            # Restructure output: scorer_worker returns [SamplerOutput(outputs=[all_prefills])]
            # but llm_engine expects [SamplerOutput(outputs=[p0]), SamplerOutput(outputs=[p1]), ...]
            # (one SamplerOutput per prefill for seq_id-based remapping)
            if output and len(output) == 1 and len(output[0].outputs) > 1:
                from vllm.model_executor.layers.sampler import SamplerOutput
                restructured = []
                for seq_output in output[0].outputs:
                    restructured.append(SamplerOutput(outputs=[seq_output]))
                output = restructured

            self.last_output_seq_ids = [
                self._get_seq_id(s) for s in prefill_seqs]
            self.last_output_num_prefills = len(prefill_seqs)

            # AR mode: prefill only, no drafting
            if not draft_seqs:
                t_total_ms = (time.monotonic() - t_step_start) * 1000
                self._log_step("AR", len(prefill_seqs), 0, 0,
                               t_prefill_ms=t_prefill_ms,
                               t_total_ms=t_total_ms)
            else:
                t_total_ms = (time.monotonic() - t_step_start) * 1000
                self._log_step("SD", len(prefill_seqs), len(draft_seqs), 0,
                               t_draft_ms=t_draft_ms,
                               t_prefill_ms=t_prefill_ms,
                               t_total_ms=t_total_ms)
            return output

        # Pure SD bootstrap - draft only, no output
        t_total_ms = (time.monotonic() - t_step_start) * 1000
        self._log_step("SD", 0, len(draft_seqs), 0,
                       t_draft_ms=t_draft_ms, t_total_ms=t_total_ms)

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
        proposals = self._build_verify_proposals(
            verify_row_indices, len(prefill_seqs), gamma)

        # Build execute request
        execute_req = ExecuteModelRequest(
            seq_group_metadata_list=target_batch,
            num_lookahead_slots=gamma,
        )

        # Score proposals using original scorer
        proposal_scores = self.sdw.scorer.score_proposals(
            execute_req, proposals)

        # Verify using original method
        accepted_token_ids, target_logprobs = self.sdw._verify_tokens(
            target_batch, proposal_scores, proposals, gamma)

        # Create output using original method
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
                            if not self.sdw._disable_logprobs else None),
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
    ) -> SpeculativeProposals:
        """Convert RPC response dict to SpeculativeProposals."""
        device = self.sdw.device

        # Read probs from shared buffer if available
        if (proposals_dict.get("probs_in_shared_buffer")
                and self.shared_logit_buffer):
            proposal_probs, _, _ = self.shared_logit_buffer.read_logits()
        else:
            proposal_probs = proposals_dict["proposal_probs"]
            if proposal_probs is not None:
                proposal_probs = proposal_probs.to(device)

        proposal_token_ids = proposals_dict["proposal_token_ids"].to(device)
        proposal_lens = proposals_dict["proposal_lens"]
        if isinstance(proposal_lens, torch.Tensor):
            proposal_lens = proposal_lens.to(device)
        else:
            proposal_lens = torch.tensor(proposal_lens, device=device)

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
        """Rotate queues after step completion."""
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
        """Log a single summary line for this step.

        Fields:
          mode     - AR / SD / CoSpec
          P/D/V    - prefill / draft / verify batch sizes this step
          pend     - sequences deferred to next step (load balancing)
          accept   - tokens accepted/proposed this step (cumulative %)
          timing   - per-phase wall-clock breakdown
                     draft  = gamma draft model iterations (async on draft process)
                     target = target forward pass + rejection sampling + output
                     prefill = target prefill + draft KV sync
          total    - end-to-end step time

        Example output:
          [CoSpec step=42] CoSpec | P=1 D=4 V=4 pend=2 | accept: 18/20=90% (cum 91.2%) | draft=2.1ms target=18.7ms | 34.3ms
          [CoSpec step=43] SD    | P=0 D=6 V=0 | draft=15.1ms | 15.1ms
          [CoSpec step=44] AR    | P=2 D=0 V=0 | prefill=8.3ms | 8.3ms
        """
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

    @staticmethod
    def _get_seq_id(sgm: SequenceGroupMetadata) -> int:
        """Get sequence ID from metadata."""
        return next(iter(sgm.seq_data.keys()))
