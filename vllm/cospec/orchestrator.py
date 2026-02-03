"""CoSpec v2 Orchestrator.

Lives in the target process. Coordinates per-step execution using SM
partitioning for concurrent draft + target execution on the same GPU.
"""

import enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.cospec.sm_controller import SMController
from vllm.cospec.worker_rpc import DraftWorkerRPC
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores
from vllm.spec_decode.util import split_batch_by_proposal_len

logger = init_logger(__name__)


class Mode(enum.Enum):
    """Execution modes for CoSpec."""
    AR = "ar"
    VANILLA_SD = "vanilla_sd"
    COLOCATED_SD = "colocated_sd"


class CoSpecOrchestrator:
    """Central orchestrator for CoSpec v2.

    Two-queue colocated SD: sequences alternate between draft_queue (need
    proposals) and verify_queue (have proposals). Each step runs drafting
    and verification concurrently on partitioned SMs.

    Args:
        target_worker: The target model worker.
        draft_rpc: RPC client to draft worker process.
        sm_controller: SM partition controller.
        max_spec_tokens: Maximum speculative tokens (γ).
        target_sm_ratio: Fraction of SMs for target model (draft gets 1.0 - target_sm_ratio).
    """

    def __init__(
        self,
        target_worker: Any,
        draft_rpc: DraftWorkerRPC,
        sm_controller: SMController,
        spec_decode_worker: Any = None,
        max_spec_tokens: int = 7,
        target_sm_ratio: float = 0.7,
        shared_logit_buffer: Any = None,
    ):
        self.target_worker = target_worker
        self.draft_rpc = draft_rpc
        self.sm_controller = sm_controller
        self.spec_decode_worker = spec_decode_worker
        self.max_spec_tokens = max_spec_tokens
        self.target_sm_ratio = target_sm_ratio
        self.shared_logit_buffer = shared_logit_buffer

        # Two-queue state for pipelining
        self._draft_queue: Dict[int, SequenceGroupMetadata] = {}
        self._verify_queue: Dict[int, Tuple[SequenceGroupMetadata,
                                             Dict[str, Any]]] = {}
        self._pending_pool: Dict[int, SequenceGroupMetadata] = {}

        # Stats
        self._step_count = 0
        self._accepted = 0
        self._total_spec = 0

        # Set after each step: seq_ids present in the output
        self.last_output_seq_ids: Optional[List[int]] = None

    def step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        num_lookahead_slots: int,
    ) -> List[Any]:
        """Execute one orchestrated step (always colocated SD mode)."""
        if not seq_group_metadata_list:
            return []

        self._step_count += 1
        gamma = num_lookahead_slots or self.max_spec_tokens

        return self._step_colocated_sd(
            seq_group_metadata_list, gamma, self.target_sm_ratio)

    def _step_colocated_sd(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        gamma: int,
        target_sm_ratio: float,
    ) -> List[Any]:
        """Colocated SD mode: concurrent draft + target with SM partitioning.

        Two-queue pipelining:
        - draft_queue sequences get proposals (draft model)
        - verify_queue sequences get verified (target model)
        - Both run concurrently on partitioned SMs
        - Results are tagged with seq_ids for correct matching
        """
        stream = torch.cuda.current_stream()

        # Step 0: Move pending sequences into draft_queue (waited 1 step)
        for sid, sgm in self._pending_pool.items():
            self._draft_queue[sid] = sgm
        self._pending_pool.clear()

        # Split scheduler batch into draft/verify/prefill/new groups
        draft_seqs: List[SequenceGroupMetadata] = []
        verify_seqs: List[SequenceGroupMetadata] = []
        verify_proposals: List[Dict[str, Any]] = []
        prefill_seqs: List[SequenceGroupMetadata] = []

        for sgm in seq_group_metadata_list:
            sid = self._get_seq_id(sgm)
            if sgm.is_prompt:
                prefill_seqs.append(sgm)
            elif sid in self._verify_queue:
                meta, props = self._verify_queue.pop(sid)
                verify_seqs.append(sgm)
                verify_proposals.append(props)
            elif sid in self._draft_queue:
                self._draft_queue.pop(sid)
                draft_seqs.append(sgm)
            else:
                # New decode sequence — load-balanced entry
                draft_size = len(self._draft_queue) + len(draft_seqs)
                verify_size = len(self._verify_queue) + len(verify_seqs)
                if draft_size <= verify_size:
                    draft_seqs.append(sgm)
                else:
                    self._pending_pool[sid] = sgm

        # Bootstrap: if nothing to verify, just draft and return empty
        if not verify_seqs:
            if draft_seqs:
                # Set SM partitions
                self.sm_controller.set_full_gpu(stream)
                self.draft_rpc.set_full_gpu()

                proposals = self.draft_rpc.propose(draft_seqs, gamma)
                self._materialize_probs(proposals)
                # Move drafted sequences to verify_queue
                for i, sgm in enumerate(draft_seqs):
                    sid = self._get_seq_id(sgm)
                    self._verify_queue[sid] = (
                        sgm, self._slice_proposal(proposals, i,
                                                  len(draft_seqs)))

            self.last_output_seq_ids = None
            # Process prefills with full GPU (no verify happening)
            if prefill_seqs:
                return self._run_prefills_only(prefill_seqs)
            return []  # no-op step

        # === Concurrent phase ===
        self.sm_controller.set_partition(stream, target_sm_ratio)
        self.draft_rpc.set_partition(1.0 - target_sm_ratio)

        # Draft proposes draft_seqs concurrently
        if draft_seqs:
            self.draft_rpc.propose_async(draft_seqs, gamma)

        # Target: score verify_seqs proposals
        merged_proposals = self._merge_proposals(verify_proposals)

        # Include prefill sequences in the target batch for scoring
        target_batch = verify_seqs
        if prefill_seqs:
            target_batch = prefill_seqs + verify_seqs

        target_scores = self._score_proposals(
            target_batch, merged_proposals, gamma)

        # Collect draft results
        new_proposals = None
        if draft_seqs:
            new_proposals = self.draft_rpc.propose_collect()

        # Barrier
        torch.cuda.synchronize()

        # Sync draft KV cache for prefill sequences
        self._sync_draft_prefills(target_batch, merged_proposals)

        # Verify
        accepted, logprobs = self._verify(
            target_batch, target_scores, merged_proposals, gamma)
        self._update_acceptance_stats(accepted)

        # Build output — tagged with seq_ids from target_batch
        output = self._create_output(
            target_batch, accepted, logprobs, gamma, merged_proposals)

        # Store seq_ids present in the output for engine matching
        seq_ids = []
        for sgm in target_batch:
            seq_ids.append(self._get_seq_id(sgm))
        self.last_output_seq_ids = seq_ids

        # Rotate queues:
        # Verified sequences → draft_queue (need new proposals next step)
        for sgm in verify_seqs:
            sid = self._get_seq_id(sgm)
            self._draft_queue[sid] = sgm

        # Drafted sequences → verify_queue (have proposals, verify next step)
        if draft_seqs and new_proposals is not None:
            self._materialize_probs(new_proposals)
            for i, sgm in enumerate(draft_seqs):
                sid = self._get_seq_id(sgm)
                self._verify_queue[sid] = (
                    sgm, self._slice_proposal(new_proposals, i,
                                              len(draft_seqs)))

        return output

    def flush(self) -> List[Any]:
        """Drain pipeline: verify all remaining sequences in verify_queue."""
        if not self._verify_queue:
            return []

        stream = torch.cuda.current_stream()
        self.sm_controller.set_full_gpu(stream)

        verify_seqs = []
        verify_proposals = []
        for sid, (meta, props) in self._verify_queue.items():
            verify_seqs.append(meta)
            verify_proposals.append(props)

        gamma = self.max_spec_tokens
        merged = self._merge_proposals(verify_proposals)
        scores = self._score_proposals(verify_seqs, merged, gamma)
        accepted, logprobs = self._verify(verify_seqs, scores, merged, gamma)
        output = self._create_output(verify_seqs, accepted, logprobs,
                                     gamma, merged)
        self._verify_queue.clear()
        self._draft_queue.clear()
        self._pending_pool.clear()
        return output

    def remove_sequence(self, seq_id: int) -> None:
        """Remove finished/preempted sequence from queues."""
        self._draft_queue.pop(seq_id, None)
        self._verify_queue.pop(seq_id, None)
        self._pending_pool.pop(seq_id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_seq_id(self, sgm: SequenceGroupMetadata) -> int:
        """Extract the single sequence ID from a SequenceGroupMetadata."""
        if not sgm.seq_data:
            raise ValueError("SequenceGroupMetadata has no sequences")
        return next(iter(sgm.seq_data.keys()))

    def _dict_to_proposals(self, proposals_dict: Dict[str, Any],
                           device: torch.device) -> SpeculativeProposals:
        """Convert a dict from the draft RPC into a SpeculativeProposals."""
        token_ids = proposals_dict["proposal_token_ids"]
        lens = proposals_dict["proposal_lens"]

        # Read probs from shared GPU buffer if available (zero-copy),
        # otherwise fall back to CPU tensors sent over the pipe.
        if proposals_dict.get("probs_in_shared_buffer") and \
                self.shared_logit_buffer is not None:
            probs_view, batch_size, num_tokens = \
                self.shared_logit_buffer.read_logits()
            probs = probs_view.clone()  # detach from shared buffer
        else:
            probs = proposals_dict["proposal_probs"]
            if isinstance(probs, torch.Tensor) and probs is not None:
                probs = probs.to(device)

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.to(device)
        if isinstance(lens, torch.Tensor):
            lens = lens.to(device)
        return SpeculativeProposals(
            proposal_token_ids=token_ids,
            proposal_probs=probs,
            proposal_lens=lens,
            no_proposals=proposals_dict.get("no_proposals", False),
        )

    def _materialize_probs(self, proposals_dict: Dict[str, Any]) -> None:
        """If probs are in the shared logit buffer, read them into the dict.

        Must be called before _slice_proposal so each slice gets real probs.
        """
        if not proposals_dict.get("probs_in_shared_buffer", False):
            return
        if self.shared_logit_buffer is None:
            return
        logits, batch_size, num_tokens = self.shared_logit_buffer.read_logits()
        # logits shape: [batch_size, num_tokens, vocab_size]
        # Convert to probs (softmax) to match what _verify_tokens expects
        proposals_dict["proposal_probs"] = torch.nn.functional.softmax(
            logits, dim=-1)
        proposals_dict["probs_in_shared_buffer"] = False

    def _slice_proposal(self, proposals_dict: Dict[str, Any],
                        index: int, batch_size: int) -> Dict[str, Any]:
        """Extract a single-sequence proposal from a batched proposals dict."""
        if index < 0 or index >= batch_size:
            raise IndexError(
                f"Proposal slice index {index} out of bounds for "
                f"batch_size {batch_size}")

        result = {}
        for key in ("proposal_token_ids", "proposal_probs", "proposal_lens"):
            val = proposals_dict.get(key)
            if val is None:
                result[key] = None
            elif isinstance(val, torch.Tensor):
                result[key] = val[index:index + 1]
            elif isinstance(val, list):
                result[key] = [val[index]]
            else:
                result[key] = val
        result["no_proposals"] = proposals_dict.get("no_proposals", False)
        result["probs_in_shared_buffer"] = False  # sliced, not in buffer
        return result

    def _merge_proposals(
            self, proposal_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine individual single-sequence proposals into batched format."""
        if not proposal_list:
            return {
                "proposal_token_ids": torch.empty(0),
                "proposal_probs": None,
                "proposal_lens": torch.empty(0, dtype=torch.long),
                "no_proposals": True,
                "probs_in_shared_buffer": False,
            }

        token_ids_list = []
        probs_list = []
        lens_list = []
        has_probs = False

        for p in proposal_list:
            tid = p["proposal_token_ids"]
            if isinstance(tid, torch.Tensor):
                token_ids_list.append(tid)
            pprobs = p["proposal_probs"]
            if pprobs is not None:
                has_probs = True
                if isinstance(pprobs, torch.Tensor):
                    probs_list.append(pprobs)
            pl = p["proposal_lens"]
            if isinstance(pl, torch.Tensor):
                lens_list.append(pl)

        merged = {
            "proposal_token_ids": torch.cat(token_ids_list, dim=0)
            if token_ids_list else torch.empty(0),
            "proposal_probs": torch.cat(probs_list, dim=0)
            if has_probs and probs_list else None,
            "proposal_lens": torch.cat(lens_list, dim=0)
            if lens_list else torch.empty(0, dtype=torch.long),
            "no_proposals": all(p.get("no_proposals", False)
                                for p in proposal_list),
            "probs_in_shared_buffer": False,
        }
        return merged

    def _run_prefills_only(
            self,
            prefill_seqs: List[SequenceGroupMetadata]) -> List[Any]:
        """Run prefills through target model when no verify is happening."""
        stream = torch.cuda.current_stream()
        self.sm_controller.set_full_gpu(stream)

        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=prefill_seqs,
            num_lookahead_slots=0,
        )
        output = self.target_worker.execute_model(execute_model_req)

        # Sync draft KV for these prefills
        prefill_only = [seq for seq in prefill_seqs if seq.is_prompt]
        if prefill_only:
            self.draft_rpc.execute_prefill(prefill_only)

        return output

    def _score_proposals(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposals_dict: Dict[str, Any],
        gamma: int,
    ) -> SpeculativeScores:
        """Score proposals using the target model."""
        sdw = self.spec_decode_worker
        device = sdw.device

        proposals = self._dict_to_proposals(proposals_dict, device)

        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=gamma,
        )

        return sdw.scorer.score_proposals(execute_model_req, proposals)

    def _verify(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        target_scores: SpeculativeScores,
        proposals_dict: Dict[str, Any],
        gamma: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify proposals using speculative decoding acceptance."""
        sdw = self.spec_decode_worker
        device = sdw.device

        proposals = self._dict_to_proposals(proposals_dict, device)
        max_proposal_len = max(proposals.proposal_lens).item() \
            if isinstance(proposals.proposal_lens, torch.Tensor) \
            else max(proposals.proposal_lens)

        return sdw._verify_tokens(
            seq_group_metadata_list, target_scores,
            proposals, max_proposal_len)

    def _create_output(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        accepted_token_ids: torch.Tensor,
        logprobs: torch.Tensor,
        gamma: int,
        proposals_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Create SamplerOutput list from accepted tokens."""
        sdw = self.spec_decode_worker

        if proposals_dict is not None:
            # Use real proposal_lens from draft (handles prefill=0 correctly)
            lens = proposals_dict["proposal_lens"]
            if isinstance(lens, torch.Tensor):
                proposal_lens_list = lens.tolist()
            else:
                proposal_lens_list = list(lens)
        else:
            # Fallback: construct from batch metadata
            proposal_lens_list = []
            for sgm in seq_group_metadata_list:
                if sgm.is_prompt:
                    proposal_lens_list.append(0)
                else:
                    proposal_lens_list.append(gamma)

        stage_times = (0.0, 0.0, 0.0)  # timing not tracked in orchestrator

        return sdw._create_output_sampler_list(
            seq_group_metadata_list,
            accepted_token_ids,
            target_logprobs=logprobs,
            prompt_logprobs=None,
            k=gamma,
            stage_times=stage_times,
            proposal_lens_list=proposal_lens_list,
        )

    def _sync_draft_prefills(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposals_dict: Dict[str, Any],
    ) -> None:
        """Sync draft proposer KV cache for prefill sequences.

        After target scoring, prefill sequences (proposal_lens=0) need their
        KV cache populated in the draft model. This mirrors the logic in
        spec_decode_worker.py:921-936.
        """
        lens = proposals_dict["proposal_lens"]
        if isinstance(lens, torch.Tensor):
            proposal_lens_list = lens.tolist()
        else:
            proposal_lens_list = list(lens)

        _, (non_spec_seqs, non_spec_indices) = split_batch_by_proposal_len(
            seq_group_metadata_list, proposal_lens_list)

        if not non_spec_seqs:
            return

        # Filter to only actual prefill sequences
        prefill_seqs = [seq for seq in non_spec_seqs if seq.is_prompt]
        if prefill_seqs:
            self.draft_rpc.execute_prefill(prefill_seqs)

    def _update_acceptance_stats(self, accepted_token_ids: torch.Tensor
                                  ) -> None:
        """Update acceptance statistics."""
        if accepted_token_ids is not None and accepted_token_ids.numel() > 0:
            spec_tokens = accepted_token_ids[:, 1:]  # exclude first token
            self._total_spec += spec_tokens.numel()
            self._accepted += (spec_tokens != -1).sum().item()

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        acceptance_rate = (self._accepted / self._total_spec
                          if self._total_spec > 0 else 0.0)
        return {
            "step_count": self._step_count,
            "acceptance_rate": acceptance_rate,
            "accepted_tokens": self._accepted,
            "total_spec_tokens": self._total_spec,
        }

    def shutdown(self) -> None:
        """Gracefully shut down the draft worker."""
        logger.info("CoSpecOrchestrator shutting down. Stats: %s",
                     self.get_stats())
        try:
            remaining = self.flush()
            if remaining:
                logger.info("CoSpecOrchestrator: flushed %d remaining outputs",
                            len(remaining))
        except Exception:
            pass
        self.draft_rpc.shutdown()
        if self.shared_logit_buffer is not None:
            self.shared_logit_buffer.cleanup()
