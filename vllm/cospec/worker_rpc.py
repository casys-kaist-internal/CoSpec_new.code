"""Draft Worker RPC for CoSpec v2.

Communication between orchestrator (target process) and draft worker
(draft process) via multiprocessing.Connection. Only lightweight metadata
flows over the pipe; large data (logits, KV cache) is shared via GPU
memory (CUDA IPC).

Protocol:
- Orchestrator sends commands as (cmd_type, kwargs) tuples.
- Draft worker executes and sends back (status, result) tuples.
"""

import copy
import enum
import multiprocessing
import multiprocessing.connection
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.profiler import record_function

from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest

logger = init_logger(__name__)


def _strip_sgm_for_pipe(seq_group_metadata_list: list) -> list:
    """Create lightweight copies of SequenceGroupMetadata for pipe send.

    Strips redundant token caches from SequenceData to reduce pickle size.
    The draft worker needs _prompt_token_ids, _output_token_ids, and
    _cached_all_token_ids (used by get_token_ids() in model runner).
    We can safely clear _prompt_token_ids_tuple and _new_appended_tokens.

    Uses shallow copy of SGM + shallow copy of SequenceData to avoid
    modifying the originals (which are still used by the target scorer).
    """
    stripped = []
    for sgm in seq_group_metadata_list:
        sgm_copy = copy.copy(sgm)
        new_seq_data = {}
        for sid, sd in sgm.seq_data.items():
            sd_copy = copy.copy(sd)
            # _prompt_token_ids_tuple is a redundant copy of
            # _prompt_token_ids, only used by the prompt_token_ids
            # property (not called during model forward pass).
            sd_copy._prompt_token_ids_tuple = ()
            # _new_appended_tokens is for delta tracking, not needed
            # for draft model forward pass.
            sd_copy._new_appended_tokens = []
            new_seq_data[sid] = sd_copy
        sgm_copy.seq_data = new_seq_data
        # Clear fields not needed by draft worker
        sgm_copy.multi_modal_data = None
        sgm_copy.multi_modal_placeholders = None
        sgm_copy.mm_processor_kwargs = None
        sgm_copy.encoder_seq_data = None
        sgm_copy.cross_block_table = None
        stripped.append(sgm_copy)
    return stripped


class DraftCommand(enum.Enum):
    """Commands sent from orchestrator to draft worker."""
    PROPOSE = "propose"
    SET_PARTITION = "set_partition"
    SET_FULL_GPU = "set_full_gpu"
    EXECUTE_PREFILL = "execute_prefill"
    SHUTDOWN = "shutdown"


class DraftResponse(enum.Enum):
    """Response status from draft worker."""
    OK = "ok"
    ERROR = "error"


class DraftWorkerRPC:
    """Client-side RPC proxy for the draft worker.

    Lives in the target (orchestrator) process. Sends commands to the
    draft worker process over a pipe connection.

    Args:
        conn: The parent-side of a multiprocessing.Pipe().
    """

    _RPC_TIMEOUT_S = 60.0

    def __init__(self, conn: multiprocessing.connection.Connection):
        self._conn = conn

    def _send_recv(self, cmd: DraftCommand,
                   kwargs: Optional[Dict[str, Any]] = None
                   ) -> Any:
        """Send a command and wait for response."""
        with record_function("cospec::rpc_send_recv"):
            kwargs = kwargs or {}
            self._conn.send((cmd, kwargs))
            if not self._conn.poll(timeout=self._RPC_TIMEOUT_S):
                raise RuntimeError(
                    f"Draft worker did not respond to {cmd.value} "
                    f"within {self._RPC_TIMEOUT_S}s (may have crashed)")
            status, result = self._conn.recv()
            if status == DraftResponse.ERROR:
                raise RuntimeError(f"Draft worker error: {result}")
            return result

    def propose(self, seq_group_metadata_list: list,
                num_spec_tokens: int,
                seq_ids_with_bonus_token: Optional[Set[int]] = None,
                ) -> Dict[str, Any]:
        """Request draft proposals (blocking).

        Args:
            seq_group_metadata_list: Sequence metadata for proposal generation.
            num_spec_tokens: Number of speculative tokens to generate.
            seq_ids_with_bonus_token: Seq IDs that received a bonus token
                from the last verification step. Passed to the draft model's
                multi-step worker for correct KV cache handling.

        Returns:
            Dict with proposal metadata. Actual logits are in the shared
            logit buffer (no data transfer over pipe).
        """
        return self._send_recv(DraftCommand.PROPOSE, {
            "seq_group_metadata_list": _strip_sgm_for_pipe(
                seq_group_metadata_list),
            "num_spec_tokens": num_spec_tokens,
            "seq_ids_with_bonus_token": seq_ids_with_bonus_token or set(),
        })

    def propose_async(self, seq_group_metadata_list: list,
                      num_spec_tokens: int,
                      seq_ids_with_bonus_token: Optional[Set[int]] = None,
                      ) -> None:
        """Send propose command without waiting for response.

        Used for colocated SD mode where draft proposes concurrently
        with target scoring.
        """
        self._conn.send((DraftCommand.PROPOSE, {
            "seq_group_metadata_list": _strip_sgm_for_pipe(
                seq_group_metadata_list),
            "num_spec_tokens": num_spec_tokens,
            "seq_ids_with_bonus_token": seq_ids_with_bonus_token or set(),
        }))

    def propose_collect(self, timeout: float = 60.0) -> Dict[str, Any]:
        """Collect response from a previous propose_async call.

        Args:
            timeout: Maximum seconds to wait for response.

        Raises:
            RuntimeError: If draft worker responds with error or times out.
        """
        if not self._conn.poll(timeout):
            raise RuntimeError(
                f"Draft worker did not respond within {timeout}s "
                "(may have crashed)")
        status, result = self._conn.recv()
        if status == DraftResponse.ERROR:
            raise RuntimeError(f"Draft worker error: {result}")
        return result

    def set_partition(self, ratio: float) -> None:
        """Set SM partition ratio for the draft worker (blocking)."""
        self._send_recv(DraftCommand.SET_PARTITION, {"ratio": ratio})

    def set_partition_async(self, ratio: float) -> None:
        """Set SM partition ratio for the draft worker (fire-and-forget).

        Pipe is FIFO — draft processes set_partition before the next
        propose command, so no response is needed.
        """
        self._conn.send((DraftCommand.SET_PARTITION, {
            "ratio": ratio,
            "_no_response": True,
        }))

    def set_full_gpu(self) -> None:
        """Give draft worker full GPU access (blocking)."""
        self._send_recv(DraftCommand.SET_FULL_GPU)

    def set_full_gpu_async(self) -> None:
        """Give draft worker full GPU access (fire-and-forget)."""
        self._conn.send((DraftCommand.SET_FULL_GPU, {
            "_no_response": True,
        }))

    def execute_prefill(self, seq_group_metadata_list: list) -> None:
        """Run draft model on prefill sequences to sync KV cache (blocking).

        Args:
            seq_group_metadata_list: Prefill sequences to run through draft.
        """
        self._send_recv(DraftCommand.EXECUTE_PREFILL, {
            "seq_group_metadata_list": _strip_sgm_for_pipe(
                seq_group_metadata_list),
        })

    def execute_prefill_async(self, seq_group_metadata_list: list) -> None:
        """Run draft model on prefill sequences (fire-and-forget).

        Pipe FIFO guarantees ordering: the next step's set_partition and
        propose arrive after this execute_prefill, so draft KV cache
        is synced before the next propose starts.
        """
        self._conn.send((DraftCommand.EXECUTE_PREFILL, {
            "seq_group_metadata_list": _strip_sgm_for_pipe(
                seq_group_metadata_list),
            "_no_response": True,
        }))

    def shutdown(self) -> None:
        """Gracefully shut down the draft worker."""
        try:
            self._conn.send((DraftCommand.SHUTDOWN, {}))
            # Wait for acknowledgment before closing (with short timeout)
            if self._conn.poll(timeout=2.0):
                self._conn.recv()  # Discard response
        except Exception:
            pass
        finally:
            try:
                self._conn.close()
            except Exception:
                pass


class DraftWorkerServer:
    """Server-side handler in the draft worker process.

    Receives commands from the orchestrator, executes them using the
    draft model, and writes logits to the shared logit buffer.

    Args:
        conn: The child-side of a multiprocessing.Pipe().
        draft_worker: The actual draft model worker (e.g., MultiStepWorker).
        sm_controller: Optional SMController for SM partitioning.
        shared_logit_buffer: Optional SharedLogitBuffer for writing logits.
    """

    def __init__(
        self,
        conn: multiprocessing.connection.Connection,
        draft_worker: Any,
        sm_controller: Optional[Any] = None,
        shared_logit_buffer: Optional[Any] = None,
    ):
        self._conn = conn
        self._worker = draft_worker
        self._sm_controller = sm_controller
        self._shared_logit_buffer = shared_logit_buffer
        self._running = False

        # Dispatch table: avoids if/elif chain on every command
        self._dispatch = {
            DraftCommand.PROPOSE: self._dispatch_propose,
            DraftCommand.SET_PARTITION: self._dispatch_set_partition,
            DraftCommand.SET_FULL_GPU: self._dispatch_set_full_gpu,
            DraftCommand.EXECUTE_PREFILL: self._dispatch_execute_prefill,
            DraftCommand.SHUTDOWN: self._dispatch_shutdown,
        }

    def serve(self) -> None:
        """Main event loop. Blocks until shutdown command received."""
        self._running = True
        logger.info("DraftWorkerServer: starting event loop")

        while self._running:
            try:
                if not self._conn.poll(timeout=1.0):
                    continue

                cmd, kwargs = self._conn.recv()
                self._handle_command(cmd, kwargs)

            except EOFError:
                logger.info("DraftWorkerServer: connection closed")
                break
            except Exception as e:
                logger.error("DraftWorkerServer: unexpected error: %s", e)
                traceback.print_exc()
                break

        logger.info("DraftWorkerServer: event loop ended")

    def _handle_command(self, cmd: DraftCommand,
                        kwargs: Dict[str, Any]) -> None:
        """Dispatch a command and send response."""
        try:
            handler = self._dispatch.get(cmd)
            if handler is not None:
                handler(kwargs)
            else:
                self._conn.send((DraftResponse.ERROR,
                                 f"Unknown command: {cmd}"))
        except Exception as e:
            logger.error("DraftWorkerServer: error handling %s: %s", cmd, e)
            traceback.print_exc()
            try:
                self._conn.send((DraftResponse.ERROR, str(e)))
            except Exception:
                pass

    def _dispatch_propose(self, kwargs: Dict[str, Any]) -> None:
        result = self._handle_propose(**kwargs)
        self._conn.send((DraftResponse.OK, result))

    def _dispatch_set_partition(self, kwargs: Dict[str, Any]) -> None:
        if self._sm_controller is not None:
            stream = torch.cuda.current_stream()
            self._sm_controller.set_partition(stream, kwargs["ratio"])
        if not kwargs.get("_no_response"):
            self._conn.send((DraftResponse.OK, None))

    def _dispatch_set_full_gpu(self, kwargs: Dict[str, Any]) -> None:
        if self._sm_controller is not None:
            stream = torch.cuda.current_stream()
            self._sm_controller.set_full_gpu(stream)
        if not kwargs.get("_no_response"):
            self._conn.send((DraftResponse.OK, None))

    def _dispatch_execute_prefill(self, kwargs: Dict[str, Any]) -> None:
        no_response = kwargs.pop("_no_response", False)
        result = self._handle_execute_prefill(**kwargs)
        if not no_response:
            self._conn.send((DraftResponse.OK, result))

    def _dispatch_shutdown(self, kwargs: Dict[str, Any]) -> None:
        self._running = False
        self._conn.send((DraftResponse.OK, None))

    def _handle_propose(self, seq_group_metadata_list: list,
                        num_spec_tokens: int,
                        seq_ids_with_bonus_token: Optional[Set[int]] = None,
                        ) -> Dict[str, Any]:
        """Execute draft model proposals.

        Runs the draft model forward pass and writes logits to the
        shared logit buffer. Returns metadata (proposal_lens, token_ids)
        over the pipe — only small CPU tensors, not the full logits.

        Args:
            seq_group_metadata_list: Sequence metadata for proposals.
            num_spec_tokens: Number of speculative tokens to generate.
            seq_ids_with_bonus_token: Seq IDs that had bonus tokens from
                the last verification. The multi-step worker uses this to
                expand the batch and correctly handle the KV cache for the
                bonus token position (same mechanism as regular SD).
        """

        bonus_set = seq_ids_with_bonus_token or set()

        # Fix num_computed_tokens for decode sequences. The scheduler sets
        # this based on the target model's view (all tokens computed), but
        # the draft model_runner with is_multi_step=False uses it to compute
        # context_len. If num_computed == seq_len, the model_runner gets 0
        # tokens to process. Set it to seq_len - 1 so there's always 1 token
        # to compute (matching what the is_multi_step=True path does).
        for sgm in seq_group_metadata_list:
            if not sgm.is_prompt:
                for seq_id, seq_data in sgm.seq_data.items():
                    seq_len = seq_data.get_len()
                    num_computed = seq_data.get_num_computed_tokens()
                    if num_computed >= seq_len:
                        delta = num_computed - (seq_len - 1)
                        seq_data.update_num_computed_tokens(-delta)

        # Build an ExecuteModelRequest for the draft worker
        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=num_spec_tokens,
        )

        # Run the draft model's proposal generation.
        # Pass bonus token seq_ids so the multi-step worker can expand
        # the batch and correctly recompute KV for the bonus position.
        proposals = self._worker.get_spec_proposals(
            execute_model_req,
            seq_ids_with_bonus_token_in_last_step=bonus_set,
        )

        # When shared logit buffer is available, write probs there
        # and skip serializing the large probs tensor over the pipe.
        use_shared_buffer = (self._shared_logit_buffer is not None
                             and proposals.proposal_probs is not None)
        if use_shared_buffer:
            batch_size = proposals.proposal_probs.shape[0]
            num_tokens = proposals.proposal_probs.shape[1]
            self._shared_logit_buffer.write_logits(
                proposals.proposal_probs, batch_size, num_tokens)
            # No explicit sync needed: the .cpu() calls below implicitly
            # synchronize the current stream, which guarantees the shared
            # buffer write is committed before we send the pipe response.

        # Use non_blocking=True for GPU→CPU transfers to avoid implicit
        # sync per tensor. Single explicit sync before pipe send.
        token_ids_cpu = proposals.proposal_token_ids.to(
            "cpu", non_blocking=True)
        probs_cpu = None
        if not use_shared_buffer and proposals.proposal_probs is not None:
            probs_cpu = proposals.proposal_probs.to(
                "cpu", non_blocking=True)
        lens_cpu = (proposals.proposal_lens.to("cpu", non_blocking=True)
                    if isinstance(proposals.proposal_lens, torch.Tensor)
                    else proposals.proposal_lens)
        # Single sync ensures all non-blocking transfers complete
        torch.cuda.synchronize()

        result = {
            "proposal_token_ids": token_ids_cpu,
            "proposal_probs": probs_cpu,
            "proposal_lens": lens_cpu,
            "no_proposals": proposals.no_proposals,
            "probs_in_shared_buffer": use_shared_buffer,
        }
        return result

    def _handle_execute_prefill(
            self, seq_group_metadata_list: list) -> None:
        """Run draft model forward pass on prefill sequences to sync KV cache."""
        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=0,
        )
        self._worker.execute_model(execute_model_req)
        return None


def create_draft_worker_pipe() -> Tuple[
    multiprocessing.connection.Connection,
    multiprocessing.connection.Connection,
]:
    """Create a pipe pair for orchestrator <-> draft worker communication.

    Returns:
        (parent_conn, child_conn) tuple.
    """
    return multiprocessing.Pipe(duplex=True)
