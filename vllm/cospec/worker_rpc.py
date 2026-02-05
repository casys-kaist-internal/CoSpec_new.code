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
from array import array
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Token array type (must match vllm.sequence.VLLM_TOKEN_ID_ARRAY_TYPE)
_TOKEN_ARRAY_TYPE = "l"


def _strip_sgm_for_draft(sgm_list: List[Any]) -> List[Any]:
    """Create lightweight copies of SequenceGroupMetadata for RPC.

    Strips heavy fields that the draft model doesn't need:
    - Truncates token IDs to last token only (KV cache has the rest)
    - Removes multi_modal_data, encoder_seq_data, etc.

    Args:
        sgm_list: List of SequenceGroupMetadata objects.

    Returns:
        List of stripped SequenceGroupMetadata copies.
    """
    from vllm.sequence import SequenceData, SequenceGroupMetadata

    stripped = []
    for sgm in sgm_list:
        # Shallow copy the metadata
        new_sgm = copy.copy(sgm)

        # Strip heavy optional fields
        new_sgm.multi_modal_data = None
        new_sgm.multi_modal_placeholders = None
        new_sgm.mm_processor_kwargs = None
        new_sgm.encoder_seq_data = None
        new_sgm.cross_block_table = None
        new_sgm.computed_block_nums = None
        new_sgm.token_type_ids = None

        # For decode sequences, keep full token list to preserve correct positions.
        # The token list is small (~8KB for 1024 tokens) so RPC overhead is minimal.
        # We need the full list so model_runner computes correct attention positions.
        if sgm.seq_data and not sgm.is_prompt:
            new_seq_data = {}
            for seq_id, seq_data in sgm.seq_data.items():
                # Copy the full token sequence to preserve positions
                all_tokens = seq_data.get_token_ids()
                new_sd = SequenceData(
                    _prompt_token_ids=array(_TOKEN_ARRAY_TYPE, all_tokens),
                )
                # Copy essential state
                new_sd._num_computed_tokens = seq_data.get_num_computed_tokens()
                new_sd._stage = seq_data.stage
                new_seq_data[seq_id] = new_sd
            new_sgm.seq_data = new_seq_data

        stripped.append(new_sgm)
    return stripped


class DraftCommand(enum.Enum):
    """Commands sent from orchestrator to draft worker."""
    PROPOSE = "propose"
    SET_PARTITION = "set_partition"
    SET_FULL_GPU = "set_full_gpu"
    SET_NUM_SPEC_TOKENS = "set_num_spec_tokens"
    EXECUTE_PREFILL = "execute_prefill"
    SHUTDOWN = "shutdown"
    PING = "ping"


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

    def __init__(self, conn: multiprocessing.connection.Connection):
        self._conn = conn

    def _send_recv(self, cmd: DraftCommand,
                   kwargs: Optional[Dict[str, Any]] = None
                   ) -> Any:
        """Send a command and wait for response."""
        kwargs = kwargs or {}
        self._conn.send((cmd, kwargs))
        status, result = self._conn.recv()
        if status == DraftResponse.ERROR:
            raise RuntimeError(f"Draft worker error: {result}")
        return result

    def propose(self, seq_group_metadata_list: list,
                num_spec_tokens: int) -> Dict[str, Any]:
        """Request draft proposals (blocking).

        Args:
            seq_group_metadata_list: Sequence metadata for proposal generation.
            num_spec_tokens: Number of speculative tokens to generate (γ).

        Returns:
            Dict with proposal metadata. Actual logits are in the shared
            logit buffer (no data transfer over pipe).
        """
        # Strip heavy fields to reduce serialization overhead
        stripped = _strip_sgm_for_draft(seq_group_metadata_list)
        return self._send_recv(DraftCommand.PROPOSE, {
            "seq_group_metadata_list": stripped,
            "num_spec_tokens": num_spec_tokens,
        })

    def propose_async(self, seq_group_metadata_list: list,
                      num_spec_tokens: int) -> None:
        """Send propose command without waiting for response.

        Used for colocated SD mode where draft proposes concurrently
        with target scoring.
        """
        # Strip heavy fields to reduce serialization overhead
        stripped = _strip_sgm_for_draft(seq_group_metadata_list)
        self._conn.send((DraftCommand.PROPOSE, {
            "seq_group_metadata_list": stripped,
            "num_spec_tokens": num_spec_tokens,
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
        """Set SM partition ratio for the draft worker."""
        self._send_recv(DraftCommand.SET_PARTITION, {"ratio": ratio})

    def set_full_gpu(self) -> None:
        """Give draft worker full GPU access."""
        self._send_recv(DraftCommand.SET_FULL_GPU)

    def set_num_spec_tokens(self, num_spec_tokens: int) -> None:
        """Update the number of speculative tokens."""
        self._send_recv(DraftCommand.SET_NUM_SPEC_TOKENS, {
            "num_spec_tokens": num_spec_tokens,
        })

    def ping(self) -> bool:
        """Check if the draft worker is alive."""
        try:
            result = self._send_recv(DraftCommand.PING)
            return result == "pong"
        except Exception:
            return False

    def execute_prefill(self, seq_group_metadata_list: list) -> None:
        """Run draft model on prefill sequences to sync KV cache.

        Args:
            seq_group_metadata_list: Prefill sequences to run through draft.
        """
        self._send_recv(DraftCommand.EXECUTE_PREFILL, {
            "seq_group_metadata_list": seq_group_metadata_list,
        })

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
            if cmd == DraftCommand.PROPOSE:
                result = self._handle_propose(**kwargs)
                self._conn.send((DraftResponse.OK, result))

            elif cmd == DraftCommand.SET_PARTITION:
                if self._sm_controller is not None:
                    stream = torch.cuda.current_stream()
                    self._sm_controller.set_partition(
                        stream, kwargs["ratio"])
                self._conn.send((DraftResponse.OK, None))

            elif cmd == DraftCommand.SET_FULL_GPU:
                if self._sm_controller is not None:
                    stream = torch.cuda.current_stream()
                    self._sm_controller.set_full_gpu(stream)
                self._conn.send((DraftResponse.OK, None))

            elif cmd == DraftCommand.EXECUTE_PREFILL:
                result = self._handle_execute_prefill(**kwargs)
                self._conn.send((DraftResponse.OK, result))

            elif cmd == DraftCommand.SET_NUM_SPEC_TOKENS:
                # Forward to the underlying worker if it supports this
                num = kwargs["num_spec_tokens"]
                if hasattr(self._worker, 'set_num_spec_tokens'):
                    self._worker.set_num_spec_tokens(num)
                self._conn.send((DraftResponse.OK, None))

            elif cmd == DraftCommand.PING:
                self._conn.send((DraftResponse.OK, "pong"))

            elif cmd == DraftCommand.SHUTDOWN:
                self._running = False
                self._conn.send((DraftResponse.OK, None))

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

    def _handle_propose(self, seq_group_metadata_list: list,
                        num_spec_tokens: int) -> Dict[str, Any]:
        """Execute draft model proposals.

        Runs the draft model forward pass and writes logits to the
        shared logit buffer. Returns metadata (proposal_lens, token_ids)
        over the pipe — only small CPU tensors, not the full logits.
        """
        from vllm.sequence import ExecuteModelRequest

        # Build an ExecuteModelRequest for the draft worker
        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=num_spec_tokens,
        )

        # Run the draft model's proposal generation
        proposals = self._worker.get_spec_proposals(
            execute_model_req,
            seq_ids_with_bonus_token_in_last_step=set(),
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
            # Sync to ensure logits are visible to target process via IPC
            torch.cuda.current_stream().synchronize()

        result = {
            "proposal_token_ids": proposals.proposal_token_ids.cpu(),
            "proposal_probs": None if use_shared_buffer else (
                proposals.proposal_probs.cpu()
                if proposals.proposal_probs is not None else None),
            "proposal_lens": proposals.proposal_lens.cpu()
                if isinstance(proposals.proposal_lens, torch.Tensor)
                else proposals.proposal_lens,
            "no_proposals": proposals.no_proposals,
            "probs_in_shared_buffer": use_shared_buffer,
        }
        return result

    def _handle_execute_prefill(
            self, seq_group_metadata_list: list) -> None:
        """Run draft model forward pass on prefill sequences to sync KV cache."""
        from vllm.sequence import ExecuteModelRequest

        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=0,
        )
        self._worker.execute_model(execute_model_req, is_target=False)
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
