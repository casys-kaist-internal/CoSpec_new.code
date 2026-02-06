"""Shared GPU Memory via CUDA IPC for CoSpec v2.

Provides SharedLogitBuffer for sharing draft model logits between
target and draft processes without copies. Target process allocates and exports
CUDA IPC handles to /dev/shm; draft process imports them.
"""

import os
import pickle
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOGIT_BUFFER_SHM_PATH = "/dev/shm/cospec_logit_buffer_{instance_id}.pkl"


class SharedLogitBuffer:
    """Shared GPU buffer for draft model logits.

    Pre-allocated buffer [max_batch, max_spec_tokens, vocab_size] shared
    between processes. Draft writes logits; target reads for verification.
    Metadata (batch_size, num_tokens) is stored in a separate shared tensor
    to avoid file I/O on the hot path.

    Args:
        max_batch: Maximum batch size.
        max_spec_tokens: Maximum speculative tokens (γ).
        vocab_size: Vocabulary size.
        dtype: Data type for logits.
        mode: 'owner' for target, 'client' for draft.
        instance_id: Unique identifier for this buffer.
    """

    def __init__(
        self,
        max_batch: int,
        max_spec_tokens: int,
        vocab_size: int,
        dtype: torch.dtype = torch.float32,
        mode: str = "owner",
        instance_id: str = "default",
    ):
        assert mode in ("owner", "client"), f"Invalid mode: {mode}"
        self.mode = mode
        self.instance_id = instance_id
        self._handle_path = _LOGIT_BUFFER_SHM_PATH.format(instance_id=instance_id)
        self.max_batch = max_batch
        self.max_spec_tokens = max_spec_tokens
        self.vocab_size = vocab_size
        self.dtype = dtype

        if mode == "owner":
            self._buffer, self._meta_buffer = self._allocate_and_export()
        else:
            self._buffer, self._meta_buffer = self._import()

    def _allocate_and_export(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Owner: allocate buffers and export IPC handles."""
        # Main logits buffer
        shape = (self.max_batch, self.max_spec_tokens, self.vocab_size)
        buffer = torch.zeros(shape, dtype=self.dtype, device="cuda")

        storage = buffer.untyped_storage()
        cuda_ipc_handle = storage._share_cuda_()

        # Metadata buffer [batch_size, num_tokens] - 2 int64 values
        meta_buffer = torch.zeros(2, dtype=torch.int64, device="cuda")
        meta_storage = meta_buffer.untyped_storage()
        meta_ipc_handle = meta_storage._share_cuda_()

        handle_info = {
            "shape": shape,
            "dtype": self.dtype,
            "cuda_ipc_handle": cuda_ipc_handle,
            "storage_offset": buffer.storage_offset(),
            "meta_ipc_handle": meta_ipc_handle,
            "meta_storage_offset": meta_buffer.storage_offset(),
        }

        with open(self._handle_path, "wb") as f:
            pickle.dump(handle_info, f)

        logger.info("SharedLogitBuffer: allocated %s, exported to %s",
                    shape, self._handle_path)
        return buffer, meta_buffer

    def _import(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Client: import buffers from IPC handles."""
        if not os.path.exists(self._handle_path):
            raise FileNotFoundError(
                f"Logit buffer IPC handle not found: {self._handle_path}. "
                "Ensure target process started first.")

        with open(self._handle_path, "rb") as f:
            handle_info = pickle.load(f)

        # Import main buffer
        storage = torch.UntypedStorage._new_shared_cuda(
            *handle_info["cuda_ipc_handle"])
        buffer = torch.empty(
            handle_info["shape"], dtype=handle_info["dtype"], device="cuda")
        buffer.set_(storage, handle_info["storage_offset"],
                    handle_info["shape"])

        # Import metadata buffer
        meta_storage = torch.UntypedStorage._new_shared_cuda(
            *handle_info["meta_ipc_handle"])
        meta_buffer = torch.empty(2, dtype=torch.int64, device="cuda")
        meta_buffer.set_(meta_storage, handle_info["meta_storage_offset"], (2,))

        logger.info("SharedLogitBuffer: imported %s from %s",
                    handle_info["shape"], self._handle_path)
        return buffer, meta_buffer

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    def write_logits(self, logits: torch.Tensor, batch_size: int,
                     num_tokens: int) -> None:
        """Write draft logits to the shared buffer."""
        assert batch_size <= self.max_batch, (
            f"batch_size {batch_size} exceeds max_batch {self.max_batch}")
        assert num_tokens <= self.max_spec_tokens, (
            f"num_tokens {num_tokens} exceeds max_spec_tokens "
            f"{self.max_spec_tokens}")
        self._buffer[:batch_size, :num_tokens, :] = logits
        self._meta_buffer[0] = batch_size
        self._meta_buffer[1] = num_tokens

    def read_logits(self) -> tuple[torch.Tensor, int, int]:
        """Read draft logits from the shared buffer."""
        batch_size = self._meta_buffer[0].item()
        num_tokens = self._meta_buffer[1].item()
        return self._buffer[:batch_size, :num_tokens, :], batch_size, num_tokens

    def cleanup(self) -> None:
        """Remove IPC handle files (owner only)."""
        if self.mode != "owner":
            return
        if os.path.exists(self._handle_path):
            os.remove(self._handle_path)
        logger.info("SharedLogitBuffer: cleaned up handle files")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
