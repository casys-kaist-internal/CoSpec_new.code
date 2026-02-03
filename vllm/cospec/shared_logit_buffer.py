"""Shared Logit Buffer via CUDA IPC for CoSpec v2.

Pre-allocated GPU buffer [max_batch, max_spec_tokens, vocab_size] shared
between target and draft processes. The draft process writes logits here;
the target process reads them for verification (needed when temp > 0).

Target allocates + exports IPC handle. Draft opens it.
"""

import os
import pickle
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SHM_HANDLE_PATH = "/dev/shm/cospec_logit_buffer_{instance_id}.pkl"


class SharedLogitBuffer:
    """Shared GPU buffer for draft model logits.

    Args:
        max_batch: Maximum batch size.
        max_spec_tokens: Maximum number of speculative tokens (γ).
        vocab_size: Vocabulary size of the model.
        dtype: Data type for logits (typically float32 or float16).
        mode: 'owner' for target process, 'client' for draft process.
        instance_id: Unique identifier for this buffer instance.
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
        self._handle_path = _SHM_HANDLE_PATH.format(instance_id=instance_id)
        self.max_batch = max_batch
        self.max_spec_tokens = max_spec_tokens
        self.vocab_size = vocab_size
        self.dtype = dtype

        if mode == "owner":
            self._buffer = self._allocate_and_export()
        else:
            self._buffer = self._import_from_handle()

        # Metadata buffer for communicating actual dimensions per step.
        # Stored in /dev/shm as a small pickle: (actual_batch, actual_tokens)
        self._meta_path = f"/dev/shm/cospec_logit_meta_{instance_id}.pkl"

    def _allocate_and_export(self) -> torch.Tensor:
        """Owner: allocate buffer and export IPC handle."""
        shape = (self.max_batch, self.max_spec_tokens, self.vocab_size)
        buffer = torch.zeros(shape, dtype=self.dtype, device="cuda")

        storage = buffer.untyped_storage()
        cuda_ipc_handle = storage._share_cuda_()
        handle_info = {
            "shape": shape,
            "dtype": self.dtype,
            "cuda_ipc_handle": cuda_ipc_handle,
            "storage_offset": buffer.storage_offset(),
        }

        with open(self._handle_path, "wb") as f:
            pickle.dump(handle_info, f)

        logger.info("SharedLogitBuffer owner: allocated %s, exported to %s",
                     shape, self._handle_path)
        return buffer

    def _import_from_handle(self) -> torch.Tensor:
        """Client: import buffer from IPC handle."""
        if not os.path.exists(self._handle_path):
            raise FileNotFoundError(
                f"Logit buffer IPC handle not found: {self._handle_path}. "
                "Ensure the target process (owner) has started first.")

        with open(self._handle_path, "rb") as f:
            handle_info = pickle.load(f)

        cuda_ipc_data = handle_info["cuda_ipc_handle"]
        storage = torch.UntypedStorage._new_shared_cuda(*cuda_ipc_data)
        buffer = torch.empty(
            handle_info["shape"], dtype=handle_info["dtype"], device="cuda")
        buffer.set_(storage, handle_info["storage_offset"],
                    handle_info["shape"])

        logger.info("SharedLogitBuffer client: imported %s from %s",
                     handle_info["shape"], self._handle_path)
        return buffer

    @property
    def buffer(self) -> torch.Tensor:
        """Access the underlying shared buffer tensor."""
        return self._buffer

    def write_logits(self, logits: torch.Tensor, batch_size: int,
                     num_tokens: int) -> None:
        """Write draft logits to the shared buffer.

        Args:
            logits: Tensor of shape [batch_size, num_tokens, vocab_size].
            batch_size: Actual batch size for this step.
            num_tokens: Actual number of speculative tokens for this step.
        """
        self._buffer[:batch_size, :num_tokens, :] = logits
        # Write metadata so the reader knows actual dimensions
        with open(self._meta_path, "wb") as f:
            pickle.dump((batch_size, num_tokens), f)

    def read_logits(self) -> tuple[torch.Tensor, int, int]:
        """Read draft logits from the shared buffer.

        Returns:
            Tuple of (logits_view, batch_size, num_tokens).
        """
        with open(self._meta_path, "rb") as f:
            batch_size, num_tokens = pickle.load(f)
        return self._buffer[:batch_size, :num_tokens, :], batch_size, num_tokens

    def cleanup(self) -> None:
        """Remove IPC handle files (owner only)."""
        if self.mode != "owner":
            return
        for path in (self._handle_path, self._meta_path):
            if os.path.exists(path):
                os.remove(path)
        logger.info("SharedLogitBuffer owner: cleaned up handle files")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
