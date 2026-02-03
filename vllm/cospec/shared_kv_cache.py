"""Shared KV Cache via CUDA IPC for CoSpec v2.

The target process allocates KV cache tensors and exports CUDA IPC handles
to /dev/shm. The draft process opens these handles to access the same GPU
memory, enabling shared KV cache between the two processes without copies.

Pattern follows vLLM's SharedMemoryModelLoader (loader.py:1517-1627) which
uses storage._share_cuda_() / torch.Storage._new_shared_cuda().
"""

import os
import pickle
from typing import List, Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Shared memory path prefix for IPC handles
_SHM_PREFIX = "/dev/shm/cospec_kv_cache"


class SharedKVCacheAllocator:
    """Manages shared KV cache allocation and IPC handle exchange.

    In 'owner' mode (target process): allocates KV cache tensors on GPU,
    exports CUDA IPC handles to shared memory files.

    In 'client' mode (draft process): opens CUDA IPC handles from shared
    memory files to access the same GPU tensors.

    Args:
        mode: 'owner' for target process, 'client' for draft process.
        instance_id: Unique identifier to namespace the shared memory files.
    """

    def __init__(self, mode: str, instance_id: str = "default"):
        assert mode in ("owner", "client"), f"Invalid mode: {mode}"
        self.mode = mode
        self.instance_id = instance_id
        self._shm_dir = f"{_SHM_PREFIX}_{instance_id}"
        self._handles: List[dict] = []

        if mode == "owner":
            os.makedirs(self._shm_dir, exist_ok=True)

    def allocate_shared(
        self,
        kv_cache_shape: tuple,
        dtype: torch.dtype,
        num_layers: int,
        device: str = "cuda",
    ) -> List[torch.Tensor]:
        """Allocate KV cache and export/import IPC handles.

        Args:
            kv_cache_shape: Shape of each layer's KV cache tensor.
            dtype: Data type for the cache tensors.
            num_layers: Number of attention layers.
            device: Device to allocate on (only 'cuda' supported for IPC).

        Returns:
            List of KV cache tensors, one per layer.
        """
        if self.mode == "owner":
            return self._allocate_and_export(
                kv_cache_shape, dtype, num_layers, device)
        else:
            return self._import_from_handles(num_layers)

    def _allocate_and_export(
        self,
        kv_cache_shape: tuple,
        dtype: torch.dtype,
        num_layers: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Owner: allocate tensors and export IPC handles."""
        kv_cache: List[torch.Tensor] = []

        for layer_idx in range(num_layers):
            tensor = torch.zeros(kv_cache_shape, dtype=dtype, device=device)

            # Export CUDA IPC handle
            storage = tensor.untyped_storage()
            cuda_ipc_handle = storage._share_cuda_()

            handle_info = {
                "shape": kv_cache_shape,
                "dtype": dtype,
                "storage_size": storage.size(),
                "storage_offset": tensor.storage_offset(),
                "cuda_ipc_handle": cuda_ipc_handle,
            }

            handle_path = os.path.join(self._shm_dir, f"layer_{layer_idx}.pkl")
            with open(handle_path, "wb") as f:
                pickle.dump(handle_info, f)

            kv_cache.append(tensor)
            logger.debug("Exported KV cache layer %d IPC handle to %s",
                         layer_idx, handle_path)

        self._num_layers = num_layers
        logger.info("SharedKVCache owner: exported %d layers to %s",
                     num_layers, self._shm_dir)
        return kv_cache

    def _import_from_handles(self, num_layers: int) -> List[torch.Tensor]:
        """Client: import tensors from IPC handles."""
        kv_cache: List[torch.Tensor] = []

        for layer_idx in range(num_layers):
            handle_path = os.path.join(self._shm_dir, f"layer_{layer_idx}.pkl")

            # Wait for handle file to be available
            if not os.path.exists(handle_path):
                raise FileNotFoundError(
                    f"KV cache IPC handle not found: {handle_path}. "
                    "Ensure the target process (owner) has started first.")

            with open(handle_path, "rb") as f:
                handle_info = pickle.load(f)

            # Reconstruct tensor from CUDA IPC handle
            cuda_ipc_data = handle_info["cuda_ipc_handle"]
            storage = torch.UntypedStorage._new_shared_cuda(
                *cuda_ipc_data)
            tensor = torch.empty(
                handle_info["shape"],
                dtype=handle_info["dtype"],
                device="cuda",
            )
            tensor.set_(storage, handle_info["storage_offset"],
                        handle_info["shape"])

            kv_cache.append(tensor)
            logger.debug("Imported KV cache layer %d from IPC handle",
                         layer_idx)

        logger.info("SharedKVCache client: imported %d layers from %s",
                     num_layers, self._shm_dir)
        return kv_cache

    def cleanup(self) -> None:
        """Remove shared memory handle files (owner only)."""
        if self.mode != "owner":
            return
        import shutil
        if os.path.exists(self._shm_dir):
            shutil.rmtree(self._shm_dir)
            logger.info("SharedKVCache owner: cleaned up %s", self._shm_dir)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
