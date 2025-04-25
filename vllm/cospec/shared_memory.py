import mmap
import os
import posix_ipc
import numpy as np
from typing import Tuple

class CoSpecSharedMemory:
    def __init__(self, name: str = "/vllm_cospec", size: int = 2 * 1024**3):  # 2GB
        self.name = name
        self.size = size
        self._semaphore = posix_ipc.Semaphore(name, flags=posix_ipc.O_CREAT)
        self._setup_memory()

    def _setup_memory(self):
        # Create POSIX shared memory
        self.shm_fd = posix_ipc.SharedMemory(
            self.name,
            flags=posix_ipc.O_CREAT,
            size=self.size
        )
        self.mmap = mmap.mmap(
            self.shm_fd.fd,
            self.size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
        
        # Use numpy for zero-copy buffer access
        self.buffer = np.ndarray(
            (self.size,), 
            dtype=np.uint8,
            buffer=self.mmap
        )

    def write_data(self, data: np.ndarray, offset: int = 0) -> None:
        with self._semaphore:
            end = offset + data.nbytes
            self.buffer[offset:end] = np.frombuffer(data.tobytes(), dtype=np.uint8)

    def read_data(self, shape: Tuple[int], dtype: np.dtype, offset: int = 0) -> np.ndarray:
        with self._semaphore:
            itemsize = np.dtype(dtype).itemsize
            return np.ndarray(
                shape,
                dtype=dtype,
                buffer=self.buffer[offset:offset+itemsize*shape[0]]
            ).copy()

    def close(self):
        self.mmap.close()
        self.shm_fd.unlink()
        self._semaphore.unlink()

    def __del__(self):
        self.close()