import os
import fcntl
import pickle
from pathlib import Path
from typing import Any

class SharedMemoryManager:
    """Atomic shared memory store with proper file locking"""
    
    def __init__(self, namespace: str = "cospec"):
        self.namespace = namespace
        self.lock_path = f"/tmp/{namespace}.lock"
        self.shm_dir = Path(f"/dev/shm/{namespace}")
        self.shm_dir.mkdir(exist_ok=True)
        self.lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        self._init_barrier()

    def _init_barrier(self):
        """Setup synchronization barrier for 2 processes"""
        self.barrier_file = self.shm_dir / "barrier"
        if not self.barrier_file.exists():
            with open(self.barrier_file, "w") as f:
                f.write("2|0")  # Hardcode 2 parties

    def create_barrier(self):
        """Initialize synchronization barrier for 2 processes"""
        with open(self.barrier_file, "w") as f:
            f.write("2|0")  # Remove num_parties parameter

    def wait_at_barrier(self):
        """Wait until both processes reach the barrier"""
        while True:
            with open(self.barrier_file, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                current = int(f.read().split("|")[1])
                
                if current + 1 >= 2:
                    f.seek(0)
                    f.write("2|0")
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return
                
                f.seek(0)
                f.write(f"2|{current + 1}")
                fcntl.flock(f, fcntl.LOCK_UN)

    def synchronized_put(self, key: str, value: Any):
        """Atomic put with barrier synchronization"""
        self.wait_at_barrier()
        self.put(key, value)
        self.wait_at_barrier()
    
    def synchronized_get(self, key: str) -> Any:
        """Atomic get with barrier synchronization"""
        self.wait_at_barrier()
        value = self.get(key)
        self.wait_at_barrier()
        return value

    def lock(self):
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX)

    def unlock(self):
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

    def put(self, key: str, value: Any) -> None:
        """Atomic write with exclusive lock"""
        key_path = self.shm_dir / f"{key}.pkl"
        
        # Exclusive lock during write
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
        try:
            with open(key_path, "wb") as f:
                pickle.dump(value, f)
                os.fsync(f.fileno())
        finally:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

    def get(self, key: str) -> Any:
        """Block until data is available with proper locking"""
        key_path = self.shm_dir / f"{key}.pkl"
        
        while True:
            # Blocking shared lock
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
            try:
                if key_path.exists():
                    with open(key_path, "rb") as f:
                        return pickle.load(f)
            finally:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

    def __exit__(self, *args):
        if self.lock_fd:
            os.close(self.lock_fd)

        if self.barrier_file.exists():
            with open(self.barrier_file, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.truncate(0)
                f.write("0")
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
