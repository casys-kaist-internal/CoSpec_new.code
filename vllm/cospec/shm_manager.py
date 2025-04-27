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
