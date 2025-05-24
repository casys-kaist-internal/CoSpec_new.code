import os
import fcntl
import pickle
import time
from pathlib import Path
from typing import Any

class SharedMemory:    
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
            # Acquire shared (reader) lock; this blocks only if a writer holds EX lock.
            fcntl.flock(self.lock_fd, fcntl.LOCK_SH)
            try:
                if key_path.exists():
                    with open(key_path, "rb") as f:
                        return pickle.load(f)
            finally:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            
    def check_exists(self, key: str) -> bool:
        """Check if the key exists in the shared memory"""
        key_path = self.shm_dir / f"{key}.pkl"
        return key_path.exists()
    
    def wait_for_exists(self, key: str) -> None:
        """Wait until the key exists in the shared memory"""
        while not self.check_exists(key):
            pass

    def delete(self, key: str) -> None:
        """Delete the key from the shared memory"""
        key_path = self.shm_dir / f"{key}.pkl"

        # Exclusive lock during delete
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
        try:
            if key_path.exists():
                os.remove(key_path)
        finally:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

    def get_and_delete(self, key: str) -> Any:
        """Get and delete the key from the shared memory"""
        value = self.get(key)
        self.delete(key)
        return value

    def get_nowait(self, key: str) -> Any:
        """Non-blocking get. Returns None if data is not available."""
        key_path = self.shm_dir / f"{key}.pkl"
        
        fcntl.flock(self.lock_fd, fcntl.LOCK_SH)
            
        try:
            if key_path.exists():
                with open(key_path, "rb") as f:
                    return pickle.load(f)
            return None
        finally:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'lock_fd') and self.lock_fd:
            os.close(self.lock_fd)