import time
from typing import Any, Dict
from UltraDict import UltraDict

class SharedMemory:    
    def __init__(self):
        self.shared_dict = UltraDict(name="cospec_shared_3", shared_lock=True)
            
    def put(self, key: str, value: Any) -> None:
        with self.shared_dict.lock:
            self.shared_dict[key] = value

    def get(self, key: str) -> Any:
        with self.shared_dict.lock:
            if key in self.shared_dict:
                return self.shared_dict[key]
            else:
                return None

    def wait_for_exists(self, key: str) -> None:
        while key not in self.shared_dict:
            pass 

    def delete(self, key: str) -> None:
        with self.shared_dict.lock:
            if key in self.shared_dict:
                del self.shared_dict[key]

    def __del__(self):
        self.shared_dict.close()
