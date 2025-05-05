import torch
import time
import os
import fcntl

from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.cospec.shm import SharedMemory
from vllm.cospec.profiler import Profiler



logger = init_logger(__name__)

class CospecManager:
    def __init__(self, vllm_config: VllmConfig):
        self.shm = SharedMemory()
        self.profiler = Profiler(vllm_config)
        self.is_primary = vllm_config.speculative_config.is_primary
        self.start_time = None
        self.predicted_target_latency = None
        self.target_lock_fd = os.open("/tmp/cospec_target.lock", os.O_CREAT | os.O_RDWR)
        self.current_batch_size = 0
        self.current_mean_selective_validation_tokens = 0

    def target_start(self):
        torch.cuda.synchronize()
        fcntl.flock(self.target_lock_fd, fcntl.LOCK_EX)

    def target_finish(self):
        torch.cuda.synchronize()
        fcntl.flock(self.target_lock_fd, fcntl.LOCK_UN)
        # Signal the other engine to early exit draft model execution
        # And reset the flag for the current engine 
        self.shm.put(f"early_exit_{not self.is_primary}", True)
        self.shm.put(f"early_exit_{self.is_primary}", False)

    def check_early_exit_draft(self):
        if self.profiler.profiling:
            return False
        
        torch.cuda.synchronize()
        return self.shm.get_nowait(f"early_exit_{self.is_primary}")
    
    def set_current_batch_size(self, batch_size: int):
        self.current_batch_size = batch_size

    def set_current_mean_selective_validation_tokens(self, mean_selective_validation_tokens: float):
        self.current_mean_selective_validation_tokens = mean_selective_validation_tokens

    def predict_colocation_speedup_ratio(self) -> float:
        return self.profiler.predict_colocation_speedup_ratio(self.current_batch_size, 
                                                              self.current_mean_selective_validation_tokens)
