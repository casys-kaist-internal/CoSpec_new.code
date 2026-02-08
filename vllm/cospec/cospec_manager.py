import torch
import os
import fcntl
import time
import matplotlib.pyplot as plt
import numpy as np

from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.cospec.shm import SharedMemory
from vllm.cospec.profiler import Profiler
from vllm.cospec.selective_validator import SelectiveValidator

logger = init_logger(__name__)

class CospecManager:
    def __init__(self, vllm_config: VllmConfig):
        self.shm = SharedMemory()
        self.rank = vllm_config.parallel_config.rank
        self.is_primary = vllm_config.speculative_config.is_primary
        self.is_driver = vllm_config.parallel_config.rank == 0
        self.total_ranks = vllm_config.parallel_config.world_size
        self.current_batch_size = 0

        # Create rank-specific lock files
        self.target_lock_fd = os.open(f"/tmp/cospec_target_rank_{self.rank}.lock", os.O_CREAT | os.O_RDWR)
        self.draft_lock_fd = os.open(f"/tmp/cospec_draft_rank_{self.rank}.lock", os.O_CREAT | os.O_RDWR)
        self.shm.put(f"early_exit_{not self.is_primary}", False)
        self.shm.put(f"early_exit_{self.is_primary}", False)

        self.profiler = Profiler(vllm_config)
        self.selective_validator = SelectiveValidator(profiler=self.profiler)

    def start_profile(self, mode:str):
        self.profiler.start_profile(mode)

    def stop_profile(self):
        self.profiler.stop_profile()

    def is_profiling(self):
        return self.profiler.is_profiling()

    def maybe_load_cached_colocation_profile(self) -> bool:
        if self.is_driver:
            return self.profiler.maybe_load_cached_colocation_profile()
        return True

    def maybe_load_cached_tiling_profile(self) -> bool:
        if self.is_driver:
            return self.profiler.maybe_load_cached_tiling_profile()
        return True 
    
    def is_selective_validator_trained(self) -> bool:
        if self.is_driver:
            return self.selective_validator.is_selective_validator_trained()
        return True
    
    def predict_colocation_speedup_ratio(self, batch_size: int) -> float:        
        if self.is_driver:
            speedup_ratio = self.profiler.predict_colocation_speedup_ratio(batch_size,
                                                                            self.get_num_speculative_tokens_ema())
            return speedup_ratio
        return 1 
    
    def set_colocation_mode(self, colocation_mode: bool):
        self.profiler.set_colocation_mode(colocation_mode)

    def set_profile_batch_size(self, batch_size: int):
        self.profiler.set_profile_batch_size(batch_size)

    def start_step_marker(self, num_speculative_tokens:int):
        torch.cuda.synchronize()
        if self.is_driver and self.is_primary:
            self.profiler.start_step_marker(num_speculative_tokens)

    def stop_step_marker(self):
        torch.cuda.synchronize()
        if self.is_driver and self.is_primary:
            self.profiler.stop_step_marker()

    def target_start(self):
        torch.cuda.synchronize()
        if self.is_driver:
            # Driver takes exclusive lock first.
            fcntl.flock(self.target_lock_fd, fcntl.LOCK_EX)
            # Publish the driver's group (True if primary, False otherwise).
            self.shm.put("target_lock_acquired", self.is_primary)
            torch.cuda.nvtx.range_push("target_start")
            self.profiler.start_target_marker()
        else:
            # Busy-wait until the driver's group matches this process's group.
            while self.shm.get("target_lock_acquired") != self.is_primary:
                time.sleep(0.001)
            # At this point, the driver is in the same group, so acquire the lock.
            fcntl.flock(self.target_lock_fd, fcntl.LOCK_EX)

    def target_finish(self, num_tokens: int):
        torch.cuda.synchronize()
        # Before releasing the lock, initialize the target_lock_acquired flag.
        if self.is_driver:            
            self.shm.delete("target_lock_acquired")
            torch.cuda.nvtx.range_pop()
            self.profiler.stop_target_marker(num_tokens)
            # Signal the other engine to early exit draft model execution
            # And reset the flag for the current engine 
            self.shm.put(f"early_exit_{not self.is_primary}", True)
            self.shm.put(f"early_exit_{self.is_primary}", False)

        fcntl.flock(self.target_lock_fd, fcntl.LOCK_UN)
        
    def draft_start(self):
        torch.cuda.synchronize()
        if self.is_driver:
            fcntl.flock(self.draft_lock_fd, fcntl.LOCK_EX)
            self.shm.put("draft_lock_acquired", self.is_primary)
            torch.cuda.nvtx.range_push("draft_start")
        else:
            # Busy-wait until the driver's group matches this process's group.
            while self.shm.get("draft_lock_acquired") != self.is_primary:
                time.sleep(0.001)
            fcntl.flock(self.draft_lock_fd, fcntl.LOCK_EX)

    def draft_finish(self):
        torch.cuda.synchronize()
        if self.is_driver:
            self.shm.delete("draft_lock_acquired")
            torch.cuda.nvtx.range_pop()
        fcntl.flock(self.draft_lock_fd, fcntl.LOCK_UN)

    def check_early_exit_draft(self):        
        torch.cuda.synchronize()
        if self.is_driver:
            should_exit = self.shm.get(f"early_exit_{self.is_primary}")
            for rank in range(1, self.total_ranks):
                self.shm.put(f"early_exit_{self.is_primary}_{rank}", should_exit)
        else:
            # wait for driver to set the flag 
            self.shm.wait_for_exists(f"early_exit_{self.is_primary}_{self.rank}")
            should_exit = self.shm.get(f"early_exit_{self.is_primary}_{self.rank}")
            self.shm.delete(f"early_exit_{self.is_primary}_{self.rank}")

        return should_exit

    def selective_validation(self, proposals, total_non_proposal_tokens: int):
        """Perform selective validation on proposals.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            
        Returns:
            Tuple of (filtered_proposals, acceptance_probs) where:
            - filtered_proposals: Proposals with acceptance probability >= threshold
            - acceptance_probs: Predicted acceptance probabilities for all proposals
        """
        if self.profiler.is_profiling():
            return proposals

        if self.is_driver:
            filtered_proposals = self.selective_validator.selective_validation(proposals, total_non_proposal_tokens)
            return filtered_proposals
        else:
            return proposals

    def update_proposal_history(self, proposals, proposal_scores):
        """Update the history of proposal acceptance data.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            proposal_scores: Tensor containing the actual acceptance scores
        """
        if self.profiler.is_profiling():
            return

        if self.is_driver:
            self.selective_validator.update_proposal_history(proposals, proposal_scores)

    def get_num_speculative_tokens_ema(self) -> int:
        return self.selective_validator.get_mean_selective_validation_tokens_ema()
        