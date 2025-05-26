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

        self.target_lock_fd = os.open(f"/tmp/cospec_target.lock", os.O_CREAT | os.O_RDWR)
        self.draft_lock_fd = os.open(f"/tmp/cospec_draft.lock", os.O_CREAT | os.O_RDWR)
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
    
    def predict_colocation_speedup_ratio(self, total_requests: int) -> float:
        if not self.is_selective_validator_trained():
            return 1 
        
        if self.is_driver:
            torch.cuda.nvtx.range_push("predict_colocation_speedup_ratio")
            speedup_ratio = self.profiler.predict_colocation_speedup_ratio(total_requests, 
                                                                            self.selective_validator.moving_avg_mean_tokens) 
            torch.cuda.nvtx.range_pop()
            return speedup_ratio
        return 1 
    
    def set_colocation_mode(self, colocation_mode: bool):
        if self.is_driver and self.is_primary:
            self.profiler.set_colocation_mode(colocation_mode)

    def set_profile_batch_size(self, batch_size: int):
        if self.is_driver and self.is_primary:
            self.profiler.set_profile_batch_size(batch_size)

    def start_step_marker(self, num_speculative_tokens:int):
        if self.is_driver and self.is_primary:
            self.profiler.start_step_marker(num_speculative_tokens)

    def stop_step_marker(self):
        if self.is_driver and self.is_primary:
            self.profiler.stop_step_marker()

    def target_start(self):
        if self.is_driver:
            torch.cuda.synchronize()
            fcntl.flock(self.target_lock_fd, fcntl.LOCK_EX)
            self.profiler.start_target_marker()

    def target_finish(self, num_tokens: int):
        if self.is_driver:
            # print("target_num_tokens, ", num_tokens)
            torch.cuda.synchronize()
            fcntl.flock(self.target_lock_fd, fcntl.LOCK_UN)
            self.profiler.stop_target_marker(num_tokens)
            # Signal the other engine to early exit draft model execution
            # And reset the flag for the current engine 
            self.shm.put(f"early_exit_{not self.is_primary}", True)
            self.shm.put(f"early_exit_{self.is_primary}", False)
    
    def draft_start(self):
        if self.is_driver:
            torch.cuda.synchronize()
            fcntl.flock(self.draft_lock_fd, fcntl.LOCK_EX)

    def draft_finish(self):
        if self.is_driver:
            torch.cuda.synchronize()
            fcntl.flock(self.draft_lock_fd, fcntl.LOCK_UN)

    def check_early_exit_draft(self):
        if self.profiler.is_profiling():
            return False

        if self.is_driver:
            torch.cuda.synchronize()
            should_exit = self.shm.get_nowait(f"early_exit_{self.is_primary}")
            for rank in range(1, self.total_ranks):
                self.shm.put(f"early_exit_{self.is_primary}_{rank}", should_exit)
        else:
            # wait for driver to set the flag 
            self.shm.wait_for_exists(f"early_exit_{self.is_primary}_{self.rank}")
            should_exit = self.shm.get_and_delete(f"early_exit_{self.is_primary}_{self.rank}")

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
            # torch.cuda.nvtx.range_push("selective_validation")
            # start_time = time.perf_counter()
            filtered_proposals = self.selective_validator.selective_validation(proposals, total_non_proposal_tokens)
            # end_time = time.perf_counter()
            # print(f"selective_validation time in ms {((end_time - start_time) * 1000):.2f}")
            # torch.cuda.nvtx.range_pop()
            return filtered_proposals
        else:
            return proposals

    def update_proposal_history(self, proposals, proposal_scores):
        """Update the history of proposal acceptance data.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            proposal_scores: Tensor containing the actual acceptance scores
        """
        if self.profiler.profiling:
            return

        if self.is_driver:
            torch.cuda.nvtx.range_push("update_proposal_history")
            self.selective_validator.update_proposal_history(proposals, proposal_scores)
            torch.cuda.nvtx.range_pop()
