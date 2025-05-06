import torch
import time
import os
import fcntl
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque

from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.cospec.shm import SharedMemory
from vllm.cospec.profiler import Profiler
from vllm.cospec.selective_validator import SelectiveValidator

logger = init_logger(__name__)

class CospecManager:
    def __init__(self, vllm_config: VllmConfig):
        self.shm = SharedMemory()
        self.profiler = Profiler(vllm_config)
        self.selective_validator = SelectiveValidator()
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

    def selective_validation(self, proposals):
        """Perform selective validation on proposals.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            
        Returns:
            Tuple of (filtered_proposals, acceptance_probs) where:
            - filtered_proposals: Proposals with acceptance probability >= threshold
            - acceptance_probs: Predicted acceptance probabilities for all proposals
        """
        torch.cuda.nvtx.range_push("selective_validation")
        filtered_proposals = self.selective_validator.selective_validation(proposals)
        torch.cuda.nvtx.range_pop()
        return filtered_proposals

    def update_proposal_history(self, proposals, proposal_scores):
        """Update the history of proposal acceptance data.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            proposal_scores: Tensor containing the actual acceptance scores
        """
        torch.cuda.nvtx.range_push("update_proposal_history")
        self.selective_validator.update_proposal_history(proposals, proposal_scores)
        torch.cuda.nvtx.range_pop()