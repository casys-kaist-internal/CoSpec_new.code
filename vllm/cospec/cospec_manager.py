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
        self.profiler = Profiler(vllm_config)
        self.selective_validator = SelectiveValidator()
        self.rank = vllm_config.parallel_config.rank
        self.is_driver = self.rank == 0
        self.is_primary = vllm_config.speculative_config.is_primary
        self.start_time = None
        self.predicted_target_latency = None
        self.target_lock_fd = os.open(f"/tmp/cospec_target.lock", os.O_CREAT | os.O_RDWR)
        self.draft_lock_fd = os.open(f"/tmp/cospec_draft.lock", os.O_CREAT | os.O_RDWR)
        self.current_batch_size = 0
        self.shm.put(f"early_exit_{not self.is_primary}", False)
        self.shm.put(f"early_exit_{self.is_primary}", False)
        self.total_ranks = vllm_config.parallel_config.world_size
        
        # Data collection for plotting
        self.input_shapes = []
        self.durations = []

    def target_start(self, model_input):
        if self.is_driver:
            torch.cuda.synchronize()
            fcntl.flock(self.target_lock_fd, fcntl.LOCK_EX)
            self.start_time = time.perf_counter()
            input_shape = model_input.input_tokens.shape
            # Store data for plotting
            self.input_shapes.append(input_shape[0]) 

    def target_finish(self):
        if self.is_driver:
            torch.cuda.synchronize()
            fcntl.flock(self.target_lock_fd, fcntl.LOCK_UN)
            end_time = time.perf_counter()
            self.durations.append(end_time - self.start_time)
            # Signal the other engine to early exit draft model execution
            # And reset the flag for the current engine 
            self.shm.put(f"early_exit_{not self.is_primary}", True)
            self.shm.put(f"early_exit_{self.is_primary}", False)
            assert len(self.durations) == len(self.input_shapes)
            print(f"len(self.durations) % 1000: {len(self.durations) % 1000}")
            if len(self.durations) % 1000 == 0:
                self.plot_shape_vs_duration()

    
    def draft_start(self):
        if self.is_driver:
            torch.cuda.synchronize()
            fcntl.flock(self.draft_lock_fd, fcntl.LOCK_EX)

    def draft_finish(self):
        if self.is_driver:
            torch.cuda.synchronize()
            fcntl.flock(self.draft_lock_fd, fcntl.LOCK_UN)

    def check_early_exit_draft(self):
        if self.profiler.profiling:
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
    
    def set_current_batch_size(self, batch_size: int):
        self.current_batch_size = batch_size

    def predict_colocation_speedup_ratio(self, total_requests: int) -> float:
        return self.profiler.predict_colocation_speedup_ratio(total_requests, 
                                                              self.selective_validator.moving_avg_mean_tokens)

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
    
    def selective_validation_correctness_test(self, proposals):
        """Perform random drop for correctness testing purpose"""
        logger.info("Random drop for selective validation correctness test")
        filtered_proposals = self.selective_validator.random_drop(proposals)
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

    def plot_shape_vs_duration(self):
        """Plot the graph of input shape vs duration.
        
        Args:
            save_path: Optional path to save the plot. If None, plot will be displayed.
        """
        if not self.is_driver or not self.input_shapes:
            return
            
        plt.figure(figsize=(10, 6))
        plt.scatter(self.input_shapes, self.durations, alpha=0.5)
        plt.plot(self.input_shapes, self.durations, 'r--', alpha=0.3)
        
        plt.xlabel('Input Shape')
        plt.ylabel('Duration (seconds)')
        plt.title('Input Shape vs Duration')
        plt.grid(True, alpha=0.3)
        print("saved input_tokens_duration.png")
        plt.savefig(f"input_tokens_duration.png")
        plt.close()
