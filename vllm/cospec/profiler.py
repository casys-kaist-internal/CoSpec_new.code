import torch
import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List, Set
from vllm.logger import init_logger
from vllm.config import VllmConfig

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from .colocation_profiler import ColocationProfiler
from .tiling_profiler import TilingProfiler

logger = init_logger(__name__)

class Profiler:
    """
    Profiler for 
    1) measuring and analyzing performance of colocation vs non-colocation modes.
    2) measuring the tiling effect of the target model.
    
    This class handles:
    1. Profiling step times for different configurations
    2. Training regression models to predict performance
    3. Generating visualizations and metrics
    4. Managing train/test splits for model evaluation
    """
    
    def __init__(self, vllm_config: VllmConfig):
        """Initialize the profiler with configuration settings.
        
        Args:
            vllm_config: Configuration object containing model and GPU settings
        """
        # Setup paths and names
        self.gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
        self.target_model = vllm_config.model_config.model.split("/")[-1]
        self.draft_model = vllm_config.speculative_config.model.split("/")[-1]
        self.profile_dir = os.path.join("profile", f"{self.gpu_name}_{self.target_model}_{self.draft_model}")

        # Only profile for in the first rank for driver
        self.is_primary = vllm_config.speculative_config.is_primary
        self.is_driver = vllm_config.parallel_config.rank == 0

        # Initialize profiling state
        self.profiling = False
        self.profile_mode = None
        
        # Initialize inner profilers
        self.colocation_profiler = ColocationProfiler(self.profile_dir)
        self.tiling_profiler = TilingProfiler(self.profile_dir, 
                                              vllm_config.scheduler_config.max_num_batched_tokens)
        
        logger.info(f"Profile directory: {self.profile_dir}")

    def start_profile(self, mode: str = 'colocation'):
        """Start profiling in the specified mode.
        
        Args:
            mode: Either 'colocation' for colocation comparison or 'tiling' for tiling effect analysis
        """
        if mode not in ['colocation', 'tiling']:
            logger.error(f"Invalid profiling mode: {mode}. Must be either 'colocation' or 'tiling'")
            return
            
        logger.info(f"Starting cospec profiler in {mode} mode")
        self.profiling = True
        self.profile_mode = mode

    def stop_profile(self):
        """Stop profiling and save results"""
        assert self.profiling, "Profiler is not running, but stop() was called"
            
        logger.info(f"Stopping cospec profiler in {self.profile_mode} mode")
        self.profiling = False

        if self.is_primary and self.is_driver:
            if self.profile_mode == 'colocation':
                self.colocation_profiler.save_results()
            else:  # tiling mode
                self.tiling_profiler.save_results()

    def is_profiling(self):
        return self.profiling

    def maybe_load_cached_colocation_profile(self):
        return self.colocation_profiler.maybe_load_cached_results()

    def maybe_load_cached_tiling_profile(self):
        return self.tiling_profiler.maybe_load_cached_results()

    def start_step_marker(self, num_speculative_tokens: int):
        """Start timing a step"""
        if not self.profiling or self.profile_mode != 'colocation':
            return
            
        self.colocation_profiler.start_step_marker(num_speculative_tokens)

    def stop_step_marker(self):
        """Stop timing a step and record results"""
        if not self.profiling or self.profile_mode != 'colocation':
            return
            
        self.colocation_profiler.stop_step_marker()

    def start_target_marker(self):
        """Start timing the target model"""
        if not self.profiling or self.profile_mode != 'tiling':
            return
            
        self.tiling_profiler.start_target_marker()

    def stop_target_marker(self, num_tokens: int):
        """Stop timing the target model and record the latency"""
        if not self.profiling or self.profile_mode != 'tiling':
            return
            
        self.tiling_profiler.stop_target_marker(num_tokens)

    def set_colocation_mode(self, colocation_mode: bool):
        """Set the colocation mode for subsequent profiling"""
        if not self.profiling or self.profile_mode != 'colocation':
            return
        
        self.colocation_profiler.set_colocation_mode(colocation_mode)

    def set_profile_batch_size(self, batch_size: int):
        """Set the batch size for subsequent profiling"""
        if not self.profiling or self.profile_mode != 'colocation':
            return
        
        self.colocation_profiler.set_profile_batch_size(batch_size)

    def predict_colocation_speedup_ratio(self, batch_size: int, num_spec_tokens: int) -> float:
        """Predict the speedup ratio between non-colocation and colocation modes."""
        return self.colocation_profiler.predict_colocation_speedup_ratio(batch_size, num_spec_tokens)

    def get_target_model_latency(self, num_tokens: int) -> float:
        """Get the target model latency for a given number of tokens
        This should consider the tiling effect of the target model.
        Args:
            num_tokens: Number of tokens to get latency for
            
        Returns:
            Target model latency in milliseconds
        """
        return self.tiling_profiler.get_target_model_latency(num_tokens)
    
    def get_target_model_latencies(self, num_tokens: int) -> List[float]:
        """Get the target model latency for a range of token counts."""
        return self.tiling_profiler.get_target_model_latencies(num_tokens)
    
    def get_target_model_latencies_linear(self, num_tokens: int) -> List[float]:
        """Get the target model latency for a range of token counts using linear regression model."""
        return self.tiling_profiler.get_target_model_latencies_linear(num_tokens)
    
    def get_target_model_latencies_polynomial(self, num_tokens: int) -> List[float]:
        """Get the target model latency for a range of token counts using polynomial regression model."""
        return self.tiling_profiler.get_target_model_latencies_polynomial(num_tokens)