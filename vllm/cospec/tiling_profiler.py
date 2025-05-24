import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from vllm.logger import init_logger
from sklearn.linear_model import LinearRegression
from vllm.spec_decode.util import nvtx_range
import torch

logger = init_logger(__name__)

class TilingProfiler:
    """Class for handling tiling effect profiling."""
    
    def __init__(self, profile_dir: str, max_num_batched_tokens: int):
        self.profile_dir = profile_dir
        self.profile_file = os.path.join(profile_dir, "tiling_results.csv")
        
        # Profiling state
        self.target_model_latencies: Dict[int, List[float]] = {}
        self.target_start_time: Optional[float] = None
        self.run_counts: Dict[int, int] = {}
        self.max_tokens_to_capture = max_num_batched_tokens
        # Only capture multiples of 8
        self.input_tokens_capture_list = [i for i in range(8, self.max_tokens_to_capture + 1, 8)]
        
        self.target_model_latencies_mean: Dict[int, float] = {}
        # Linear regression model
        self.lr_model = None

        self.precomputed_latencies: Optional[List[float]] = None
        self.max_precomputed_tokens: Optional[int] = None
        self.precomputed_latencies_linear: Optional[List[float]] = None
        
    def maybe_load_cached_results(self):
        """Load cached profiling results if they exist and train linear regression model."""
        print("maybe_load_cached_results")
        if not os.path.exists(self.profile_file):
            return False
        print("Loading cached profiling results from ", self.profile_file)
        try:
            with open(self.profile_file, "r") as f:
                # Skip header
                next(f)
                for line in f:
                    num_tokens, mean_latency = line.strip().split(",")
                    num_tokens = int(num_tokens)
                    mean_latency = float(mean_latency)
                    
                    # Initialize with empty list and add the cached mean latency
                    self.target_model_latencies_mean[num_tokens] = mean_latency
                    self.run_counts[num_tokens] = 1
                    
            logger.info(f"Loaded cached profiling results from {self.profile_file}")
            
            self._plot_tiling_effect()
            self._train_linear_regression()
            self._update_precomputed_data()
            
        except Exception as e:
            logger.error(f"Failed to load cached profiling results: {str(e)}")

        return True

    def start_target_marker(self):
        """Start timing the target model"""
        self.target_start_time = time.perf_counter()
    
    def stop_target_marker(self, num_tokens: int):
        """Stop timing the target model and record the latency"""
        assert self.target_start_time is not None, "Target start time is not set" 
            
        target_end_time = time.perf_counter()
        target_duration = target_end_time - self.target_start_time
        # Reset target timing state
        self.target_start_time = None
        # print("num_tokens", num_tokens, "target_duration", target_duration)

        # Round up to nearest multiple of 8
        num_tokens = ((num_tokens - 1) // 8 + 1) * 8

        if num_tokens not in self.input_tokens_capture_list:
            return

        if num_tokens not in self.target_model_latencies:
            self.target_model_latencies[num_tokens] = []
            self.run_counts[num_tokens] = 0
            
        # Increment run counter
        self.run_counts[num_tokens] += 1
        
        # Only sample 10 data points
        if self.run_counts[num_tokens] > 3 and len(self.target_model_latencies[num_tokens]) < 10:
            self.target_model_latencies[num_tokens].append(target_duration)
    
    def _train_linear_regression(self):
        """Train a linear regression model on the available latency data."""
        if len(self.target_model_latencies_mean) < 2:
            logger.warning("Not enough data points to train linear regression model")
            return
            
        # Prepare training data
        X = []
        y = []
        for num_tokens, mean_latency in self.target_model_latencies_mean.items():
            X.append([num_tokens])
            y.append(mean_latency)
                
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.lr_model = LinearRegression()
        self.lr_model.fit(X, y)
        
        logger.info("Trained linear regression model for latency prediction")
        
    def _predict_latency(self, num_tokens: int) -> float:
        """Predict latency using linear regression model."""
        assert self.lr_model is not None, "Linear regression model is not trained"
        # make the num_tokens a multiple of 8
        num_tokens = ((num_tokens - 1) // 8 + 1) * 8
        prediction = self.lr_model.predict([[num_tokens]])[0]
        return prediction 
    
    def save_results(self):
        """Save profiling results and generate visualizations"""
        try:
            os.makedirs(self.profile_dir, exist_ok=True)
            
            # sort the target_model_latencies by num_tokens
            self.target_model_latencies = dict(sorted(self.target_model_latencies.items(), key=lambda x: x[0]))

            # after remove outlier just save the mean latency
            for num_tokens, latencies in self.target_model_latencies.items():
                mean_latency = self._remove_outliers(latencies)
                self.target_model_latencies_mean[num_tokens] = mean_latency

            # Train linear regression model
            self._train_linear_regression()

            # Save target model latencies
            with open(self.profile_file, "w") as f:
                f.write("num_tokens,mean_latency\n")
                for num_tokens, mean_latency in self.target_model_latencies_mean.items():
                    f.write(f"{num_tokens},{mean_latency:.6f}\n")
            
            # Generate tiling effect visualization
            self._plot_tiling_effect()
            
            logger.info(f"Saved profiling results to {self.profile_file}")

            # precompute the latency tensor
            self._update_precomputed_data()
            
        except Exception as e:
            logger.error(f"Failed to write tiling profile results: {str(e)}")
    
    def _update_precomputed_data(self):
        """Update precomputed tensor and max tokens when data changes"""
        sorted_tokens = sorted(self.target_model_latencies_mean.keys())
        self.max_precomputed_tokens = sorted_tokens[-1]
        
        # Get measured latencies up to max_precomputed_tokens
        self.precomputed_latencies = [self.get_target_model_latency(t) for t in range(1, self.max_precomputed_tokens + 1)]
        
        # If we have a trained model, extend predictions for larger token counts
        extended_max_tokens = 2048 * 2
        for t in range(self.max_precomputed_tokens + 1, extended_max_tokens + 1):
            predicted_latency = self._predict_latency(t)
            self.precomputed_latencies.append(predicted_latency)
        self.max_precomputed_tokens = extended_max_tokens
        
        logger.info(f"Precomputed latencies until length {self.max_precomputed_tokens}")

        # precompute the linear latencies
        self.precomputed_latencies_linear = [self._predict_latency(t) for t in range(1, self.max_precomputed_tokens + 1)]

    def _remove_outliers(self, data: List[float]) -> float:
        """Remove outliers using IQR method and return mean of remaining values."""
        if not data:
            return 0.0
            
        data = np.array(data)
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        if len(filtered_data) == 0:
            return np.mean(data)
            
        return np.mean(filtered_data)
    
    def _plot_tiling_effect(self):
        """Plot visualization of tiling effect on target model latency"""
        if not self.target_model_latencies_mean:
            logger.warning("No tiling effect data to plot")
            return
            
        # Get sorted num_tokens and their corresponding mean latencies
        num_tokens = sorted(self.target_model_latencies_mean.keys())
        mean_latencies = [self.target_model_latencies_mean[tokens] for tokens in num_tokens]
        
        # Create line plot
        plt.figure(figsize=(12, 6))
        plt.plot(num_tokens, mean_latencies, 'b-o', linewidth=1, markersize=2)
        
        plt.xlabel('Number of Tokens')
        plt.ylabel('Latency (seconds)')
        plt.title('Target Model Latency vs Number of Tokens')
        
        # Set x-axis ticks and grid at multiples of 8
        max_tokens = max(num_tokens)
        plt.xticks(range(0, max_tokens + 8, 8))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plot_file = os.path.join(self.profile_dir, "tiling_effect_plot.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Saved tiling effect plot to {plot_file}")

    def get_target_model_latency(self, num_tokens: int) -> float:
        """Get the target model latency for a specific number of tokens."""
        num_tokens = ((num_tokens - 1) // 8 + 1) * 8
        return self.target_model_latencies_mean[num_tokens]
    
    def get_target_model_latencies(self, num_tokens: int) -> List[float]:
        """Get the target model latency for a range of token counts."""
        if num_tokens > self.max_precomputed_tokens:
            raise ValueError(f"Requested more tokens than precomputed: {num_tokens} > {self.max_precomputed_tokens}")
        
        return self.precomputed_latencies[:num_tokens]
    
    def get_target_model_latencies_linear(self, num_tokens: int) -> List[float]:
        """Get the target model latency for a range of token counts using linear regression model."""
        if num_tokens > self.max_precomputed_tokens:
            raise ValueError(f"Requested more tokens than precomputed: {num_tokens} > {self.max_precomputed_tokens}")
        
        return self.precomputed_latencies_linear[:num_tokens]
    
