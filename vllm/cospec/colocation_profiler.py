import os
import numpy as np
from typing import Dict, Optional, Tuple, List, Set
from vllm.logger import init_logger
import time
import torch
from vllm.cospec.utils import remove_outliers

logger = init_logger(__name__)

class CustomNonColocationModel:
    """
    Second-order polynomial model for non-colocation step latency.

    Implements:
        T_hat_no_coloc(B, gamma) = gamma * (alpha0 + alpha1*B + alpha2*B^2)
                                    + delta0 + delta1*N_t + delta2*N_t^2
        where N_t = B * (gamma + 1)
    """

    def __init__(self):
        # Coefficients: [alpha0, alpha1, alpha2, delta0, delta1, delta2]
        self.coeffs = None

    def fit(self, X, y):
        # X: array of shape (n, 2) with columns [batch_size (B), num_spec_tokens (gamma)]
        batch_sizes = X[:, 0]
        gamma = X[:, 1]

        N_t = batch_sizes * (gamma + 1.0)

        # Design matrix per formula above
        X_design = np.column_stack([
            gamma,                 # alpha0 term
            gamma * batch_sizes,   # alpha1 term
            gamma * (batch_sizes ** 2),  # alpha2 term
            np.ones_like(batch_sizes),    # delta0 term
            N_t,                   # delta1 term
            N_t ** 2               # delta2 term
        ])

        coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        self.coeffs = coeffs

    def predict(self, X):
        batch_sizes = X[:, 0]
        gamma = X[:, 1]
        N_t = batch_sizes * (gamma + 1.0)

        X_design = np.column_stack([
            gamma,
            gamma * batch_sizes,
            gamma * (batch_sizes ** 2),
            np.ones_like(batch_sizes),
            N_t,
            N_t ** 2,
        ])

        return X_design @ self.coeffs

class CustomColocationModel:
    """
    Second-order polynomial model for colocation step latency.

    Implements:
        T_hat_coloc(B, gamma) = 2 * (beta0 + beta1*N_s + beta2*N_s^2)
                                    * (1 + phi1*B + phi2*B^2)
        where N_s = (B/2) * (gamma + 1)

    For efficient linear least squares, we expand the product and fit a linear
    model over the monomials of N_s and B:
        T = 2 * (c0 + c1*N_s + c2*N_s^2 + c3*B + c4*N_s*B + c5*N_s^2*B
                 + c6*B^2 + c7*N_s*B^2 + c8*N_s^2*B^2)

    This captures the specified structure while keeping fitting linear.
    """

    def __init__(self):
        # Coefficients for expanded monomials: length 9
        self.coeffs = None

    def _build_features(self, batch_sizes: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        # N_s = (B/2) * (gamma + 1)
        N_s = 0.5 * batch_sizes * (gamma + 1.0)

        base = np.column_stack([
            np.ones_like(batch_sizes),   # c0
            N_s,                         # c1
            N_s ** 2,                    # c2
            batch_sizes,                 # c3
            N_s * batch_sizes,           # c4
            (N_s ** 2) * batch_sizes,    # c5
            (batch_sizes ** 2),          # c6
            N_s * (batch_sizes ** 2),    # c7
            (N_s ** 2) * (batch_sizes ** 2),  # c8
        ])
        # Include the leading factor 2 explicitly to preserve formula
        return 2.0 * base

    def fit(self, X, y):
        # X: array of shape (n, 2) with columns [batch_size (B), num_spec_tokens (gamma)]
        batch_sizes = X[:, 0]
        gamma = X[:, 1]

        X_design = self._build_features(batch_sizes, gamma)
        coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        self.coeffs = coeffs

    def predict(self, X):
        batch_sizes = X[:, 0]
        gamma = X[:, 1]
        X_design = self._build_features(batch_sizes, gamma)
        return X_design @ self.coeffs

class ColocationProfiler:
    """Class for handling colocation vs non-colocation profiling."""
    
    def __init__(self, profile_dir: str):
        self.profile_dir = profile_dir
        self.profile_file = os.path.join(profile_dir, "colocation_results.csv")
        
        # Profiling state
        self.profile_results: Dict[Tuple[int, int, bool], List[float]] = {}
        self.run_counts: Dict[Tuple[int, int, bool], int] = {}
        self.current_set: Optional[Dict] = None
        self.start_time: Optional[float] = None
        self.colocation_mode = False
        
        # Regression models
        self.colocation_model: Optional['CustomColocationModel'] = None
        self.non_colocation_model: Optional['CustomNonColocationModel'] = None
        self.test_keys: Optional[List[Tuple[int, int]]] = None
        
        # Warmup settings
        self.warmup_steps = 3
        self.current_step = 0
        
        logger.info(f"Colocation profile file: {self.profile_file}")
    
    def maybe_load_cached_results(self) -> bool:
        """Load cached profiling results if they exist and train regression models."""
        if not os.path.exists(self.profile_file):
            return False
            
        logger.info(f"Loading cached profiling results from {self.profile_file}")
        try:
            with open(self.profile_file, "r") as f:
                # Skip header
                next(f)
                for line in f:
                    batch_size, num_spec_tokens, colocation_mode, mean_step_time = line.strip().split(",")
                    batch_size = int(batch_size)
                    num_spec_tokens = int(num_spec_tokens)
                    colocation_mode = colocation_mode.lower() == 'true'
                    mean_step_time = float(mean_step_time)
                    
                    # Create key for the configuration
                    key = (batch_size, num_spec_tokens, colocation_mode)
                    
                    # Initialize with empty list and add the cached mean latency
                    self.profile_results[key] = [mean_step_time]
                    
            logger.info(f"Loaded cached profiling results from {self.profile_file}")
            
            # Train regression models and generate plots
            self._train_regression_models()
            metrics = self._calculate_model_metrics()
            self._plot_speedup_heatmap()
            self._plot_regression_heatmap()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cached profiling results: {str(e)}")
            return False

    def set_colocation_mode(self, colocation_mode: bool):
        """Set the colocation mode for subsequent profiling"""
        self.colocation_mode = colocation_mode
    
    def set_profile_batch_size(self, batch_size: int):
        """Set the batch size for subsequent profiling"""
        self.profile_batch_size = batch_size

    def start_step_marker(self, num_speculative_tokens: int):
        """Start timing a step"""
        torch.cuda.synchronize()
        
        # Start a new timing set
        self.current_set = {
            'batch_size': self.profile_batch_size,
            'num_speculative_tokens': num_speculative_tokens,
            'step_time': None,
            'colocation_mode': self.colocation_mode,
        }
        self.start_time = time.perf_counter()
    
    def stop_step_marker(self):
        """Stop timing a step and record results"""
        if self.current_set is None or self.start_time is None:
            return
            
        torch.cuda.synchronize()
        
        duration = time.perf_counter() - self.start_time
        self.current_set['step_time'] = duration
        
        # Create key for the current configuration
        key = (
            self.current_set['batch_size'],
            self.current_set['num_speculative_tokens'],
            self.current_set['colocation_mode']
        )
        
        # Initialize tracking for this configuration if it doesn't exist
        if key not in self.profile_results:
            self.profile_results[key] = []
            self.run_counts[key] = 0
            
        # Increment run counter
        self.run_counts[key] += 1
        
        if self.run_counts[key] > 5 and self.run_counts[key] <= 15:
            self.profile_results[key].append(duration)
    
        # Reset current timing state
        self.current_set = None
        self.start_time = None
    
    def save_results(self):
        """Save profiling results and generate visualizations"""
            
        try:
            os.makedirs(self.profile_dir, exist_ok=True)
            
            # Save step times
            with open(self.profile_file, "a") as f:
                if os.stat(self.profile_file).st_size == 0:
                    f.write("batch_size,num_speculative_tokens,colocation_mode,mean_step_time\n")

                if not self.profile_results:
                    logger.warning("No profile results to write")
                    return
                    
                for (batch_size, num_spec_tokens, colocation_mode), step_times in self.profile_results.items():
                    mean_time = remove_outliers(step_times)
                    f.write(f"{batch_size},{num_spec_tokens},{colocation_mode},{mean_time:.6f}\n")
            
            # Train regression models
            self._train_regression_models()
            
            # Calculate metrics (needed for plots)
            metrics = self._calculate_model_metrics()
            
            # Generate plots
            self._plot_speedup_heatmap()
            self._plot_regression_heatmap()
            
        except Exception as e:
            logger.error(f"Failed to write colocation profile results: {str(e)}")
    
    def _get_unique_configurations(self) -> List[Tuple[int, int]]:
        """Get list of unique (batch_size, num_spec_tokens) configurations."""
        return sorted(list(set((bs, ns) for bs, ns, _ in self.profile_results.keys())))
    
    def _group_results_by_configuration(self) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
        """Group profiling results by configuration."""
        results_dict = {}
        for (batch_size, num_spec_tokens, colocation_mode), step_times in self.profile_results.items():
            key = (batch_size, num_spec_tokens)
            if key not in results_dict:
                results_dict[key] = {'colocation': [], 'non_colocation': []}
            
            mean_time = remove_outliers(step_times)
            if colocation_mode:
                results_dict[key]['colocation'].append(mean_time)
            else:
                results_dict[key]['non_colocation'].append(mean_time)
                
        return results_dict
    
    def _prepare_training_data(self, train_keys: Set[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for regression models."""
        X_colocation = []
        y_colocation = []
        X_non_colocation = []
        y_non_colocation = []
        
        for (batch_size, num_spec_tokens, colocation_mode), step_times in self.profile_results.items():
            config_key = (batch_size, num_spec_tokens)
            if config_key not in train_keys or not step_times:
                continue
                
            mean_time = remove_outliers(step_times)
            features = [batch_size, num_spec_tokens]
            
            if colocation_mode:
                X_colocation.append(features)
                y_colocation.append(mean_time)
            else:
                X_non_colocation.append(features)
                y_non_colocation.append(mean_time)
                
        return (np.array(X_colocation), np.array(y_colocation),
                np.array(X_non_colocation), np.array(y_non_colocation))
    
    def _train_regression_models(self) -> None:
        """Train regression models using train/test split based on configurations."""
        from sklearn.model_selection import train_test_split
        if not self.profile_results:
            logger.warning("No profile results to train models")
            return

        # Get unique configurations and split into train/test
        unique_keys = self._get_unique_configurations()
        if len(unique_keys) < 5:
            logger.warning(f"Only {len(unique_keys)} unique configurations found. Training on all data.")
            train_keys = set(unique_keys)
            self.test_keys = []
        else:
            train_keys, self.test_keys = train_test_split(unique_keys, test_size=0.2, random_state=42)
            train_keys = set(train_keys)
            logger.info(f"Split data into {len(train_keys)} training and {len(self.test_keys)} test configurations.")

        # Prepare training data
        X_colocation, y_colocation, X_non_colocation, y_non_colocation = self._prepare_training_data(train_keys)
        
        if len(X_colocation) == 0 or len(X_non_colocation) == 0:
            logger.error("Insufficient training data for one or both models")
            self.colocation_model = None
            self.non_colocation_model = None
            return
            
        # Train colocation model
        if len(X_colocation) > 0:
            self.colocation_model = CustomColocationModel()
            self.colocation_model.fit(X_colocation, y_colocation)
            logger.info(f"Trained colocation model on {len(X_colocation)} data points")
            
        # Train non-colocation model
        if len(X_non_colocation) > 0:
            self.non_colocation_model = CustomNonColocationModel()
            self.non_colocation_model.fit(X_non_colocation, y_non_colocation)
            logger.info(f"Trained non-colocation model on {len(X_non_colocation)} data points")
    
    def predict_colocation_speedup_ratio(self, batch_size: int, num_spec_tokens: int) -> float:
        """Predict the speedup ratio between non-colocation and colocation modes."""
        assert self.colocation_model is not None and self.non_colocation_model is not None, "Models not trained"
        
        # Prepare input features
        X = np.array([[batch_size, num_spec_tokens]])
        
        # Make predictions
        colocation_time = float(self.colocation_model.predict(X)[0])
        non_colocation_time = float(self.non_colocation_model.predict(X)[0])
        # Numerical stability: clamp to epsilon to avoid non-physical or zero division
        eps = 1e-8
        colocation_time = max(colocation_time, eps)
        non_colocation_time = max(non_colocation_time, eps)
        
        # Calculate ratio (non-colocation / colocation)
        ratio = non_colocation_time / colocation_time

        return ratio
    
    def _compute_metrics(self,
                         actual_ratios: np.ndarray,
                         predicted_ratios: np.ndarray) -> Dict:
        """Compute regression metrics: R² and MAPE only."""

        errors = predicted_ratios - actual_ratios
        # R² with zero-variance guard
        denom = np.sum((actual_ratios - np.mean(actual_ratios)) ** 2)
        r2 = float(1 - np.sum((actual_ratios - predicted_ratios) ** 2) / denom) if denom > 0 else 0.0

        # MAPE with epsilon for stability
        eps = 1e-8
        mape = float(np.mean(np.abs(errors) / (np.abs(actual_ratios) + eps)))

        return {
            'R²': r2,
            'MAPE': mape,
        }
    
    def _calculate_model_metrics(self) -> Dict:
        """Calculate evaluation metrics for the regression model on the test set."""
        if not self.profile_results or self.colocation_model is None or self.non_colocation_model is None or self.test_keys is None:
            logger.warning("Cannot calculate metrics: Models not trained or test set not defined")
            return {}
            
        # Prepare test data
        actual_ratios = []
        predicted_ratios = []
        
        # Group results by configuration
        results_dict = self._group_results_by_configuration()
        
        # Calculate metrics on test set
        for batch_size, num_spec_tokens in self.test_keys:
            key = (batch_size, num_spec_tokens)
            if key not in results_dict:
                continue
                
            colocation_times = results_dict[key]['colocation']
            non_colocation_times = results_dict[key]['non_colocation']
            
            if colocation_times and non_colocation_times:
                avg_colocation = np.mean(colocation_times)
                avg_non_colocation = np.mean(non_colocation_times)
                
                if avg_colocation > 0:
                    actual_ratio = avg_non_colocation / avg_colocation
                    predicted_ratio = self.predict_colocation_speedup_ratio(batch_size, num_spec_tokens)
                    
                    actual_ratios.append(actual_ratio)
                    predicted_ratios.append(predicted_ratio)
                    
        if not actual_ratios:
            return {}
            
        # Calculate metrics
        actual_ratios = np.array(actual_ratios)
        predicted_ratios = np.array(predicted_ratios)
        
        metrics = self._compute_metrics(actual_ratios,
                                        predicted_ratios)
        return metrics
    
    def _plot_speedup_heatmap(self):
        """Plot heatmap of speedup ratio between colocation and non-colocation modes"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        if not self.profile_results:
            logger.warning("No profile results to plot")
            return
            
        # Group results by batch_size and num_speculative_tokens
        results_dict = {}
        for (batch_size, num_spec_tokens, colocation_mode), step_times in self.profile_results.items():
            key = (batch_size, num_spec_tokens)
            if key not in results_dict:
                results_dict[key] = {'colocation': [], 'non_colocation': []}
            
            step_times = remove_outliers(step_times)

            if colocation_mode:
                results_dict[key]['colocation'].append(np.mean(step_times))
            else:
                results_dict[key]['non_colocation'].append(np.mean(step_times))
        
        # Get unique batch sizes and spec token numbers
        batch_sizes = sorted(set(k[0] for k in results_dict.keys()))
        spec_tokens = sorted(set(k[1] for k in results_dict.keys()), reverse=True)
        
        # Create speedup matrix
        speedup_matrix = np.zeros((len(spec_tokens), len(batch_sizes)))
        
        for i, num_spec_tokens in enumerate(spec_tokens):
            for j, batch_size in enumerate(batch_sizes):
                key = (batch_size, num_spec_tokens)
                if key in results_dict:
                    colocation_times = results_dict[key]['colocation']
                    non_colocation_times = results_dict[key]['non_colocation']
                    
                    if colocation_times and non_colocation_times:
                        avg_colocation = np.mean(colocation_times)
                        avg_non_colocation = np.mean(non_colocation_times)
                        speedup_matrix[i, j] = avg_non_colocation / avg_colocation
        
        # Create heatmap
        plt.figure(figsize=(7, 3))
        sns.heatmap(speedup_matrix, 
                   xticklabels=batch_sizes,
                   yticklabels=spec_tokens,
                   cmap='YlGnBu',
                   center=1.0, 
                   annot=True,
                   fmt='.2f',  
                   cbar_kws={'label': 'Speedup Ratio (Non-colocation / Colocation)\n>1: Colocation faster\n<1: Non-colocation faster'})
        
        # Add contour line at 1.0
        plt.contour(speedup_matrix, levels=[1.0], colors='red', linewidths=2)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Speculative Window Size')
        plt.title('Speedup Ratio Heatmap\nValues > 1: Colocation is faster\nValues < 1: Non-colocation is faster')
        
        # Save plot
        plot_file = os.path.join(self.profile_dir, "speedup_heatmap.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Saved speedup heatmap to {plot_file}")
    
    def _plot_regression_heatmap(self):
        """Plot heatmap of predicted speedup ratio using regression models"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        if self.colocation_model is None or self.non_colocation_model is None:
            logger.warning("Regression models not trained")
            return
            
        # Get unique batch sizes and spec token numbers from profile results
        batch_sizes = sorted(set(k[0] for k in self.profile_results.keys()))
        spec_tokens = sorted(set(k[1] for k in self.profile_results.keys()), reverse=True)
        
        # Create prediction matrix
        speedup_matrix = np.zeros((len(spec_tokens), len(batch_sizes)))
        
        for i, num_spec_tokens in enumerate(spec_tokens):
            for j, batch_size in enumerate(batch_sizes):
                ratio = self.predict_colocation_speedup_ratio(batch_size, num_spec_tokens)
                speedup_matrix[i, j] = ratio
        
        # Calculate model metrics
        metrics = self._calculate_model_metrics()
        metrics_to_display = metrics
        metrics_text = "\n".join([f"{k}: {v:.3f}" if v is not None else f"{k}: N/A" for k, v in metrics_to_display.items()])
        
        # Create heatmap
        plt.figure(figsize=(20, 8))
        sns.heatmap(speedup_matrix, 
                   xticklabels=batch_sizes,
                   yticklabels=spec_tokens,
                   cmap='RdYlGn',
                   center=1.0,
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Predicted Speedup Ratio (Non-colocation / Colocation)\n>1: Colocation faster\n<1: Non-colocation faster'})
        
        plt.xlabel('Batch Size')
        plt.ylabel('Number of Speculative Tokens')
        plt.title('Predicted Speedup Ratio Heatmap (Regression Model)\nValues > 1: Colocation is faster\nValues < 1: Non-colocation is faster')
        
        # Add metrics text box
        plt.text(1.05, 0.5, f"Model Metrics:\n{metrics_text}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='center',
                horizontalalignment='left')

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # Save plot
        plot_file = os.path.join(self.profile_dir, "predicted_speedup_heatmap.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Saved predicted speedup heatmap to {plot_file}")
        logger.info(f"Model metrics: {metrics_to_display}")
        