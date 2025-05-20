import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List, Set
from vllm.logger import init_logger
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import time
import torch
from vllm.spec_decode.util import nvtx_range

logger = init_logger(__name__)

class ColocationProfiler:
    """Class for handling colocation vs non-colocation profiling."""
    
    def __init__(self, profile_dir: str):
        self.profile_dir = profile_dir
        self.profile_file = os.path.join(profile_dir, "results.csv")
        
        # Profiling state
        self.profile_results: Dict[Tuple[int, int, bool], List[float]] = {}
        self.run_counts: Dict[Tuple[int, int, bool], int] = {}
        self.current_set: Optional[Dict] = None
        self.start_time: Optional[float] = None
        
        # Regression models
        self.colocation_model: Optional[LinearRegression] = None
        self.non_colocation_model: Optional[LinearRegression] = None
        self.poly = PolynomialFeatures(degree=2)
        self.test_keys: Optional[List[Tuple[int, int]]] = None
        
        # Warmup settings
        self.warmup_steps = 3
        self.current_step = 0
        
        logger.info(f"Colocation profile file: {self.profile_file}")
    
    def start_step_marker(self, batch_size: int, num_speculative_tokens: int, colocation_mode: bool):
        """Start timing a step"""
        if colocation_mode:
            batch_size = batch_size * 2
        
        torch.cuda.synchronize()
        
        # Start a new timing set
        self.current_set = {
            'batch_size': batch_size,
            'num_speculative_tokens': num_speculative_tokens,
            'step_time': None,
            'colocation_mode': colocation_mode,
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
        
        if self.run_counts[key] > 3 and self.run_counts[key] <= 13:
            self.profile_results[key].append(duration)
    
        # Reset current timing state
        self.current_set = None
        self.start_time = None
    
    def save_results(self):
        """Save profiling results and generate visualizations"""
        if not self.is_primary:
            return
            
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
                    mean_time = self._remove_outliers(step_times)
                    f.write(f"{batch_size},{num_spec_tokens},{colocation_mode},{mean_time:.6f}\n")
            
            # Train regression models
            self._train_regression_models()
            
            # Calculate metrics (needed for plots)
            metrics = self._calculate_model_metrics()
            
            # Generate plots
            self._plot_speedup_heatmap()
            self._plot_regression_heatmap()
            self._plot_roc_curve(metrics.get('fpr'), metrics.get('tpr'), metrics.get('AUROC'))
            
        except Exception as e:
            logger.error(f"Failed to write colocation profile results: {str(e)}")
    
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
            
            mean_time = self._remove_outliers(step_times)
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
                
            mean_time = self._remove_outliers(step_times)
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
            X_colocation_poly = self.poly.fit_transform(X_colocation)
            self.colocation_model = LinearRegression()
            self.colocation_model.fit(X_colocation_poly, y_colocation)
            logger.info(f"Trained colocation model on {len(X_colocation)} data points")
            
        # Train non-colocation model
        if len(X_non_colocation) > 0:
            X_non_colocation_poly = self.poly.transform(X_non_colocation)
            self.non_colocation_model = LinearRegression()
            self.non_colocation_model.fit(X_non_colocation_poly, y_non_colocation)
            logger.info(f"Trained non-colocation model on {len(X_non_colocation)} data points")
    
    def predict_colocation_speedup_ratio(self, batch_size: int, num_spec_tokens: int) -> float:
        """Predict the speedup ratio between non-colocation and colocation modes."""
        if self.colocation_model is None or self.non_colocation_model is None:
            self._train_regression_models()
            if self.colocation_model is None or self.non_colocation_model is None:
                return 0.0
        
        # Prepare input features
        X = np.array([[batch_size, num_spec_tokens]])
        X_poly = self.poly.transform(X)
        
        # Make predictions
        colocation_time = self.colocation_model.predict(X_poly)[0]
        non_colocation_time = self.non_colocation_model.predict(X_poly)[0]
        
        # Calculate ratio (non-colocation / colocation)
        ratio = non_colocation_time / colocation_time if colocation_time > 0 else 0.0
        
        return ratio
    
    def _compute_metrics(self, actual_ratios: np.ndarray, predicted_ratios: np.ndarray) -> Dict:
        """Compute regression and classification metrics."""
        # Regression metrics
        mae = np.mean(np.abs(predicted_ratios - actual_ratios))
        rmse = np.sqrt(np.mean((predicted_ratios - actual_ratios) ** 2))
        r2 = 1 - np.sum((actual_ratios - predicted_ratios) ** 2) / np.sum((actual_ratios - np.mean(actual_ratios)) ** 2)
        
        # Calibration error
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(predicted_ratios, bin_edges) - 1
        ece = 0
        
        for i in range(num_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_pred = np.mean(predicted_ratios[mask])
                bin_actual = np.mean(actual_ratios[mask])
                ece += np.abs(bin_pred - bin_actual) * np.sum(mask) / len(actual_ratios)
        
        # Classification metrics (AUROC)
        actual_binary = (actual_ratios > 1).astype(int)
        predicted_scores = predicted_ratios
        
        auroc = None
        fpr, tpr = None, None
        if len(np.unique(actual_binary)) > 1:
            fpr, tpr, _ = roc_curve(actual_binary, predicted_scores)
            auroc = auc(fpr, tpr)
        else:
            logger.warning("AUROC cannot be calculated: all outcomes belong to same class")
            
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'ECE': ece,
            'AUROC': auroc,
            'fpr': fpr,
            'tpr': tpr
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
        
        metrics = self._compute_metrics(actual_ratios, predicted_ratios)
        return metrics
    
    def _plot_speedup_heatmap(self):
        """Plot heatmap of speedup ratio between colocation and non-colocation modes"""
        if not self.is_primary:
            return
            
        if not self.profile_results:
            logger.warning("No profile results to plot")
            return
            
        # Group results by batch_size and num_speculative_tokens
        results_dict = {}
        for (batch_size, num_spec_tokens, colocation_mode), step_times in self.profile_results.items():
            key = (batch_size, num_spec_tokens)
            if key not in results_dict:
                results_dict[key] = {'colocation': [], 'non_colocation': []}
            
            step_times = self._remove_outliers(step_times)

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
        plt.figure(figsize=(20, 8))
        sns.heatmap(speedup_matrix, 
                   xticklabels=batch_sizes,
                   yticklabels=spec_tokens,
                   cmap='YlGnBu',
                   center=1.0, 
                   annot=True,
                   fmt='.2f',  
                   cbar_kws={'label': 'Speedup Ratio (Non-colocation / Colocation)\n>1: Colocation faster\n<1: Non-colocation faster'})
        
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
        if not self.is_primary:
            return
            
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
        # Exclude fpr/tpr from the text box
        metrics_to_display = {k: v for k, v in metrics.items() if k not in ['fpr', 'tpr']}
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
    
    def _plot_roc_curve(self, fpr, tpr, auroc):
        """Plot the ROC curve."""
        if fpr is None or tpr is None or auroc is None:
            logger.info("Skipping ROC curve plot as AUROC could not be calculated.")
            return

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        plot_file = os.path.join(self.profile_dir, "roc_curve.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved ROC curve plot to {plot_file}") 