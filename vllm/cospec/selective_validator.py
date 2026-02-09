import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from collections import deque
from vllm.config import envs
from vllm.cospec.profiler import Profiler
from vllm.logger import init_logger
from vllm.sequence import VLLM_INVALID_TOKEN_ID
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores
from vllm.spec_decode.util import nvtx_range
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import os
from datetime import datetime

logger = init_logger(__name__)

class SelectiveValidator:
    def __init__(self, profiler: Profiler):
        self.history_size = 50000  # Minimum number of samples needed to train the model
        self.history_X = deque(maxlen=self.history_size)  # Pre-temperature probabilities
        self.history_y = deque(maxlen=self.history_size)  # Actual acceptance probabilities
        self.poly = PolynomialFeatures(degree=1)
        self.regression_model = LinearRegression()
        self.is_model_trained = False
        self.selective_validation_threshold = float(envs.COSPEC_SELECTIVE_VALIDATION_THRESHOLD)
        self.mean_selective_validation_tokens_ema = 7
        self.moving_avg_alpha = 0.1  # Smoothing factor for moving average
        self.has_first_data_point = False  # Flag to track if we have received first data point
        self.profiler = profiler

        self.validation_size = 20000  # Size of validation dataset
        self.validation_X = []
        self.validation_y = []
        self.validation_completed = False

        if envs.COSPEC_SELECTIVE_VALIDATION:
            logger.info(f"Selective validation enabled with method: {envs.COSPEC_SELECTIVE_VALIDATION_METHOD}")
        else:
            logger.info("Selective validation disabled")

    def selective_validation(self, proposals: SpeculativeProposals, total_non_proposal_tokens: int) -> SpeculativeProposals:
        """Main entry point for selective validation.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            total_non_proposal_tokens: Total number of non-proposal tokens
            
        Returns:
            Modified SpeculativeProposals object with tokens to validate selected
        """
        if proposals.no_proposals or proposals.unscaled_temp_probs is None:
            return proposals

        # Random method bypasses training guards — it generates random masks
        # independently of any trained model (used for correctness testing).
        if envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "random":
            valid_mask = self._generate_random_mask(proposals)
            return self._apply_validation_mask(proposals, valid_mask)

        if not self.is_model_trained or not self.validation_completed:
            return proposals

        # Generate mask based on validation method
        if envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "tile":
            valid_mask = self._generate_tiled_mask(proposals, total_non_proposal_tokens)
        elif envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "linear":
            valid_mask = self._generate_linear_mask(proposals, total_non_proposal_tokens)
        elif envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "polynomial":
            valid_mask = self._generate_polynomial_mask(proposals, total_non_proposal_tokens)
        elif envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "threshold":
            valid_mask = self._generate_threshold_mask(proposals)
        elif envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "random": # For correctness testing purpose 
            valid_mask = self._generate_random_mask(proposals)
        else:
            raise ValueError(f"Invalid selective validation method: {envs.COSPEC_SELECTIVE_VALIDATION_METHOD}")
        
        # Apply common token masking logic
        return self._apply_validation_mask(proposals, valid_mask)

    def _apply_validation_mask(self, proposals: SpeculativeProposals, valid_mask: torch.Tensor) -> SpeculativeProposals:
        """Apply validation mask to proposals and update proposal properties.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            valid_mask: Boolean tensor mask indicating which tokens to validate
            
        Returns:
            Modified SpeculativeProposals object
        """
        new_proposal_lens = valid_mask.sum(dim=1)
        new_proposal_lens[proposals.proposal_lens == 0] = 0 # what was already 0 should remain 0
        
        max_proposal_len = new_proposal_lens.max().item()

        proposals.proposal_lens = new_proposal_lens
        proposals.no_proposals = torch.all(new_proposal_lens == 0)
        mean_selective_validation_tokens = new_proposal_lens.float().mean().item()
        
        if not self.has_first_data_point:
            self.mean_selective_validation_tokens_ema = mean_selective_validation_tokens
            self.has_first_data_point = True
        else:
            self.mean_selective_validation_tokens_ema = (
                (1 - self.moving_avg_alpha) * self.mean_selective_validation_tokens_ema + 
                self.moving_avg_alpha * mean_selective_validation_tokens
            )
                
        proposals.proposal_token_ids[~valid_mask] = 0
        proposals.proposal_token_ids = proposals.proposal_token_ids[:, :max_proposal_len]
        proposals.proposal_probs[~valid_mask] = 0
        proposals.proposal_probs = proposals.proposal_probs[:, :max_proposal_len]
        
        return proposals
    
    def get_mean_selective_validation_tokens_ema(self) -> int:
        return self.mean_selective_validation_tokens_ema

    def _generate_latency_aware_mask(self, proposals: SpeculativeProposals,
                                     total_non_proposal_tokens: int,
                                     latency_fn) -> torch.Tensor:
        """Generate mask using latency-aware throughput optimization.

        Args:
            proposals: The speculative proposals
            total_non_proposal_tokens: Number of non-proposal tokens
            latency_fn: Function that returns latencies for a given token count
        """
        acceptance_probs = self.predict_acceptance_probabilities(proposals.unscaled_temp_probs)
        cumulative_acceptance_probs = torch.cumprod(acceptance_probs, dim=1)

        # Chunked prefill tokens are filled with -1 but should be passed through
        is_negative_one = (proposals.unscaled_temp_probs == -1)

        batch_size, max_proposal_len = proposals.proposal_token_ids.shape
        device = cumulative_acceptance_probs.device

        length_mask = torch.arange(max_proposal_len, device=device)[None, :] < proposals.proposal_lens[:, None]
        masked_acceptance_probs = cumulative_acceptance_probs * length_mask

        flat_acceptance_probs = masked_acceptance_probs.flatten()
        sorted_values, sorted_indices = torch.sort(flat_acceptance_probs, descending=True)

        total_valid_tokens = len(sorted_values)
        latencies = torch.tensor(
            latency_fn(total_valid_tokens + total_non_proposal_tokens)[total_non_proposal_tokens:],
            device=device
        )
        expected_throughput = torch.cumsum(sorted_values, dim=0) / latencies

        # Find the first index where probability goes below threshold
        start_idx = torch.where(sorted_values < self.selective_validation_threshold)[0]
        if len(start_idx) > 0:
            start_idx = start_idx[0].item()
            valid_throughput = expected_throughput[start_idx:]
            optimal_total_length = start_idx + torch.argmax(valid_throughput).item() + 1
        else:
            optimal_total_length = total_valid_tokens

        flat_mask = torch.zeros(batch_size * max_proposal_len, dtype=torch.bool, device=device)
        flat_mask[sorted_indices[:optimal_total_length]] = True

        return flat_mask.reshape(batch_size, max_proposal_len) & length_mask | is_negative_one

    def _generate_tiled_mask(self, proposals: SpeculativeProposals, total_non_proposal_tokens: int) -> torch.Tensor:
        return self._generate_latency_aware_mask(
            proposals, total_non_proposal_tokens, self.profiler.get_target_model_latencies)

    def _generate_linear_mask(self, proposals: SpeculativeProposals, total_non_proposal_tokens: int) -> torch.Tensor:
        return self._generate_latency_aware_mask(
            proposals, total_non_proposal_tokens, self.profiler.get_target_model_latencies_linear)

    def _generate_polynomial_mask(self, proposals: SpeculativeProposals, total_non_proposal_tokens: int) -> torch.Tensor:
        return self._generate_latency_aware_mask(
            proposals, total_non_proposal_tokens, self.profiler.get_target_model_latencies_polynomial)

    def _generate_threshold_mask(self, proposals: SpeculativeProposals) -> torch.Tensor:
        """Generate mask for threshold-based selective validation."""
        acceptance_probs = self.predict_acceptance_probabilities(proposals.unscaled_temp_probs)
        cumulative_acceptance_probs = torch.cumprod(acceptance_probs, dim=1)
        seq_len = proposals.proposal_token_ids.shape[1]
        device = cumulative_acceptance_probs.device

        length_mask = torch.arange(seq_len, device=device)[None, :] < proposals.proposal_lens[:, None]
        is_negative_one = (proposals.unscaled_temp_probs == -1)
        threshold_mask = cumulative_acceptance_probs >= self.selective_validation_threshold

        return (threshold_mask & length_mask) | is_negative_one
        
    def _generate_random_mask(self, proposals: SpeculativeProposals) -> torch.Tensor:
        """Perform random drop for testing purpose"""
        # Create random mask with 50% probability of dropping each token
        random_acceptance_probs = self.random_predict_acceptance_probability(proposals)
        cumulative_acceptance_probs = torch.cumprod(random_acceptance_probs, dim=1)

        # Create mask for proposals that meet the threshold
        valid_mask = cumulative_acceptance_probs >= self.selective_validation_threshold
        
        # Create a mask for tokens within proposal lengths
        length_mask = torch.arange(proposals.proposal_token_ids.shape[1], 
                                 device=proposals.proposal_token_ids.device)[None, :] < proposals.proposal_lens[:, None]
        # Combine with valid_mask to get final valid tokens
        final_mask = valid_mask & length_mask

        return final_mask

    @nvtx_range("update_proposal_history")
    def update_proposal_history(self, proposals: SpeculativeProposals, proposal_scores: SpeculativeScores):
        """Update the history of proposal acceptance data for training the regression model.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            proposal_scores: Tensor containing the actual acceptance scores
        """
        if proposals.no_proposals or proposals.unscaled_temp_probs is None:
            return
        
        # Calculate actual acceptance probabilities
        acceptance_probs = self._calculate_acceptance_probabilities(
            proposals, proposal_scores)

        # Update history and train model if needed
        self._update_history(proposals.unscaled_temp_probs, acceptance_probs)

    @nvtx_range("predict_acceptance_probabilities")
    def predict_acceptance_probabilities(
        self, unscaled_temp_probs: torch.Tensor
    ) -> torch.Tensor:
        """Predict acceptance probabilities for each token."""
        if not self.is_model_trained:
            return torch.ones_like(unscaled_temp_probs)

        # Store original shape
        original_shape = unscaled_temp_probs.shape

        # Convert to numpy and reshape
        unscaled_temp_probs_np = unscaled_temp_probs.cpu().numpy().reshape(-1, 1)

        # Transform features to polynomial features
        unscaled_temp_probs_poly = self.poly.transform(unscaled_temp_probs_np)

        # Get predictions and clip to valid range
        predictions = self.regression_model.predict(unscaled_temp_probs_poly)
        predictions = np.clip(predictions, 0, 1)

        # Reshape back to original shape
        predictions = predictions.reshape(original_shape)

        # Convert back to tensor
        return torch.from_numpy(predictions).to(unscaled_temp_probs.device)

    def random_predict_acceptance_probability(self, proposals: SpeculativeProposals) -> torch.Tensor:
        """Generate random acceptance probabilities for testing purposes.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
        
        Returns:
            Tensor of shape [batch_size, max_proposal_len] containing random
            cumulative acceptance probabilities
        """
        batch_size, seq_len = proposals.proposal_token_ids.shape
        
        # Generate random probabilities between 0 and 1
        random_probs = torch.rand(batch_size, seq_len, device=proposals.proposal_token_ids.device)
        
        return random_probs

    @nvtx_range("_calculate_acceptance_probabilities")
    def _calculate_acceptance_probabilities(self, proposals: SpeculativeProposals, proposal_scores: SpeculativeScores) -> torch.Tensor:
        """Calculate actual acceptance probabilities for proposals.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            proposal_scores: Tensor containing the actual acceptance scores
            
        Returns:
            Tensor of acceptance probabilities
        """
        target_probs = proposal_scores.probs
        draft_probs = proposals.proposal_probs
        draft_token_ids = proposals.proposal_token_ids
        # Create a mask for rows that don't contain any invalid tokens
        valid_rows = ~torch.any(draft_token_ids == VLLM_INVALID_TOKEN_ID, dim=1)
        
        # Update tensors in-place by masking invalid rows
        target_probs = target_probs[valid_rows]
        draft_probs = draft_probs[valid_rows]
        draft_token_ids = draft_token_ids[valid_rows]

        # Get probabilities for proposed tokens
        selected_target_probs = torch.gather(
            target_probs,
            dim=-1,
            index=draft_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        selected_draft_probs = torch.gather(
            draft_probs,
            dim=-1,
            index=draft_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Calculate acceptance probability as min(target_prob/draft_prob, 1)
        acceptance_probability = torch.minimum(
            selected_target_probs / selected_draft_probs,
            torch.full((1, ), 1, device=target_probs.device))

        return acceptance_probability

    @nvtx_range("_update_history")
    def _update_history(self, unscaled_temp_probs, acceptance_probs):
        """Update the history of proposal acceptance data and train model if enough data is available.
        
        Args:
            unscaled_temp_probs: Tensor of pre-temperature probabilities
            acceptance_probs: Tensor of actual acceptance probabilities
        """
        if self.is_model_trained:
            if not self.validation_completed:
                self.collect_validation_data(unscaled_temp_probs, acceptance_probs)
            return

        # Convert to numpy and flatten
        unscaled_temp_probs_np = unscaled_temp_probs.cpu().numpy()
        acceptance_probs_np = acceptance_probs.cpu().numpy()
        
        # Flatten both arrays
        unscaled_temp_probs_np = unscaled_temp_probs_np.flatten()
        acceptance_probs_np = acceptance_probs_np.flatten()

        # Filter out -1 values
        valid_mask = unscaled_temp_probs_np != -1
        unscaled_temp_probs_np = unscaled_temp_probs_np[valid_mask]
        acceptance_probs_np = acceptance_probs_np[valid_mask]

        assert len(unscaled_temp_probs_np) == len(acceptance_probs_np)

        # Assert there is no nan in the data
        assert not np.isnan(unscaled_temp_probs_np).any()
        assert not np.isnan(acceptance_probs_np).any()

        # Add new data to history
        self.history_X.extend(unscaled_temp_probs_np)
        self.history_y.extend(acceptance_probs_np)

        # Check if we have enough data in each bin
        if len(self.history_X) >= self.history_size:
            self.train_model()

    def train_model(self):
        """Train the polynomial regression model on historical data."""
        if len(self.history_X) < 2:
            return

        # Convert history to numpy arrays
        X = np.array(list(self.history_X)).reshape(-1, 1)
        y = np.array(list(self.history_y))

        # Transform features to polynomial features
        X_poly = self.poly.fit_transform(X)

        # Train the model
        self.regression_model.fit(X_poly, y)
        self.is_model_trained = True

        logger.info(
            f"Trained polynomial regression model (degree 2). "
            f"Model coefficients: {self.regression_model.coef_}, "
            f"intercept: {self.regression_model.intercept_:.4f}, "
        )

    def is_selective_validator_trained(self) -> bool:
        return self.is_model_trained and self.validation_completed

    def collect_validation_data(self, unscaled_temp_probs: torch.Tensor, acceptance_probs: torch.Tensor):
        """Collect validation data for model evaluation.
        
        Args:
            unscaled_temp_probs: Tensor of pre-temperature probabilities
            acceptance_probs: Tensor of actual acceptance probabilities
        """
        if not self.is_model_trained:
            logger.warning("Model is not trained yet. Cannot collect validation data.")
            return
        
        # Convert to numpy and flatten
        unscaled_temp_probs_np = unscaled_temp_probs.cpu().numpy()
        acceptance_probs_np = acceptance_probs.cpu().numpy()
        
        # Flatten both arrays
        unscaled_temp_probs_np = unscaled_temp_probs_np.flatten()
        acceptance_probs_np = acceptance_probs_np.flatten()

        # Filter out -1 values
        valid_mask = unscaled_temp_probs_np != -1
        unscaled_temp_probs_np = unscaled_temp_probs_np[valid_mask]
        acceptance_probs_np = acceptance_probs_np[valid_mask]

        # Ensure both arrays have the same length
        assert len(unscaled_temp_probs_np) == len(acceptance_probs_np)

        # Assert there is no nan in the data
        assert not np.isnan(unscaled_temp_probs_np).any()
        assert not np.isnan(acceptance_probs_np).any()

        # Add new data to validation set
        self.validation_X.extend(unscaled_temp_probs_np)
        self.validation_y.extend(acceptance_probs_np)

        # Check if we have enough validation data
        if len(self.validation_X) >= self.validation_size:
            self.has_enough_validation_data = True
            logger.info(f"Collected enough validation data ({len(self.validation_X)} samples)")
            self.evaluate_validation_data()
    

    def evaluate_validation_data(self, save_path: str = 'validation_evaluation.png'):
        """Evaluate model performance on collected validation data.
        
        Args:
            save_path: Path to save the evaluation plots
        """
        # Convert validation data to numpy arrays
        X_val = np.array(self.validation_X, dtype=np.float32).reshape(-1, 1)
        y_val = np.array(self.validation_y, dtype=np.float32)

        # Get predictions
        y_pred = self.predict_acceptance_probabilities(torch.from_numpy(X_val).to(torch.device('cpu')))
        y_pred = y_pred.cpu().numpy()
        y_pred = np.clip(y_pred, 0, 1)

        # Save data to CSV files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory for data if it doesn't exist
        data_dir = "validation_data"
        os.makedirs(data_dir, exist_ok=True)

        # Save raw data
        raw_data = pd.DataFrame({
            'unscaled_temp_probs': X_val.flatten(),
            'actual_acceptance': y_val.flatten(),
            'predicted_acceptance': y_pred.flatten()
        })
        raw_data.to_csv(os.path.join(data_dir, f'raw_data_{timestamp}.csv'), index=False)

        # Calculate and save ROC curve data
        # Convert to binary classification by using a threshold
        threshold = 0.5
        y_val_binary = (y_val.flatten() >= threshold).astype(int)
        y_pred_binary = (y_pred.flatten() >= threshold).astype(int)
        
        try:
            fpr, tpr, thresholds = roc_curve(y_val_binary, y_pred.flatten())
            roc_data = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            })
            roc_data.to_csv(os.path.join(data_dir, f'roc_data_{timestamp}.csv'), index=False)
            auroc = roc_auc_score(y_val_binary, y_pred.flatten())
        except ValueError as e:
            logger.warning(f"Could not calculate ROC curve: {e}")
            auroc = 0.0
            fpr, tpr = np.array([0, 1]), np.array([0, 1])

        # Calculate and save calibration data
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred.flatten(), bin_edges) - 1
        
        bin_means = []
        bin_true_means = []
        bin_counts = []
        bin_centers = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_pred = np.mean(y_pred.flatten()[mask])
                bin_true = np.mean(y_val.flatten()[mask])
                bin_count = np.sum(mask)
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                
                bin_means.append(bin_pred)
                bin_true_means.append(bin_true)
                bin_counts.append(bin_count)
                bin_centers.append(bin_center)

        calibration_data = pd.DataFrame({
            'bin_center': bin_centers,
            'predicted_mean': bin_means,
            'actual_mean': bin_true_means,
            'count': bin_counts,
            'density': np.array(bin_counts) / len(y_val.flatten())
        })
        calibration_data.to_csv(os.path.join(data_dir, f'calibration_data_{timestamp}.csv'), index=False)

        # Calculate ECE
        ece = sum(np.abs(np.array(bin_means) - np.array(bin_true_means)) * np.array(bin_counts) / len(y_val.flatten()))

        # Save metrics
        metrics_data = pd.DataFrame({
            'metric': ['AUROC', 'ECE'],
            'value': [auroc, ece]
        })
        metrics_data.to_csv(os.path.join(data_dir, f'metrics_{timestamp}.csv'), index=False)

        # Create visualization
        plt.figure(figsize=(10, 5))
        
        # Plot 1: ROC curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUROC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Plot 2: Calibration curve with histogram
        plt.subplot(1, 2, 2)    
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(bin_means, bin_true_means, 'o-', label=f'Model (ECE={ece:.4f})')
        
        # Calculate density instead of count
        total_samples = len(y_val.flatten())
        bin_densities = np.array(bin_counts) / total_samples
        
        # Add histogram of predictions
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, bin_densities, width=0.1, alpha=0.3, color='gray', label='Density')
        ax2.set_ylabel('Density')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title(f'Calibration Curve (ECE={ece:.4f})')
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Validation data evaluation - AUROC: {auroc:.4f}, ECE: {ece:.4f}")
        logger.info(f"Data saved to {data_dir} directory with timestamp {timestamp}")
        
        # Set validation plot completed flag
        self.validation_completed = True
        
        return auroc, ece