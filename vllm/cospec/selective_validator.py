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
import seaborn as sns

logger = init_logger(__name__)

class SelectiveValidator:
    def __init__(self, profiler: Profiler):
        self.history_size = 10000  # Minimum number of samples needed to train the model
        self.history_X = deque(maxlen=self.history_size)  # Pre-temperature probabilities
        self.history_y = deque(maxlen=self.history_size)  # Actual acceptance probabilities
        self.poly_features = PolynomialFeatures(degree=2)  # Use degree 2 polynomial
        self.regression_model = LinearRegression()
        self.is_model_trained = False
        # Get threshold from environment variable
        self.selective_validation_threshold = float(envs.COSPEC_SELECTIVE_VALIDATION_THRESHOLD)
        self.moving_avg_mean_tokens = 7  # Initialize moving average
        self.moving_avg_alpha = 0.1  # Smoothing factor for moving average
        self.profiler = profiler
        self.min_samples_per_bin = 100  # Minimum samples required per bin
        self.n_bins = 10  # Number of bins for probability distribution

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
        if proposals.no_proposals or proposals.unscaled_temp_probs is None or not self.is_model_trained:
            return proposals
        
        # Generate mask based on validation method
        if envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "tile":
            valid_mask = self._generate_tiled_mask(proposals, total_non_proposal_tokens)
        elif envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "threshold":
            valid_mask = self._generate_threshold_mask(proposals)
        elif envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "linear":
            valid_mask = self._generate_linear_mask(proposals, total_non_proposal_tokens)
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
        # Calculate new lengths and max length in one operation
        new_proposal_lens = valid_mask.sum(dim=1)
        new_proposal_lens[proposals.proposal_lens == 0] = 0 # what was already 0 should remain 0
        
        max_proposal_len = new_proposal_lens.max().item()

        # total_tokens = new_proposal_lens.sum().item()

        # Update proposal lengths and no_proposals flag
        proposals.proposal_lens = new_proposal_lens

        proposals.no_proposals = torch.all(new_proposal_lens == 0)
        
        # Update moving average in one operation
        self.moving_avg_mean_tokens = (
            (1 - self.moving_avg_alpha) * self.moving_avg_mean_tokens + 
            self.moving_avg_alpha * new_proposal_lens.float().mean().item()
        )
        
        # Mask invalid tokens and truncate in-place
        proposals.proposal_token_ids[~valid_mask] = 0
        proposals.proposal_token_ids = proposals.proposal_token_ids[:, :max_proposal_len]
        proposals.proposal_probs[~valid_mask] = 0
        proposals.proposal_probs = proposals.proposal_probs[:, :max_proposal_len]
        
        return proposals

    def _generate_tiled_mask(self, proposals: SpeculativeProposals, total_non_proposal_tokens: int) -> torch.Tensor:
        # Get predicted acceptance probabilities for all proposals
        acceptance_probs = self.predict_acceptance_probability(proposals)
        
        # This is for chunked prefill all tokens are filled with negative one but we should pass them 
        is_negative_one = (proposals.unscaled_temp_probs == -1)

        # Get shape and device info
        batch_size, max_proposal_len = proposals.proposal_token_ids.shape
        device = acceptance_probs.device
        
        # Create length mask and apply it to acceptance probabilities in one step
        length_mask = torch.arange(max_proposal_len, device=device)[None, :] < proposals.proposal_lens[:, None]
        masked_acceptance_probs = acceptance_probs * length_mask
        
        # Flatten and get non-zero elements more efficiently
        flat_acceptance_probs = masked_acceptance_probs.flatten()
        non_zero_mask = flat_acceptance_probs > 0
        
        if not non_zero_mask.any():
            return torch.zeros_like(length_mask)
            
        # Get sorted indices and values in one operation, avoiding unnecessary device transfers
        sorted_values, sorted_indices = torch.sort(flat_acceptance_probs[non_zero_mask], descending=True)
        non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
        sorted_original_indices = non_zero_indices[sorted_indices]
        
        # Calculate expected throughput more efficiently
        total_valid_tokens = len(sorted_values)
        latencies = torch.tensor(
            self.profiler.get_target_model_latencies(total_valid_tokens + total_non_proposal_tokens)[total_non_proposal_tokens:],
            device=device
        )
        
        # Calculate expected throughput in one operation
        expected_throughput = (torch.cumsum(sorted_values, dim=0)) / latencies
        
        # Find optimal validation length
        optimal_total_length = torch.argmax(expected_throughput).item() + 1
        
        # Create final mask in one operation
        flat_mask = torch.zeros(batch_size * max_proposal_len, dtype=torch.bool, device=device)
        flat_mask[sorted_original_indices[:optimal_total_length]] = True
        
        return flat_mask.reshape(batch_size, max_proposal_len) & length_mask | is_negative_one

    def _generate_threshold_mask(self, proposals: SpeculativeProposals) -> torch.Tensor:
        """Generate mask for threshold-based selective validation."""
        # Get predictions and create masks in one go
        acceptance_probs = self.predict_acceptance_probability(proposals)
        seq_len = proposals.proposal_token_ids.shape[1]
        device = acceptance_probs.device
        
        # Create all masks in one go
        length_mask = torch.arange(seq_len, device=device)[None, :] < proposals.proposal_lens[:, None]
        is_negative_one = (proposals.unscaled_temp_probs == -1)
        threshold_mask = acceptance_probs >= self.selective_validation_threshold
        
        return (threshold_mask & length_mask) | is_negative_one
    
    def _generate_linear_mask(self, proposals: SpeculativeProposals, total_non_proposal_tokens: int) -> torch.Tensor:
        # Get predicted acceptance probabilities for all proposals
        acceptance_probs = self.predict_acceptance_probability(proposals)

        # This is for chunked prefill all tokens are filled with negative one but we should pass them 
        is_negative_one = (proposals.unscaled_temp_probs == -1)
        
        # Get shape and device info
        batch_size, max_proposal_len = proposals.proposal_token_ids.shape
        device = acceptance_probs.device
        
        # Create length mask and apply it to acceptance probabilities in one step
        length_mask = torch.arange(max_proposal_len, device=device)[None, :] < proposals.proposal_lens[:, None]
        masked_acceptance_probs = acceptance_probs * length_mask
        
        # Flatten and get non-zero elements more efficiently
        flat_acceptance_probs = masked_acceptance_probs.flatten()
        non_zero_mask = flat_acceptance_probs > 0
        
        if not non_zero_mask.any():
            return torch.zeros_like(length_mask)
            
        # Get sorted indices and values in one operation, avoiding unnecessary device transfers
        sorted_values, sorted_indices = torch.sort(flat_acceptance_probs[non_zero_mask], descending=True)
        non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
        sorted_original_indices = non_zero_indices[sorted_indices]
        
        # Calculate expected throughput more efficiently
        total_valid_tokens = len(sorted_values)
        latencies = torch.tensor(
            self.profiler.get_target_model_latencies_linear(total_valid_tokens + total_non_proposal_tokens)[total_non_proposal_tokens:],
            device=device
        )
        
        # Calculate expected throughput in one operation
        expected_throughput = torch.cumsum(sorted_values, dim=0) / latencies
        
        # Find optimal validation length
        optimal_total_length = torch.argmax(expected_throughput).item() + 1
        
        # Create final mask in one operation
        flat_mask = torch.zeros(batch_size * max_proposal_len, dtype=torch.bool, device=device)
        flat_mask[sorted_original_indices[:optimal_total_length]] = True
        
        return flat_mask.reshape(batch_size, max_proposal_len) & length_mask | is_negative_one

    def _generate_random_mask(self, proposals: SpeculativeProposals) -> torch.Tensor:
        """Perform random drop for testing purpose"""
        # Create random mask with 50% probability of dropping each token
        random_acceptance_probs = self.random_predict_acceptance_probability(proposals)

        # Create mask for proposals that meet the threshold
        valid_mask = random_acceptance_probs >= self.selective_validation_threshold
        
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
        
        unscaled_temp_probs = proposals.unscaled_temp_probs

        # Update history and train model if needed
        self._update_history(unscaled_temp_probs, acceptance_probs)

    @nvtx_range("predict_acceptance_probability")
    def predict_acceptance_probability(self, proposals: SpeculativeProposals) -> torch.Tensor:
        """Predict acceptance probability using the trained regression model."""
        assert self.is_model_trained

        unscaled_temp_probs = proposals.unscaled_temp_probs

        # Convert to numpy for prediction
        unscaled_temp_probs_np = unscaled_temp_probs.cpu().numpy()
        original_shape = unscaled_temp_probs_np.shape
        
        # Reshape for prediction
        unscaled_temp_probs_np = unscaled_temp_probs_np.reshape(-1, 1)
        
        # Transform features to polynomial
        unscaled_temp_probs_poly = self.poly_features.transform(unscaled_temp_probs_np)
        
        # Get predictions and clip to valid range
        predictions = self.regression_model.predict(unscaled_temp_probs_poly)
        predictions = np.clip(predictions, 0, 1)
        
        # Reshape back to original shape
        predictions = predictions.reshape(original_shape)
        
        # Convert to tensor
        predictions = torch.from_numpy(predictions).to(unscaled_temp_probs.device)
        
        # If proposal len is 0 then set the acceptance probability to 0
        predictions[proposals.proposal_lens == 0] = 0
        
        # Calculate cumulative probabilities
        # For each position, multiply with all previous positions
        _, seq_len = predictions.shape
        cumulative_predictions = torch.ones_like(predictions)
        
        for i in range(seq_len):
            # For each position, multiply with all previous positions
            cumulative_predictions[:, i] = torch.prod(predictions[:, :i+1], dim=1)
        
        return cumulative_predictions

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
        
        # Calculate cumulative probabilities
        # For each position, multiply with all previous positions
        cumulative_probs = torch.ones_like(random_probs)
        
        for i in range(seq_len):
            # For each position, multiply with all previous positions
            cumulative_probs[:, i] = torch.prod(random_probs[:, :i+1], dim=1)
        
        return cumulative_probs

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

        assert len(unscaled_temp_probs_np) == len(acceptance_probs_np)

        # Assert there is no nan in the data
        assert not np.isnan(unscaled_temp_probs_np).any()
        assert not np.isnan(acceptance_probs_np).any()

        # Add new data to history
        self.history_X.extend(unscaled_temp_probs_np)
        self.history_y.extend(acceptance_probs_np)

        # Check if we have enough data in each bin
        if len(self.history_X) >= self.history_size:
            self._train_model()
            # # Convert history to numpy arrays
            # X = np.array(self.history_X)
            # y = np.array(self.history_y)
            
            # # Create bins for non-negative values only
            # valid_X = X[X >= 0]  # Filter out -1 values
            # if len(valid_X) > 0:
            #     bin_edges = np.linspace(0, 1, self.n_bins + 1)
            #     bin_indices = np.digitize(valid_X, bin_edges) - 1
                
            #     # Count samples in each bin
            #     bin_counts = np.zeros(self.n_bins)
            #     for i in range(self.n_bins):
            #         bin_counts[i] = np.sum(bin_indices == i)
                
            #     # Check if all bins have enough samples
            #     if np.all(bin_counts >= self.min_samples_per_bin):
            #         logger.info(f"Training model with bin distribution: {bin_counts}")
            #         self._train_model()
            #     else:
            #         logger.info(f"Waiting for more data. Current bin distribution: {bin_counts}")
            # else:
            #     logger.info("No valid data points (all values are -1)")

    @nvtx_range("_train_model")
    def _train_model(self):
        # Convert history to numpy arrays efficiently
        X = np.array(self.history_X, dtype=np.float32).reshape(-1, 1)
        y = np.array(self.history_y, dtype=np.float32).reshape(-1)

        # assert X and y have the same number of samples
        assert len(X) == len(y)

        # Transform features to polynomial
        X_poly = self.poly_features.fit_transform(X)

        # Train the model
        self.regression_model.fit(X_poly, y)
        self.is_model_trained = True

        # Calculate predictions for analysis
        y_pred = self.regression_model.predict(X_poly)
        
        # Generate visualization and calculate metrics
        auroc, ece = self._analyze_model_performance(X, y, y_pred)

        logger.info(
            f"Trained polynomial acceptance prediction model. "
            f"Model coefficients: {self.regression_model.coef_}, "
            f"intercept: {self.regression_model.intercept_:.4f}, "
            f"AUROC: {auroc:.4f}, ECE: {ece:.4f}"
        )

    def _analyze_model_performance(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        """Analyze model performance by generating visualizations and calculating metrics."""
        # Calculate AUROC
        auroc = roc_auc_score(y, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred)
        
        # Calculate ECE
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges) - 1
        ece = 0
        
        # Calculate bin statistics for ECE
        bin_means = []
        bin_true_means = []
        bin_counts = []
        bin_centers = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_pred = np.mean(y_pred[mask])
                bin_true = np.mean(y[mask])
                bin_count = np.sum(mask)
                ece += np.abs(bin_pred - bin_true) * bin_count / len(y)
                
                bin_means.append(bin_pred)
                bin_true_means.append(bin_true)
                bin_counts.append(bin_count)
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Plot 2: ROC curve
        plt.subplot(1, 3, 2)
        plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUROC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Plot 3: Calibration curve with histogram
        plt.subplot(1, 3, 3)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(bin_means, bin_true_means, 'o-', label=f'Model (ECE={ece:.4f})')
        
        # Calculate density instead of count
        total_samples = len(y)
        bin_densities = np.array(bin_counts) / total_samples
        
        # Add histogram of predictions
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, bin_densities, width=0.1, alpha=0.3, color='gray', label='Density')
        ax2.set_ylabel('Density')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Ensure x-label is visible
        plt.subplots_adjust(bottom=0.2)
        
        plt.tight_layout()
        plt.savefig('calibration_analysis.png')
        plt.close()

        # Create separate figure for AUROC and ECE
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
        
        # Add histogram of predictions with density
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, bin_densities, width=0.1, alpha=0.3, color='gray', label='Density')
        ax2.set_ylabel('Density')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Ensure x-label is visible
        plt.subplots_adjust(bottom=0.2)
        
        plt.tight_layout()
        plt.savefig('auroc_ece_analysis.png')
        plt.close()

        return auroc, ece

    def is_selective_validator_trained(self) -> bool:
        print("!!! is_model_trained", self.is_model_trained)
        return self.is_model_trained