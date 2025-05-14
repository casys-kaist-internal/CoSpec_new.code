import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque
from vllm.logger import init_logger
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores

logger = init_logger(__name__)

class SelectiveValidator:
    def __init__(self):
        self.history_size = 10000  # Minimum number of samples needed to train the model
        self.history_X = deque(maxlen=self.history_size)  # Pre-temperature probabilities
        self.history_y = deque(maxlen=self.history_size)  # Actual acceptance probabilities
        self.regression_model = LinearRegression()
        self.is_model_trained = False
        self.selective_validation_threshold = 0.3  # Threshold for accepting proposals
        self.moving_avg_mean_tokens = 7  # Initialize moving average
        self.moving_avg_alpha = 0.1  # Smoothing factor for moving average

    def selective_validation(self, proposals: SpeculativeProposals) -> SpeculativeProposals:
        """Main entry point for selective validation.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            
        Returns:
            Modified SpeculativeProposals object with invalid tokens masked
        """
        if proposals.no_proposals or proposals.unscaled_temp_probs is None or not self.is_model_trained:
            return proposals
        
        # Get predictions for all proposals
        torch.cuda.nvtx.range_push("predict_acceptance_probability")
        acceptance_probs = self.predict_acceptance_probability(proposals)
        torch.cuda.nvtx.range_pop()
        
        # Create mask for proposals that meet the threshold
        valid_mask = acceptance_probs >= self.selective_validation_threshold
        
        # print("acceptance_probs", acceptance_probs)
        # print("valid_mask", valid_mask)
        
        # Calculate new proposal lengths based on valid tokens
        # Create a mask for tokens within proposal lengths
        length_mask = torch.arange(proposals.proposal_token_ids.shape[1], 
                                 device=proposals.proposal_token_ids.device)[None, :] < proposals.proposal_lens[:, None]
        # Combine with valid_mask to get final valid tokens
        final_mask = valid_mask & length_mask
        new_proposal_lens = final_mask.sum(dim=1)
        
        # Update proposal lengths in-place
        proposals.proposal_lens = new_proposal_lens
        max_proposal_len = max(new_proposal_lens)
        
        # Mask invalid tokens by setting them to 0
        proposals.proposal_token_ids[~final_mask] = 0
        proposals.proposal_token_ids = proposals.proposal_token_ids[:, :max_proposal_len]
        proposals.proposal_probs[~final_mask] = 0
        proposals.proposal_probs = proposals.proposal_probs[:, :max_proposal_len]
        
        # Update no_proposals flag
        proposals.no_proposals = torch.all(new_proposal_lens == 0)

        # Calculate current mean and update moving average
        current_mean = new_proposal_lens.float().mean().item()
        self.moving_avg_mean_tokens = (1 - self.moving_avg_alpha) * self.moving_avg_mean_tokens + self.moving_avg_alpha * current_mean
        
        # print(f"Moving average of selective validated tokens: {self.moving_avg_mean_tokens:.2f}")

        return proposals

    def random_drop(self, proposals: SpeculativeProposals) -> SpeculativeProposals:
        """Perform random drop for testing purpose"""
        if proposals.no_proposals:
            return proposals
            
        # Create random mask with 50% probability of dropping each token
        random_acceptance_probs = self.random_predict_acceptance_probability(proposals)

        # Create mask for proposals that meet the threshold
        valid_mask = random_acceptance_probs >= self.selective_validation_threshold
        
        # print("acceptance_probs", acceptance_probs)
        # print("valid_mask", valid_mask)
        
        # Calculate new proposal lengths based on valid tokens
        # Create a mask for tokens within proposal lengths
        length_mask = torch.arange(proposals.proposal_token_ids.shape[1], 
                                 device=proposals.proposal_token_ids.device)[None, :] < proposals.proposal_lens[:, None]
        # Combine with valid_mask to get final valid tokens
        final_mask = valid_mask & length_mask
        new_proposal_lens = final_mask.sum(dim=1)

        # Update proposal lengths in-place
        proposals.proposal_lens = new_proposal_lens
        max_proposal_len = max(new_proposal_lens)
        
        # Mask invalid tokens by setting them to 0
        proposals.proposal_token_ids[~final_mask] = 0
        proposals.proposal_token_ids = proposals.proposal_token_ids[:, :max_proposal_len]
        proposals.proposal_probs[~final_mask] = 0
        proposals.proposal_probs = proposals.proposal_probs[:, :max_proposal_len]
        
        # Update no_proposals flag
        proposals.no_proposals = torch.all(new_proposal_lens == 0)

        # Calculate current mean and update moving average
        current_mean = new_proposal_lens.float().mean().item()
        self.moving_avg_mean_tokens = (1 - self.moving_avg_alpha) * self.moving_avg_mean_tokens + self.moving_avg_alpha * current_mean
        
        # print(f"Moving average of selective validated tokens: {self.moving_avg_mean_tokens:.2f}")

        return proposals

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

    def predict_acceptance_probability(self, proposals: SpeculativeProposals) -> torch.Tensor:
        """Predict acceptance probability using the trained regression model.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
        
        Returns:
            Tensor of shape [batch_size, max_proposal_len] containing predicted
            cumulative acceptance probabilities
        """
        assert self.is_model_trained

        unscaled_temp_probs = proposals.unscaled_temp_probs

        # Convert to numpy for prediction
        unscaled_temp_probs_np = unscaled_temp_probs.cpu().numpy()
        original_shape = unscaled_temp_probs_np.shape
        
        # Reshape for prediction
        unscaled_temp_probs_np = unscaled_temp_probs_np.reshape(-1, 1)
        
        # Get predictions and clip to valid range
        predictions = self.regression_model.predict(unscaled_temp_probs_np)
        predictions = np.clip(predictions, 0, 1)
        
        # Reshape back to original shape
        predictions = predictions.reshape(original_shape)
        
        # Convert to tensor
        predictions = torch.from_numpy(predictions).to(unscaled_temp_probs.device)
        
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

    def _update_history(self, unscaled_temp_probs, acceptance_probs):
        """Update the history of proposal acceptance data and train model if enough data is available.
        
        Args:
            unscaled_temp_probs: Tensor of pre-temperature probabilities
            acceptance_probs: Tensor of actual acceptance probabilities
        """
        if self.is_model_trained:
            return
        
        # Convert to numpy and flatten
        unscaled_temp_probs_np = unscaled_temp_probs.cpu().numpy().flatten()
        acceptance_probs_np = acceptance_probs.cpu().numpy().flatten()

        # Add new data to history
        self.history_X.extend(unscaled_temp_probs_np)
        self.history_y.extend(acceptance_probs_np)

        # Train model if we have enough data
        if len(self.history_X) >= self.history_size and not self.is_model_trained:
            self._train_model()

    def _train_model(self):
        # Convert history to numpy arrays efficiently
        X = np.array(self.history_X, dtype=np.float32).reshape(-1, 1)
        y = np.array(self.history_y, dtype=np.float32).reshape(-1)

        # Ensure X and y have the same number of samples
        min_samples = min(len(X), len(y))
        X = X[:min_samples]
        y = y[:min_samples]

        # Train the model
        self.regression_model.fit(X, y)
        self.is_model_trained = True

        logger.info(
            f"Trained acceptance prediction model with {min_samples} samples. "
            f"Model coefficients: {self.regression_model.coef_[0]:.4f}, "
            f"intercept: {self.regression_model.intercept_:.4f}"
        )

    def is_selective_validator_trained(self) -> bool:
        return self.is_model_trained