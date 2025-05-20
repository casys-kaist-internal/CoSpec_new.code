import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque
from vllm.config import envs
from vllm.cospec.profiler import Profiler
from vllm.logger import init_logger
from vllm.sequence import VLLM_INVALID_TOKEN_ID
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores
from vllm.spec_decode.util import nvtx_range

logger = init_logger(__name__)

class SelectiveValidator:
    def __init__(self, profiler: Profiler):
        self.history_size = 10000  # Minimum number of samples needed to train the model
        self.history_X = deque(maxlen=self.history_size)  # Pre-temperature probabilities
        self.history_y = deque(maxlen=self.history_size)  # Actual acceptance probabilities
        self.regression_model = LinearRegression()
        self.is_model_trained = False
        # Get threshold from environment variable
        self.selective_validation_threshold = float(envs.COSPEC_SELECTIVE_VALIDATION_THRESHOLD)
        self.moving_avg_mean_tokens = 7  # Initialize moving average
        self.moving_avg_alpha = 0.1  # Smoothing factor for moving average
        self.profiler = profiler

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
        elif envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "threshold_tile":
            valid_mask = self._generate_threshold_tile_mask(proposals, total_non_proposal_tokens)    
        elif envs.COSPEC_SELECTIVE_VALIDATION_METHOD == "linear":
            valid_mask = self._generate_linear_mask(proposals, total_non_proposal_tokens)
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
        max_proposal_len = new_proposal_lens.max().item()
        
        # Update proposal lengths and no_proposals flag
        proposals.proposal_lens = new_proposal_lens
        proposals.no_proposals = torch.all(new_proposal_lens == 0)
        
        # Update moving average in one operation
        self.moving_avg_mean_tokens = (
            (1 - self.moving_avg_alpha) * self.moving_avg_mean_tokens + 
            self.moving_avg_alpha * new_proposal_lens.float().mean().item()
        )
        
        # Mask invalid tokens and truncate in-place
        if not proposals.no_proposals:
            proposals.proposal_token_ids[~valid_mask] = 0
            proposals.proposal_token_ids = proposals.proposal_token_ids[:, :max_proposal_len]
            proposals.proposal_probs[~valid_mask] = 0
            proposals.proposal_probs = proposals.proposal_probs[:, :max_proposal_len]
        
        return proposals

    def _generate_tiled_mask(self, proposals: SpeculativeProposals, total_non_proposal_tokens: int) -> torch.Tensor:
        """Generate mask for tiled selective validation.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            total_non_proposal_tokens: Total number of non-proposal tokens
            
        Returns:
            Boolean tensor mask indicating which tokens to validate
        """
        # Get predicted acceptance probabilities for all proposals
        acceptance_probs = self.predict_acceptance_probability(proposals)
        
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
        expected_throughput = torch.cumsum(sorted_values, dim=0) / latencies
        
        # Find optimal validation length
        optimal_total_length = torch.argmax(expected_throughput).item() + 1
        
        # Create final mask in one operation
        flat_mask = torch.zeros(batch_size * max_proposal_len, dtype=torch.bool, device=device)
        flat_mask[sorted_original_indices[:optimal_total_length]] = True
        
        return flat_mask.reshape(batch_size, max_proposal_len) & length_mask

    def _generate_threshold_mask(self, proposals: SpeculativeProposals) -> torch.Tensor:
        """Generate mask for threshold-based selective validation.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            
        Returns:
            Boolean tensor mask indicating which tokens to validate
        """
        # Get predictions for all proposals
        torch.cuda.nvtx.range_push("predict_acceptance_probability")
        acceptance_probs = self.predict_acceptance_probability(proposals)
        torch.cuda.nvtx.range_pop()
        
        # Create length mask and combine with threshold mask in one step
        seq_len = proposals.proposal_token_ids.shape[1]
        device = proposals.proposal_token_ids.device
        return (
            (acceptance_probs >= self.selective_validation_threshold) & 
            (torch.arange(seq_len, device=device)[None, :] < proposals.proposal_lens[:, None])
        )
    
    def _generate_threshold_tile_mask(
            self,
            proposals: SpeculativeProposals,
            total_non_proposal_tokens: int
        ) -> torch.Tensor:
        """
        Generate mask for threshold-based selective validation, ensuring
        (total_non_proposal_tokens + validated token count) is near a multiple of 64
        by rounding to whichever is closer (up or down).
        """

        # 1) Get predicted acceptance probabilities.
        torch.cuda.nvtx.range_push("predict_acceptance_probability")
        acceptance_probs = self.predict_acceptance_probability(proposals)
        torch.cuda.nvtx.range_pop()

        batch_size, max_len = proposals.proposal_token_ids.shape
        device = proposals.proposal_token_ids.device

        # 2) Create length mask and threshold mask.
        length_mask = (torch.arange(max_len, device=device)[None, :] <
                    proposals.proposal_lens[:, None])
        threshold_mask = acceptance_probs >= self.selective_validation_threshold
        combined_mask = threshold_mask & length_mask

        # 3) Flatten and filter by the combined_mask.
        flat_acceptance = acceptance_probs.flatten()
        flat_mask = combined_mask.flatten()
        passing_indices = torch.nonzero(flat_mask).squeeze(-1)

        # If no tokens pass threshold, just return all False.
        if len(passing_indices) == 0:
            return combined_mask  # all False

        # 4) Sort by acceptance probability descending.
        passing_probs = flat_acceptance[passing_indices]
        sorted_vals, sorted_idx = torch.sort(passing_probs, descending=True)
        sorted_indices = passing_indices[sorted_idx]
        max_candidates = len(sorted_indices)

        current_count_mod64 = total_non_proposal_tokens % 64
        # If we're already at multiple of 64, just keep all passing tokens.
        if current_count_mod64 == 0:
            chosen_count = max_candidates
        else:
            # remainder_down is how many tokens we'd drop to round down
            remainder_down = current_count_mod64
            # remainder_up is how many tokens we'd add to round up
            remainder_up = (64 - remainder_down) % 64

            # If we can't add enough tokens (remainder_up is too big), go down.
            can_round_up = remainder_up <= max_candidates
            # If we can't drop enough tokens, we have to go up.
            can_round_down = remainder_down <= max_candidates

            if not can_round_up and not can_round_down:
                # If neither is feasible, keep as many as possible.
                chosen_count = max_candidates
            else:
                # Compare which remainder is smaller, provided it's feasible.
                if can_round_up and can_round_down:
                    if remainder_up < remainder_down:
                        chosen_count = remainder_up
                    else:
                        # remainder_down <= remainder_up
                        chosen_count = max_candidates - remainder_down
                elif can_round_up:
                    chosen_count = remainder_up
                else:
                    # can_round_down only
                    chosen_count = max_candidates - remainder_down

            # Clamp chosen_count if somehow the logic above goes out of range
            chosen_count = max(0, min(chosen_count, max_candidates))

        # 6) Mark exactly those chosen_count highest-acceptance positions.
        final_chosen_indices = sorted_indices[:chosen_count]
        final_flat_mask = torch.zeros_like(flat_mask)
        final_flat_mask[final_chosen_indices] = True

        # 7) Reshape to [batch_size, max_len].
        final_mask = final_flat_mask.reshape(batch_size, max_len)
        return final_mask

    def _generate_linear_mask(self, proposals: SpeculativeProposals, total_non_proposal_tokens: int) -> torch.Tensor:
        """Generate mask for linear selective validation.
        
        Args:
            proposals: SpeculativeProposals object containing the proposal data
            
        Returns:
            Boolean tensor mask indicating which tokens to validate
        """
        # Get predicted acceptance probabilities for all proposals
        acceptance_probs = self.predict_acceptance_probability(proposals)
        
        # Flatten and sort all tokens by acceptance probability
        batch_size, max_proposal_len = proposals.proposal_token_ids.shape
        device = acceptance_probs.device
        
        # Create length mask first to avoid unnecessary computations
        length_mask = torch.arange(max_proposal_len, device=device)[None, :] < proposals.proposal_lens[:, None]
        
        # Apply length mask to acceptance probabilities and flatten
        masked_acceptance_probs = acceptance_probs * length_mask
        flat_acceptance_probs = masked_acceptance_probs.flatten()
        
        # Get non-zero elements more efficiently
        non_zero_mask = flat_acceptance_probs > 0
        if not non_zero_mask.any():
            return torch.zeros_like(length_mask)
            
        # Get sorted indices and values in one operation
        sorted_values, sorted_indices = torch.sort(flat_acceptance_probs[non_zero_mask], descending=True)
        non_zero_indices = torch.nonzero(non_zero_mask).squeeze(-1)
        sorted_original_indices = non_zero_indices[sorted_indices]
        
        # Calculate expected throughput more efficiently
        total_valid_tokens = len(sorted_values)
        latencies = self.profiler.get_target_model_latencies_linear(total_valid_tokens + total_non_proposal_tokens)
        latencies = latencies[total_non_proposal_tokens:]
        latencies = torch.tensor(latencies, device=device)
        
        # Pre-compute cumulative sums for expected accepted tokens
        cumsum_expected_tokens = torch.cumsum(sorted_values, dim=0)
        
        # Calculate expected throughput in one operation
        expected_throughput = (cumsum_expected_tokens + total_non_proposal_tokens) / latencies
        
        # Find optimal validation length
        max_throughput_idx = torch.argmax(expected_throughput)
        optimal_total_length = max_throughput_idx + 1

        print("threshold for validation", sorted_values[optimal_total_length - 1])
                
        # Create flat mask directly with the optimal indices
        flat_mask = torch.zeros(batch_size * max_proposal_len, dtype=torch.bool, device=device)
        flat_mask[sorted_original_indices[:optimal_total_length]] = True
        
        # Reshape and combine with length mask in one operation
        return flat_mask.reshape(batch_size, max_proposal_len) & length_mask

    def random_drop(self, proposals: SpeculativeProposals) -> SpeculativeProposals:
        """Perform random drop for testing purpose"""
        if proposals.no_proposals:
            return proposals
            
        # Create random mask with 50% probability of dropping each token
        random_acceptance_probs = self.random_predict_acceptance_probability(proposals)

        # Create mask for proposals that meet the threshold
        valid_mask = random_acceptance_probs >= self.selective_validation_threshold
        
        # Create a mask for tokens within proposal lengths
        length_mask = torch.arange(proposals.proposal_token_ids.shape[1], 
                                 device=proposals.proposal_token_ids.device)[None, :] < proposals.proposal_lens[:, None]
        # Combine with valid_mask to get final valid tokens
        final_mask = valid_mask & length_mask

        # Apply common token masking logic
        return self._apply_validation_mask(proposals, final_mask)

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
        unscaled_temp_probs_np = unscaled_temp_probs.cpu().numpy().flatten()
        acceptance_probs_np = acceptance_probs.cpu().numpy().flatten()

        # Assert there is no nan in the data
        assert not np.isnan(unscaled_temp_probs_np).any()
        assert not np.isnan(acceptance_probs_np).any()

        # Add new data to history
        self.history_X.extend(unscaled_temp_probs_np)
        self.history_y.extend(acceptance_probs_np)

        # Train model if we have enough data
        if len(self.history_X) >= self.history_size and not self.is_model_trained:
            self._train_model()

    @nvtx_range("_train_model")
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