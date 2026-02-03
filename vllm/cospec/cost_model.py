"""Cost Model for CoSpec v2 mode selection.

Decides per-step which execution mode to use:
- AR (autoregressive): target model only, 100% SMs
- Vanilla SD: sequential draft → target, 100% SMs each
- Colocated SD: concurrent draft + target with SM partitioning

The cost model uses analytical latency formulas parameterized by:
- B: batch size
- α: acceptance rate (EMA from verification results)
- S: number of speculative tokens (γ)

Users should fill in the latency formulas based on profiling data
for their specific target/draft model pair and GPU.
"""

import enum
from dataclasses import dataclass
from typing import Optional, Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)


class Mode(enum.Enum):
    """Execution modes for CoSpec v2."""
    AR = "ar"
    VANILLA_SD = "vanilla_sd"
    COLOCATED_SD = "colocated_sd"


@dataclass
class ModeDecision:
    """Result of the cost model's per-step decision."""
    mode: Mode
    gamma: int  # number of speculative tokens (0 for AR)
    sm_ratio: float  # fraction of SMs for target (1.0 for AR/Vanilla)


class CostModel:
    """Analytical cost model for per-step mode selection.

    Args:
        max_spec_tokens: Maximum speculative tokens (γ) to consider.
        default_sm_ratio: Default SM ratio for colocated mode.
    """

    def __init__(
        self,
        max_spec_tokens: int = 7,
        default_sm_ratio: float = 0.7,
    ):
        self.max_spec_tokens = max_spec_tokens
        self.default_sm_ratio = default_sm_ratio

        # Acceptance rate tracking via EMA
        self._alpha_ema = 0.8  # initial acceptance rate estimate
        self._ema_weight = 0.3  # EMA smoothing factor

        # Batch size tracking via EMA
        self._batch_size_ema = 1.0
        self._batch_ema_weight = 0.5

    def update_acceptance_rate(self, accepted: int, total: int) -> None:
        """Update acceptance rate EMA from verification results.

        Args:
            accepted: Number of accepted speculative tokens.
            total: Total number of speculative tokens verified.
        """
        if total == 0:
            return
        alpha = accepted / total
        self._alpha_ema = (self._ema_weight * alpha +
                           (1 - self._ema_weight) * self._alpha_ema)

    def update_batch_size(self, batch_size: int) -> None:
        """Update batch size EMA."""
        self._batch_size_ema = (
            self._batch_ema_weight * batch_size +
            (1 - self._batch_ema_weight) * self._batch_size_ema)

    @property
    def acceptance_rate(self) -> float:
        return self._alpha_ema

    @property
    def batch_size_ema(self) -> float:
        return self._batch_size_ema

    def decide(self, batch_size: int, acceptance_rate: Optional[float] = None,
               num_spec_tokens: Optional[int] = None) -> ModeDecision:
        """Decide execution mode for this step.

        Currently hardcoded to always use Colocated SD mode.
        The analytical cost model decision logic is preserved below
        for future use once profiling data is available.

        Args:
            batch_size: Current batch size (B).
            acceptance_rate: Override acceptance rate (α). Uses EMA if None.
            num_spec_tokens: Override spec tokens (γ). Uses optimal if None.

        Returns:
            ModeDecision with mode, gamma, and sm_ratio.
        """
        S = num_spec_tokens if num_spec_tokens is not None else self.max_spec_tokens
        self.update_batch_size(batch_size)
        return ModeDecision(
            mode=Mode.COLOCATED_SD,
            gamma=S,
            sm_ratio=self.default_sm_ratio,
        )

    # ------------------------------------------------------------------
    # Latency formulas — SKELETON, user fills with profiling data
    # ------------------------------------------------------------------

    def _latency_ar(self, B: int) -> float:
        """Latency of one AR step (target model forward only).

        T_ar(B) = a0 + a1*B + a2*B^2

        TODO: Fill in coefficients from profiling.
        """
        # Placeholder: linear in batch size
        a0, a1, a2 = 1.0, 0.1, 0.001
        return a0 + a1 * B + a2 * B * B

    def _latency_vanilla_sd(self, B: int, alpha: float, S: int) -> float:
        """Latency of one Vanilla SD step (sequential draft + target).

        T_vanilla(B, α, S) = T_draft(B, S) + T_target(B, S)
        where:
            T_draft(B, S) = S * (d0 + d1*B)
            T_target(B, S) = t0 + t1*N_t + t2*N_t^2, N_t = B*(S+1)

        TODO: Fill in coefficients from profiling.
        """
        # Draft latency (S sequential steps)
        d0, d1 = 0.5, 0.02
        t_draft = S * (d0 + d1 * B)

        # Target latency (one forward pass verifying all tokens)
        t0, t1, t2 = 1.0, 0.05, 0.0005
        N_t = B * (S + 1)
        t_target = t0 + t1 * N_t + t2 * N_t * N_t

        return t_draft + t_target

    def _latency_colocated_sd(self, B: int, alpha: float, S: int,
                               r: float) -> float:
        """Latency of one Colocated SD step (concurrent draft + target).

        In colocated mode, draft and target run concurrently with SM
        partitioning (r fraction to target, 1-r to draft). The step
        latency is max(T_draft_partitioned, T_target_partitioned) plus
        overhead from contention.

        T_colocated(B, α, S, r) = max(T_draft / (1-r), T_target / r) * φ
        where φ is a contention factor > 1.

        TODO: Fill in coefficients from profiling.
        """
        d0, d1 = 0.5, 0.02
        t_draft = S * (d0 + d1 * B)

        t0, t1, t2 = 1.0, 0.05, 0.0005
        N_t = B * (S + 1)
        t_target = t0 + t1 * N_t + t2 * N_t * N_t

        # Scale by partition ratio
        t_draft_partitioned = t_draft / max(1 - r, 0.01)
        t_target_partitioned = t_target / max(r, 0.01)

        # Contention overhead factor
        phi = 1.1
        return max(t_draft_partitioned, t_target_partitioned) * phi
