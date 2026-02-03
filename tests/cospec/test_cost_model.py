"""Unit tests for CoSpec v2 CostModel."""

import pytest

from vllm.cospec.cost_model import CostModel, Mode, ModeDecision


class TestCostModel:

    def test_decide_returns_valid_mode(self):
        model = CostModel(max_spec_tokens=5)
        decision = model.decide(batch_size=8)
        assert isinstance(decision, ModeDecision)
        assert decision.mode in (Mode.AR, Mode.VANILLA_SD, Mode.COLOCATED_SD)
        assert decision.gamma >= 0
        assert 0.0 < decision.sm_ratio <= 1.0

    def test_hardcoded_colocated_mode(self):
        """decide() is currently hardcoded to always return COLOCATED_SD."""
        model = CostModel(max_spec_tokens=5)
        decision = model.decide(batch_size=8)
        assert decision.mode == Mode.COLOCATED_SD
        assert decision.gamma == 5
        assert decision.sm_ratio == 0.7

    def test_update_acceptance_rate(self):
        model = CostModel()
        initial = model.acceptance_rate
        model.update_acceptance_rate(accepted=3, total=5)
        # Should have moved toward 0.6
        assert model.acceptance_rate != initial

    def test_update_acceptance_rate_zero_total(self):
        model = CostModel()
        initial = model.acceptance_rate
        model.update_acceptance_rate(accepted=0, total=0)
        assert model.acceptance_rate == initial

    def test_update_batch_size_ema(self):
        model = CostModel()
        model.update_batch_size(16)
        assert model.batch_size_ema > 1.0

    def test_decide_with_overrides(self):
        model = CostModel()
        decision = model.decide(
            batch_size=4, acceptance_rate=0.9, num_spec_tokens=3)
        assert isinstance(decision, ModeDecision)

    def test_modes_enum_values(self):
        assert Mode.AR.value == "ar"
        assert Mode.VANILLA_SD.value == "vanilla_sd"
        assert Mode.COLOCATED_SD.value == "colocated_sd"
