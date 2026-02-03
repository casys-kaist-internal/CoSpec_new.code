"""Unit tests for CoSpec Mode enum."""

from vllm.cospec.orchestrator import Mode


class TestMode:

    def test_modes_enum_values(self):
        assert Mode.AR.value == "ar"
        assert Mode.VANILLA_SD.value == "vanilla_sd"
        assert Mode.COLOCATED_SD.value == "colocated_sd"
