"""Unit tests for CoSpec v2 SMController.

These tests verify the SMController interface without requiring
an actual GPU or libsmctrl.so. GPU-dependent tests are marked
with pytest.mark.skipif.
"""

import os

import pytest

# Check if libsmctrl.so exists for integration tests
_LIBSMCTRL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "cospec", "csrc", "build", "libsmctrl.so")
_HAS_LIBSMCTRL = os.path.exists(_LIBSMCTRL_PATH)


class TestSMControllerUnit:
    """Unit tests that don't require GPU."""

    def test_c_uint128_zero(self):
        from vllm.cospec.sm_controller import c_uint128
        v = c_uint128(0)
        assert v.value == 0

    def test_c_uint128_large(self):
        from vllm.cospec.sm_controller import c_uint128
        val = (1 << 127) | 42
        v = c_uint128(val)
        assert v.value == val

    def test_default_libsmctrl_path(self):
        from vllm.cospec.sm_controller import _DEFAULT_LIBSMCTRL_PATH
        assert _DEFAULT_LIBSMCTRL_PATH.endswith("libsmctrl.so")


@pytest.mark.skipif(not _HAS_LIBSMCTRL, reason="libsmctrl.so not built")
class TestSMControllerGPU:
    """Integration tests that require GPU and libsmctrl."""

    @pytest.fixture
    def controller(self):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from vllm.cospec.sm_controller import SMController
        return SMController(libsmctrl_path=_LIBSMCTRL_PATH, is_target=True)

    def test_init(self, controller):
        assert controller.total_tpcs > 0
        assert controller.is_target is True

    def test_set_full_gpu(self, controller):
        import torch
        stream = torch.cuda.current_stream()
        try:
            controller.set_full_gpu(stream)
        except RuntimeError as e:
            if "MPS not available" in str(e):
                pytest.skip("libsmctrl requires MPS or elevated privileges")
            raise

    def test_set_partition(self, controller):
        import torch
        stream = torch.cuda.current_stream()
        try:
            controller.set_partition(stream, 0.7)
        except RuntimeError as e:
            if "MPS not available" in str(e):
                pytest.skip("libsmctrl requires MPS or elevated privileges")
            raise

    def test_set_partition_explicit_stream(self, controller):
        """Test with a non-default CUDA stream (uses stream mask, not global)."""
        import torch
        stream = torch.cuda.Stream()
        try:
            controller.set_partition(stream, 0.7)
        except RuntimeError as e:
            if "MPS not available" in str(e):
                pytest.skip("libsmctrl requires MPS or elevated privileges")
            raise
