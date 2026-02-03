"""SM Controller: ctypes wrapper around libsmctrl for SM partitioning.

Allows target and draft processes to run concurrently on the same GPU
with configurable SM (streaming multiprocessor) partitioning via MPS.
"""

import ctypes
import os
from functools import wraps
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Default path to libsmctrl.so relative to the CoSpec csrc build directory
_DEFAULT_LIBSMCTRL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "cospec", "csrc", "build", "libsmctrl.so")


def _check_ret_code(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        rets = func(*args, **kwargs)
        if isinstance(rets, tuple):
            ret_code = rets[0]
            ret_res = rets[1]
        else:
            ret_code = rets or 0
            ret_res = None
        if ret_code != 0:
            raise OSError(ret_code,
                          f"{os.strerror(ret_code)} in {func.__name__}")
        return ret_res
    return wrapper


class c_uint128(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_uint64)]

    def __init__(self, value=0):
        super().__init__()
        if isinstance(value, int):
            self.low = value & 0xFFFFFFFFFFFFFFFF
            self.high = (value >> 64) & 0xFFFFFFFFFFFFFFFF
        else:
            self.low = 0
            self.high = 0

    @property
    def value(self):
        return self.low | (self.high << 64)


class _LibSMCtrl:
    """Low-level ctypes bindings to libsmctrl."""

    def __init__(self, libsmctrl_path: str):
        device_props = torch.cuda.get_device_properties(
            torch.cuda.current_device())
        self.total_sms = device_props.multi_processor_count
        try:
            self.lib = ctypes.CDLL(libsmctrl_path)
        except Exception as e:
            raise OSError(
                f"Failed to load libsmctrl.so from {libsmctrl_path}: {e}. "
                "Build it with: cd cospec/csrc && mkdir -p build && cd build "
                "&& cmake .. && make")

        # Set return types for ctypes functions.
        # libsmctrl_set_global_mask and _ext are void - ctypes defaults to
        # c_int which reads garbage from memory, causing spurious errors.
        self.lib.libsmctrl_set_global_mask.restype = None
        self.lib.libsmctrl_set_global_mask.argtypes = [ctypes.c_uint64]
        # The _ext version also returns void
        if hasattr(self.lib, 'libsmctrl_set_global_mask_ext'):
            self.lib.libsmctrl_set_global_mask_ext.restype = None
            self.lib.libsmctrl_set_global_mask_ext.argtypes = [c_uint128]
        # Stream mask functions return int
        self.lib.libsmctrl_set_stream_mask.restype = ctypes.c_int
        self.lib.libsmctrl_set_stream_mask.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64]
        if hasattr(self.lib, 'libsmctrl_set_stream_mask_ext'):
            self.lib.libsmctrl_set_stream_mask_ext.restype = ctypes.c_int
            self.lib.libsmctrl_set_stream_mask_ext.argtypes = [
                ctypes.c_void_p, c_uint128]

    @_check_ret_code
    def set_global_mask(self, mask: int) -> None:
        if self.total_sms >= 128:
            raise ValueError(
                f"total_sms {self.total_sms} >= 128, use stream mask instead")
        return self.lib.libsmctrl_set_global_mask(ctypes.c_uint64(mask))

    @_check_ret_code
    def set_stream_mask(self, stream: torch.cuda.Stream, mask: int) -> None:
        stream_ptr = stream.cuda_stream
        if stream_ptr == 0:
            # Default stream (null handle) — use global mask instead
            if self.total_sms < 128:
                return self.lib.libsmctrl_set_global_mask(
                    ctypes.c_uint64(mask))
            else:
                return self.lib.libsmctrl_set_global_mask_ext(c_uint128(mask))
        if self.total_sms < 128:
            return self.lib.libsmctrl_set_stream_mask(
                ctypes.c_void_p(stream_ptr), ctypes.c_uint64(mask))
        else:
            return self.lib.libsmctrl_set_stream_mask_ext(
                ctypes.c_void_p(stream_ptr), c_uint128(mask))

    @_check_ret_code
    def get_tpc_count(self, cuda_dev: int) -> int:
        num_tpcs = ctypes.c_uint32()
        ret = self.lib.libsmctrl_get_tpc_info_cuda(
            ctypes.byref(num_tpcs), cuda_dev)
        return ret, num_tpcs.value

    @_check_ret_code
    def make_mask(self, low: int, high_exclusive: int) -> int:
        result = ctypes.c_uint64()
        ret = self.lib.libsmctrl_make_mask(
            ctypes.byref(result), low, high_exclusive)
        return ret, result.value


class SMController:
    """High-level SM partitioning controller for CoSpec v2.

    Manages SM allocation between target and draft processes using
    libsmctrl stream masks. Each process creates its own SMController
    and calls set_partition() or set_full_gpu() as directed by the
    orchestrator.

    Args:
        libsmctrl_path: Path to libsmctrl.so shared library.
        is_target: Whether this controller is for the target process.
    """

    def __init__(self, libsmctrl_path: Optional[str] = None,
                 is_target: bool = True):
        if libsmctrl_path is None:
            libsmctrl_path = _DEFAULT_LIBSMCTRL_PATH
        self.is_target = is_target
        self._lib = _LibSMCtrl(libsmctrl_path)
        self.total_tpcs = self._lib.get_tpc_count(
            torch.cuda.current_device())
        logger.info("SMController initialized: total_tpcs=%d, is_target=%s",
                     self.total_tpcs, is_target)

    def set_partition(self, stream: torch.cuda.Stream, ratio: float) -> None:
        """Partition SMs for this process.

        Args:
            stream: CUDA stream to apply the mask to.
            ratio: Fraction of TPCs to allocate to this process (0.0-1.0).
                For target process, allocates TPCs from the bottom.
                For draft process, allocates TPCs from the top.
        """
        num_tpcs = max(1, int(self.total_tpcs * ratio))
        if self.is_target:
            low, high = 0, num_tpcs
        else:
            low, high = self.total_tpcs - num_tpcs, self.total_tpcs
        mask = self._lib.make_mask(low, high)
        try:
            self._lib.set_stream_mask(stream, mask)
        except PermissionError as e:
            raise RuntimeError(
                "SMController: set_partition failed - MPS not available. "
                "CoSpec requires NVIDIA MPS for SM partitioning. "
                "Start MPS with: bash cospec/scripts/start_mps.sh"
            ) from e
        logger.debug("SMController partition: tpcs [%d, %d) for %s",
                     low, high, "target" if self.is_target else "draft")

    def set_full_gpu(self, stream: torch.cuda.Stream) -> None:
        """Give this process access to all SMs.

        Args:
            stream: CUDA stream to apply the full mask to.
        """
        mask = self._lib.make_mask(0, self.total_tpcs)
        try:
            self._lib.set_stream_mask(stream, mask)
        except PermissionError as e:
            raise RuntimeError(
                "SMController: set_full_gpu failed - MPS not available. "
                "CoSpec requires NVIDIA MPS for SM partitioning. "
                "Start MPS with: bash cospec/scripts/start_mps.sh"
            ) from e
        logger.debug("SMController full GPU for %s",
                     "target" if self.is_target else "draft")


class CospecManager:
    """Central coordinator for CoSpec v2.

    Creates and holds the SMController for SM partitioning between
    target and draft processes running concurrently via MPS.
    """

    def __init__(self, vllm_config):
        from vllm.cospec import cleanup_cospec_resources
        cleanup_cospec_resources()

        self.rank = vllm_config.parallel_config.rank
        self.is_primary = vllm_config.speculative_config.is_primary
        self.is_driver = self.rank == 0

        is_target = self.is_primary
        self.sm_controller = SMController(is_target=is_target)
        self.target_sm_ratio: float = 1.0  # default: full GPU

        logger.info("CospecManager initialized: is_target=%s, rank=%d",
                    is_target, self.rank)
