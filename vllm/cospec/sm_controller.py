"""SM Controller: ctypes wrapper around libsmctrl for SM partitioning.

Allows target and draft processes to run concurrently on the same GPU
with configurable SM (streaming multiprocessor) partitioning via MPS.
"""

import ctypes
import os
from typing import Dict, Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_LIBSMCTRL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "cospec", "csrc", "build", "libsmctrl.so")

_MPS_ERROR = ("CoSpec requires NVIDIA MPS for SM partitioning. "
              "Start MPS with: bash cospec/scripts/start_mps.sh")


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


class SMController:
    """SM partitioning controller for CoSpec.

    Manages SM allocation between target and draft processes using
    libsmctrl stream masks. Each process creates its own SMController.

    Args:
        libsmctrl_path: Path to libsmctrl.so shared library.
        is_target: Whether this controller is for the target process.
    """

    def __init__(self, libsmctrl_path: Optional[str] = None,
                 is_target: bool = True):
        if libsmctrl_path is None:
            libsmctrl_path = _DEFAULT_LIBSMCTRL_PATH
        self.is_target = is_target

        # Load library
        try:
            self._lib = ctypes.CDLL(libsmctrl_path)
        except Exception as e:
            raise OSError(
                f"Failed to load libsmctrl.so from {libsmctrl_path}: {e}. "
                "Build it with: cd cospec/csrc && mkdir -p build && cd build "
                "&& cmake .. && make")

        # GPU info
        device_props = torch.cuda.get_device_properties(
            torch.cuda.current_device())
        self.total_sms = device_props.multi_processor_count

        # Configure ctypes signatures
        # set_global_mask is void (ctypes defaults to c_int, causing garbage)
        self._lib.libsmctrl_set_global_mask.restype = None
        self._lib.libsmctrl_set_global_mask.argtypes = [ctypes.c_uint64]
        if hasattr(self._lib, 'libsmctrl_set_global_mask_ext'):
            self._lib.libsmctrl_set_global_mask_ext.restype = None
            self._lib.libsmctrl_set_global_mask_ext.argtypes = [c_uint128]
        # set_stream_mask returns int (error code)
        self._lib.libsmctrl_set_stream_mask.restype = ctypes.c_int
        self._lib.libsmctrl_set_stream_mask.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64]
        if hasattr(self._lib, 'libsmctrl_set_stream_mask_ext'):
            self._lib.libsmctrl_set_stream_mask_ext.restype = ctypes.c_int
            self._lib.libsmctrl_set_stream_mask_ext.argtypes = [
                ctypes.c_void_p, c_uint128]

        # Get TPC count
        num_tpcs = ctypes.c_uint32()
        ret = self._lib.libsmctrl_get_tpc_info_cuda(
            ctypes.byref(num_tpcs), torch.cuda.current_device())
        if ret != 0:
            raise OSError(ret, f"{os.strerror(ret)} in get_tpc_info_cuda")
        self.total_tpcs = num_tpcs.value

        self._mask_cache: Dict[Tuple[int, int], int] = {}

        logger.info("SMController initialized: total_tpcs=%d, is_target=%s",
                     self.total_tpcs, is_target)

    def _make_mask(self, low: int, high_exclusive: int) -> int:
        """Create a TPC bitmask for the range [low, high_exclusive).

        Results are cached since masks are deterministic for a given range.
        """
        key = (low, high_exclusive)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        result = ctypes.c_uint64()
        ret = self._lib.libsmctrl_make_mask(
            ctypes.byref(result), low, high_exclusive)
        if ret != 0:
            raise OSError(ret, f"{os.strerror(ret)} in make_mask")
        self._mask_cache[key] = result.value
        return result.value

    def _set_stream_mask(self, stream: torch.cuda.Stream, mask: int) -> None:
        """Apply a TPC mask to a CUDA stream."""
        stream_ptr = stream.cuda_stream
        if stream_ptr == 0:
            # Default stream (null handle) — use global mask
            if self.total_sms < 128:
                self._lib.libsmctrl_set_global_mask(ctypes.c_uint64(mask))
            else:
                self._lib.libsmctrl_set_global_mask_ext(c_uint128(mask))
        elif self.total_sms < 128:
            ret = self._lib.libsmctrl_set_stream_mask(
                ctypes.c_void_p(stream_ptr), ctypes.c_uint64(mask))
            if ret != 0:
                raise PermissionError(ret, os.strerror(ret))
        else:
            ret = self._lib.libsmctrl_set_stream_mask_ext(
                ctypes.c_void_p(stream_ptr), c_uint128(mask))
            if ret != 0:
                raise PermissionError(ret, os.strerror(ret))

    def set_partition(self, stream: torch.cuda.Stream, ratio: float) -> None:
        """Partition SMs for this process.

        Args:
            stream: CUDA stream to apply the mask to.
            ratio: Fraction of TPCs to allocate (0.0-1.0).
                Target allocates from the bottom, draft takes the rest.
                Draft's range starts exactly where target ends, so no
                TPCs are left idle due to int() truncation.
        """
        if self.is_target:
            num_tpcs = max(1, int(self.total_tpcs * ratio))
            low, high = 0, num_tpcs
        else:
            # Compute where target ends and take everything above it.
            # target_ratio = 1.0 - draft_ratio
            target_end = max(1, int(self.total_tpcs * (1.0 - ratio)))
            low, high = target_end, self.total_tpcs
        mask = self._make_mask(low, high)
        try:
            self._set_stream_mask(stream, mask)
        except PermissionError as e:
            raise RuntimeError(
                f"SMController: set_partition failed - {_MPS_ERROR}") from e
        logger.debug("SMController partition: tpcs [%d, %d) for %s",
                     low, high, "target" if self.is_target else "draft")

    def set_full_gpu(self, stream: torch.cuda.Stream) -> None:
        """Give this process access to all SMs."""
        mask = self._make_mask(0, self.total_tpcs)
        try:
            self._set_stream_mask(stream, mask)
        except PermissionError as e:
            raise RuntimeError(
                f"SMController: set_full_gpu failed - {_MPS_ERROR}") from e
        logger.debug("SMController full GPU for %s",
                     "target" if self.is_target else "draft")
