import glob
import os
import shutil
from typing import Optional

from vllm.config import VllmConfig
from vllm.cospec.sm_controller import SMController
from vllm.logger import init_logger

logger = init_logger(__name__)


def cleanup_cospec_resources() -> None:
    """Remove stale CoSpec IPC handles from shared memory and temp dirs."""
    count = 0
    for pattern in ['/dev/shm/cospec_*', '/tmp/cospec_*']:
        for p in glob.glob(pattern):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.isfile(p):
                    os.remove(p)
                count += 1
            except Exception as e:
                logger.warning("Failed to remove %s: %s", p, e)
    if count:
        logger.info("Cleaned up %d CoSpec IPC entries", count)


class CospecManager:
    """Central coordinator for CoSpec v2.

    Uses SM partitioning via libsmctrl instead of file locks.
    The SMController manages GPU resource allocation between
    target and draft processes running concurrently via MPS.
    """

    def __init__(self, vllm_config: VllmConfig):
        # Clean stale IPC handles from previous runs
        cleanup_cospec_resources()
        self.rank = vllm_config.parallel_config.rank
        self.is_primary = vllm_config.speculative_config.is_primary
        self.is_driver = self.rank == 0
        self.total_ranks = vllm_config.parallel_config.world_size

        # SM partitioning controller
        is_target = self.is_primary
        self.sm_controller = SMController(is_target=is_target)
        self.target_sm_ratio: float = 1.0  # default: full GPU

        logger.info("CospecManager initialized: is_target=%s, rank=%d",
                     is_target, self.rank)
