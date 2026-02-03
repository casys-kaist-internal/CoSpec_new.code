"""CoSpec v2: Colocated Speculative Decoding for vLLM.

Runs target and draft processes on the same GPU via MPS with SM partitioning.
"""

import glob
import os
import shutil

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
