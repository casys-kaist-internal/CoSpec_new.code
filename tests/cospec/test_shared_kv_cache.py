"""Unit tests for CoSpec v2 SharedKVCacheAllocator."""

import os
import shutil

import pytest


@pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="No GPU available")
class TestSharedKVCache:

    def test_owner_allocate_and_cleanup(self):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from vllm.cospec.shared_kv_cache import SharedKVCacheAllocator

        allocator = SharedKVCacheAllocator(mode="owner",
                                            instance_id="test_unit")
        try:
            kv_cache = allocator.allocate_shared(
                kv_cache_shape=(4, 16, 8, 64),
                dtype=torch.float16,
                num_layers=2,
                device="cuda",
            )
            assert len(kv_cache) == 2
            assert kv_cache[0].shape == (4, 16, 8, 64)
            assert kv_cache[0].dtype == torch.float16
            assert kv_cache[0].is_cuda
        finally:
            allocator.cleanup()

    def test_client_without_owner_raises(self):
        from vllm.cospec.shared_kv_cache import SharedKVCacheAllocator

        # Clean up any stale files
        shm_dir = "/dev/shm/cospec_kv_cache_test_noowner"
        if os.path.exists(shm_dir):
            shutil.rmtree(shm_dir)

        allocator = SharedKVCacheAllocator(mode="client",
                                            instance_id="test_noowner")
        with pytest.raises(FileNotFoundError):
            allocator.allocate_shared(
                kv_cache_shape=(4, 16, 8, 64),
                dtype=None,  # unused by client
                num_layers=1,
            )
