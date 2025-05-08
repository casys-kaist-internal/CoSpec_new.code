# SPDX-License-Identifier: Apache-2.0

import random
from typing import Optional

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils import get_max_shared_memory_bytes

if not current_platform.is_rocm():
    from xformers import ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

    from vllm.attention.backends.xformers import _make_alibi_bias

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 4321  # Arbitrary values for testing
PARTITION_SIZE = 512
PARTITION_SIZE_ROCM = 256
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
# DTYPES = [
#     torch.half, torch.bfloat16, torch.float
# ] if not current_platform.is_rocm() else [torch.half, torch.bfloat16]
DTYPES = [
    torch.half, torch.bfloat16
] if not current_platform.is_rocm() else [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# This should be sync with get_supported_head_sizes() in
# vllm.attention.ops.paged_attn.PagedAttention
# HEAD_SIZES = [32, 64, 80, 96, 112, 120, 128, 192, 256]
HEAD_SIZES = [112]

# BLOCK_SIZES = [16, 32]
BLOCK_SIZES = [16]

USE_ALIBI = [False, True]
# KV_CACHE_DTYPE = ["auto", "fp8"]
KV_CACHE_DTYPE = ["auto"]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

QUERY_SIZE = 8

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: list[torch.Tensor] = []
        values_lst: list[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)

@pytest.mark.parametrize(
    "version",
    ["v1", "v2"])
@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_consolidated_paged_attention(
    kv_cache_factory,
    version: str,
    num_seqs: int,
    num_heads: tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    if ((kv_cache_dtype == "fp8" and head_size % 16)
            or (version == "rocm" and head_size not in (64, 128))):
        pytest.skip()

    global PARTITION_SIZE

    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    seq_lens = []
    query_lens = []

    for seq_idx in range(num_seqs):
        query_len = random.randint(2, QUERY_SIZE - 1)
        query_lens.append(query_len)

    for query_len in query_lens:
        start = random.randint(1, 1000 - query_len)
        for query_idx in range(query_len):
            seq_lens.append(start + query_idx)

    # query should be repeated query_len at the first dimension
    repeated_query = []
    query_idx = 0
    for query_len in query_lens:
        for _ in range(query_len):
            repeated_query.append(query[query_idx])
        query_idx += 1
    query = torch.stack(repeated_query)

    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)
    query_lens = torch.tensor(query_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: list[list[int]] = []
    
    for seq_idx in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        for query_idx in range(query_lens[seq_idx]):
            block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v1":
        # Debug: Print input tensors
        print("\nInput tensors:")
        print(f"query shape: {query.shape}")
        print(f"key_cache shape: {key_cache.shape}")
        print(f"value_cache shape: {value_cache.shape}")
        print(f"block_tables shape: {block_tables.shape}")
        print(f"seq_lens shape: {seq_lens.shape}")
        print(f"query_lens shape: {query_lens.shape}")
        
        ops.consolidated_paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            query_lens,
            block_size,
            max_seq_len,
            None,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        # Debug: Print output after consolidated kernel
        print("\nConsolidated kernel output:")
        print(f"output shape: {output.shape}")
        # print(f"output sample values:\n{output[0, 0, :10]}")  # Print first 10 values of first head

    elif version in ("v2"):
        num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
        )
        max_logits = torch.empty_like(exp_sums)

        ops.consolidated_paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            query_lens,
            block_size,
            max_seq_len,
            None,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        raise AssertionError(f"Unknown version: {version}")

    # Run the reference implementation.
    if kv_cache_dtype == "fp8":
        # Convert cache data back to dtype.
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x,
                           block_size, x)
        dequantized_key_cache = torch.empty(size=key_cache_shape,
                                            dtype=dtype,
                                            device=device)
        ops.convert_fp8(dequantized_key_cache, key_cache)
        key_cache = dequantized_key_cache

        value_cache_shape = value_cache.shape
        dequantized_value_cache = torch.empty(size=value_cache_shape,
                                              dtype=dtype,
                                              device=device)
        ops.convert_fp8(dequantized_value_cache, value_cache)
        value_cache = dequantized_value_cache

    ref_output = torch.empty_like(query)
    if version == "v1":
        # Debug: Print reference implementation input
        print("\nReference implementation input:")
        print(f"query shape: {query.shape}")
        print(f"key_cache shape: {key_cache.shape}")
        print(f"value_cache shape: {value_cache.shape}")
        
        ops.paged_attention_v1(
            ref_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            None,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        # Debug: Print reference output
        print("\nReference kernel output:")
        print(f"ref_output shape: {ref_output.shape}")
        print(f"ref_output sample values:\n{ref_output[0, 0, :10]}")  # Print first 10 values of first head
    else:
        num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
        )
        max_logits = torch.empty_like(exp_sums)
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
        )
        ops.paged_attention_v2(
            ref_output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            None,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    atol = get_default_atol(output) if current_platform.is_rocm() else 1e-3
    rtol = get_default_rtol(output) if current_platform.is_rocm() else 1e-5

    # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # so we use a relaxed tolerance for the test.
    atol, rtol = 1e-3, 1e-5
    if kv_cache_dtype == "fp8":
        atol, rtol = 1e-2, 1e-5
    
    # Calculate and print max differences
    abs_diff = torch.abs(output - ref_output)
    rel_diff = abs_diff / (torch.abs(ref_output) + 1e-6)  # Add small epsilon to avoid division by zero
    max_abs_diff = torch.max(abs_diff).item()
    max_rel_diff = torch.max(rel_diff).item()
    print(f"\nMax absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    
    # Print detailed comparison for first few elements
    print("\nDetailed comparison of first few elements:")
    for i in range(min(20, output.shape[0])):
        for j in range(min(1, output.shape[1])):
            for k in range(min(1, output.shape[2])):
                print(f"Element [{i},{j},{k}]:")
                print(f"  Consolidated: {output[i,j,k].item():.6f}")
                print(f"  Reference:    {ref_output[i,j,k].item():.6f}")
                print(f"  Diff:         {abs_diff[i,j,k].item():.6e}")
    
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)