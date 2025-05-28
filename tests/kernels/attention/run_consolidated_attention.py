import torch
import random
import time
import numpy as np
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

# Constants
NUM_BLOCKS = 4321
PARTITION_SIZE = 512
NUM_GEN_SEQS = 64
NUM_HEADS = (20, 20)
HEAD_SIZE = 192
BLOCK_SIZE = 16
DTYPE = torch.half
KV_CACHE_DTYPE = "auto"
SEED = 0
DEVICE = "cuda:0"
QUERY_SIZE = 8
VERSION = "v1"
NUM_WARMUP = 3
NUM_ITERATIONS = 10

def kv_cache_factory(num_blocks, block_size, num_layers, num_kv_heads, head_size, 
                    kv_cache_dtype, dtype, seed, device):
    """Create key and value caches for testing."""
    key_caches = []
    value_caches = []
    
    for _ in range(num_layers):
        key_cache = torch.empty(
            (num_blocks, num_kv_heads, head_size, block_size),
            dtype=dtype,
            device=device
        )
        value_cache = torch.empty(
            (num_blocks, num_kv_heads, head_size, block_size),
            dtype=dtype,
            device=device
        )
        
        # Initialize with random values
        key_cache.uniform_(-1.0, 1.0)
        value_cache.uniform_(-1.0, 1.0)
        
        key_caches.append(key_cache)
        value_caches.append(value_cache)
    
    return key_caches, value_caches

def generate_random_query_lengths(num_seqs, target_avg):
    """Generate random query lengths with specified average using weighted distribution."""
    # For fractional averages, we need to use two adjacent integers
    lower = int(target_avg)
    upper = min(lower + 1, 8)  # Don't exceed max length of 8
    
    if lower == upper:  # If target is an integer
        return [lower] * num_seqs
    
    # Calculate weights to achieve exact average
    # For example, if target is 1.5, we need 50% 1s and 50% 2s
    lower_weight = upper - target_avg
    upper_weight = target_avg - lower
    
    # Calculate number of each value needed
    num_lower = int(round(num_seqs * lower_weight))
    num_upper = num_seqs - num_lower
    
    # Generate the sequence
    lengths = [lower] * num_lower + [upper] * num_upper
    
    # Shuffle the sequence to randomize the order
    np.random.shuffle(lengths)
    
    return lengths

def run_attention_test(num_gen_seqs, target_avg_query_len):
    """Run attention test with specific batch size and target average query length."""
    # Set random seed
    current_platform.seed_everything(SEED)
    torch.set_default_device(DEVICE)
    
    # Calculate scale
    scale = float(1.0 / (HEAD_SIZE**0.5))
    num_query_heads, num_kv_heads = NUM_HEADS
    
    # Create query tensor
    query = torch.empty(num_gen_seqs, num_query_heads, HEAD_SIZE, dtype=DTYPE)
    query.uniform_(-scale, scale)
    
    # Generate sequence lengths and query lengths
    seq_lens = []
    query_lens = generate_random_query_lengths(num_gen_seqs, target_avg_query_len)
    # print("target_avg", target_avg_query_len)
    # print("query_lens", query_lens)
    actual_avg = sum(query_lens) / len(query_lens)
    
    for query_len in query_lens:
        start = 800  # Using 1000 as max_seq_len
        for query_idx in range(query_len):
            seq_lens.append(start + query_idx)
    
    # Repeat query based on query lengths
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
    
    # Create block tables
    max_num_blocks_per_seq = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables_lst = []
    
    for seq_idx in range(num_gen_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        for query_idx in range(query_lens[seq_idx]):
            block_tables_lst.append(block_table)
    
    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)
    
    # Create KV caches
    key_caches, value_caches = kv_cache_factory(
        NUM_BLOCKS, BLOCK_SIZE, 1, num_kv_heads, HEAD_SIZE,
        KV_CACHE_DTYPE, DTYPE, SEED, DEVICE
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    
    # Using default kv_scale
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    
    # Call the paged attention kernel
    output = torch.empty_like(query)
    ref_output = torch.empty_like(query)
    
    # Warmup
    for _ in range(NUM_WARMUP):
        if VERSION == "v1":
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
                BLOCK_SIZE,
                max_seq_len,
                None,  # alibi_slopes
                KV_CACHE_DTYPE,
                k_scale,
                v_scale,
            )
            ops.paged_attention_v1(
                ref_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                BLOCK_SIZE,
                max_seq_len,
                None,  # alibi_slopes
                KV_CACHE_DTYPE,
                k_scale,
                v_scale,
            )
        elif VERSION == "v2":
            num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
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
                BLOCK_SIZE,
                max_seq_len,
                None,
                KV_CACHE_DTYPE,
                k_scale,
                v_scale,
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
                BLOCK_SIZE,
                max_seq_len,
                None,
                KV_CACHE_DTYPE,
                k_scale,
                v_scale,
            )
    
    # Measure consolidated attention latency
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(NUM_ITERATIONS):
        if VERSION == "v1":
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
                BLOCK_SIZE,
                max_seq_len,
                None,  # alibi_slopes
                KV_CACHE_DTYPE,
                k_scale,
                v_scale,
            )
        elif VERSION == "v2":
            num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
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
                BLOCK_SIZE,
                max_seq_len,
                None,
                KV_CACHE_DTYPE,
                k_scale,
                v_scale,
            )
    
    torch.cuda.synchronize()
    end_time = time.time()
    consolidated_latency = (end_time - start_time) / NUM_ITERATIONS * 1000  # Convert to milliseconds

    # Measure normal attention latency
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(NUM_ITERATIONS):
        if VERSION == "v1":
            ops.paged_attention_v1(
                ref_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                BLOCK_SIZE,
                max_seq_len,
                None,  # alibi_slopes
                KV_CACHE_DTYPE,
                k_scale,
                v_scale,
            )
        elif VERSION == "v2":
            num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
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
                BLOCK_SIZE,
                max_seq_len,
                None,
                KV_CACHE_DTYPE,
                k_scale,
                v_scale,
            )
    
    torch.cuda.synchronize()
    end_time = time.time()
    normal_latency = (end_time - start_time) / NUM_ITERATIONS * 1000  # Convert to milliseconds

    return consolidated_latency, normal_latency, actual_avg

def main():
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    target_avg_query_lens = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    
    # Print header for consolidated attention
    print("\nConsolidated Attention Latency (ms)")
    print("=" * 80)
    print("Target Avg | Actual Avg |", end="")
    for batch_size in batch_sizes:
        print(f" Batch {batch_size:^4} |", end="")
    print("\n" + "-" * 80)
    
    # Print consolidated attention results
    for target_avg in target_avg_query_lens:
        print(f"{target_avg:^10} |", end="")
        # Get actual average from first batch size (they should be similar across batch sizes)
        _, _, actual_avg = run_attention_test(batch_sizes[0], target_avg)
        print(f" {actual_avg:^10.4f} |", end="")
        for batch_size in batch_sizes:
            consolidated_latency, _, _ = run_attention_test(batch_size, target_avg)
            print(f" {consolidated_latency:^10.3f} |", end="")
        print()  # New line after each query size
    
    print("=" * 80)

    # Print header for normal attention
    print("\nNormal Attention Latency (ms)")
    print("=" * 80)
    print("Target Avg | Actual Avg |", end="")
    for batch_size in batch_sizes:
        print(f" Batch {batch_size:^4} |", end="")
    print("\n" + "-" * 80)
    
    # Print normal attention results
    for target_avg in target_avg_query_lens:
        print(f"{target_avg:^10} |", end="")
        # Get actual average from first batch size (they should be similar across batch sizes)
        _, _, actual_avg = run_attention_test(batch_sizes[0], target_avg)
        print(f" {actual_avg:^10.4f} |", end="")
        for batch_size in batch_sizes:
            _, normal_latency, _ = run_attention_test(batch_size, target_avg)
            print(f" {normal_latency:^10.3f} |", end="")
        print()  # New line after each query size
    
    print("=" * 80)

    # Print speedup comparison
    print("\nSpeedup (Normal/Consolidated)")
    print("=" * 80)
    print("Target Avg | Actual Avg |", end="")
    for batch_size in batch_sizes:
        print(f" Batch {batch_size:^4} |", end="")
    print("\n" + "-" * 80)
    
    # Print speedup results
    for target_avg in target_avg_query_lens:
        print(f"{target_avg:^10} |", end="")
        # Get actual average from first batch size (they should be similar across batch sizes)
        _, _, actual_avg = run_attention_test(batch_sizes[0], target_avg)
        print(f" {actual_avg:^10.4f} |", end="")
        for batch_size in batch_sizes:
            consolidated_latency, normal_latency, _ = run_attention_test(batch_size, target_avg)
            speedup = normal_latency / consolidated_latency
            print(f" {speedup:^10.3f} |", end="")
        print()  # New line after each query size
    
    print("=" * 80)

if __name__ == "__main__":
    main()