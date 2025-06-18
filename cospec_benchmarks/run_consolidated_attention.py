import torch
import random
import time
import numpy as np
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Model selection (uncomment one of these)
MODEL = "OPT-6.7B"
# MODEL = "OPT-13B"
# MODEL = "OPT-30B"

# Model configurations
MODEL_CONFIGS = {
    "OPT-6.7B": {
        "NUM_HEADS": (32, 32),
        "HEAD_SIZE": 128
    },
    "OPT-13B": {
        "NUM_HEADS": (40, 40),
        "HEAD_SIZE": 128
    },
    "OPT-30B": {
        "NUM_HEADS": (56, 56),
        "HEAD_SIZE": 128
    }
}

# Get model configuration
NUM_HEADS = MODEL_CONFIGS[MODEL]["NUM_HEADS"]
HEAD_SIZE = MODEL_CONFIGS[MODEL]["HEAD_SIZE"]

# Constants
NUM_BLOCKS = 4321
PARTITION_SIZE = 512
BLOCK_SIZE = 16
DTYPE = torch.half
KV_CACHE_DTYPE = "auto"
SEED = 0
DEVICE = "cuda:0"
QUERY_SIZE = 8
VERSION = "v1"
NUM_WARMUP = 10
NUM_ITERATIONS = 100
SEQ_LEN = 1024 

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

def run_attention_test(num_gen_seqs, target_query_len, version):
    """Run attention test with specific batch size and target query length."""
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
    query_lens = [target_query_len] * num_gen_seqs  # All sequences have the same length
    
    for query_len in query_lens:
        start = SEQ_LEN
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
        if version == "v1":
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
        elif version == "v2":
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
        if version == "v1":
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
        elif version == "v2":
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
        if version == "v1":
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
        elif version == "v2":
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

    return consolidated_latency, normal_latency, target_query_len

def run_benchmark(version):
    """Run benchmark for a specific version."""
    # Test different batch sizes
    batch_sizes = [8 * i for i in range(1, 128//8 + 1)]
    target_query_lens = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Store results in matrices
    consolidated_matrix = np.zeros((len(target_query_lens), len(batch_sizes)))
    normal_matrix = np.zeros((len(target_query_lens), len(batch_sizes)))
    speedup_matrix = np.zeros((len(target_query_lens), len(batch_sizes)))
    
    # Print header for consolidated attention
    print(f"\nBenchmarking {MODEL} with {version}")
    print("\nConsolidated Attention Latency (ms)")
    print("=" * 80)
    print("Query Len |", end="")
    for batch_size in batch_sizes:
        print(f" Batch {batch_size:^4} |", end="")
    print("\n" + "-" * 80)
    
    # Run all tests and store results
    for i, target_len in enumerate(target_query_lens):
        print(f"{target_len:^9} |", end="")
        for j, batch_size in enumerate(batch_sizes):
            consolidated_latency, normal_latency, _ = run_attention_test(batch_size, target_len, version)
            consolidated_matrix[i, j] = consolidated_latency
            normal_matrix[i, j] = normal_latency
            speedup = normal_latency / consolidated_latency
            speedup_matrix[i, j] = speedup
            print(f" {consolidated_latency:^10.3f} |", end="")
        print()  # New line after each query size
    
    print("=" * 80)

    # Print normal attention results
    print("\nNormal Attention Latency (ms)")
    print("=" * 80)
    print("Query Len |", end="")
    for batch_size in batch_sizes:
        print(f" Batch {batch_size:^4} |", end="")
    print("\n" + "-" * 80)
    
    for i, target_len in enumerate(target_query_lens):
        print(f"{target_len:^9} |", end="")
        for j, batch_size in enumerate(batch_sizes):
            print(f" {normal_matrix[i, j]:^10.3f} |", end="")
        print()  # New line after each query size
    
    print("=" * 80)

    # Print speedup results
    print("\nSpeedup (Normal/Consolidated)")
    print("=" * 80)
    print("Query Len |", end="")
    for batch_size in batch_sizes:
        print(f" Batch {batch_size:^4} |", end="")
    print("\n" + "-" * 80)
    
    for i, target_len in enumerate(target_query_lens):
        print(f"{target_len:^9} |", end="")
        for j, batch_size in enumerate(batch_sizes):
            print(f" {speedup_matrix[i, j]:^10.3f} |", end="")
        print()  # New line after each query size
    
    print("=" * 80)

    # Create DataFrames for each metric
    consolidated_df = pd.DataFrame(consolidated_matrix, 
                                 index=[f"{x}" for x in target_query_lens],
                                 columns=[f"{x}" for x in batch_sizes])
    
    normal_df = pd.DataFrame(normal_matrix,
                           index=[f"{x}" for x in target_query_lens],
                           columns=[f"{x}" for x in batch_sizes])
    
    speedup_df = pd.DataFrame(speedup_matrix,
                            index=[f"{x}" for x in target_query_lens],
                            columns=[f"{x}" for x in batch_sizes])

    # Create results directory if it doesn't exist
    os.makedirs('consolidated_attention_results', exist_ok=True)

    # Save DataFrames to CSV files
    model_name = MODEL.replace("-", "_").lower()
    consolidated_df.to_csv(f'consolidated_attention_results/{model_name}_{version}_consolidated_latency.csv')
    normal_df.to_csv(f'consolidated_attention_results/{model_name}_{version}_normal_latency.csv')
    speedup_df.to_csv(f'consolidated_attention_results/{model_name}_{version}_speedup.csv')

def main():
    # Run benchmarks for both v1 and v2
    run_benchmark("v1")
    run_benchmark("v2")

if __name__ == "__main__":
    main()