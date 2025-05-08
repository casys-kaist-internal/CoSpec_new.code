import torch
import random
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

# Constants
NUM_BLOCKS = 4321
PARTITION_SIZE = 512
NUM_GEN_SEQS = 7
NUM_HEADS = (40, 40)
HEAD_SIZE = 32
BLOCK_SIZE = 16
DTYPE = torch.half
KV_CACHE_DTYPE = "auto"
SEED = 0
DEVICE = "cuda:0"
QUERY_SIZE = 8

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

def main():
    # Set random seed
    current_platform.seed_everything(SEED)
    torch.set_default_device(DEVICE)
    
    # Calculate scale
    scale = float(1.0 / (HEAD_SIZE**0.5))
    num_query_heads, num_kv_heads = NUM_HEADS
    
    # Create query tensor
    query = torch.empty(NUM_GEN_SEQS, num_query_heads, HEAD_SIZE, dtype=DTYPE)
    query.uniform_(-scale, scale)
    
    # Generate sequence lengths and query lengths
    seq_lens = []
    query_lens = []
    
    for seq_idx in range(NUM_GEN_SEQS):
        query_len = random.randint(2, QUERY_SIZE - 1)
        query_lens.append(query_len)
    
    for query_len in query_lens:
        start = random.randint(1, 1000 - query_len)  # Using 1000 as max_seq_len
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
    
    for seq_idx in range(NUM_GEN_SEQS):
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
    
    # Run consolidated attention
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
    
    # all_close 
    assert torch.allclose(output, ref_output, atol=1e-3)
    
if __name__ == "__main__":
    main()