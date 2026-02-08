import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file without using first row as header
df = pd.read_csv('kv_size_analysis.csv', header=None)

# Extract the data we need
models = df.iloc[0, 1:].values  # Model names from first row
kv_size_per_req_bytes = df.iloc[8, 1:].values.astype(float)  # KV size per request in bytes
context_lengths = df.iloc[7, 1:].values.astype(int)  # Context lengths

# Convert bytes to GB
kv_size_per_req_gb = kv_size_per_req_bytes / (1024**3)

# Batch sizes to plot
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

# Create the plot
plt.figure(figsize=(8, 4))

# Colors for different models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# Plot for each model configuration
for i, (model, kv_size_gb, context_len) in enumerate(zip(models, kv_size_per_req_gb, context_lengths)):
    # Calculate KV cache size for each batch size
    kv_cache_sizes = [kv_size_gb * batch_size for batch_size in batch_sizes]
    
    # Create label with model and context length
    model_str = str(model)  # Convert to string first
    if "8B" in model_str:
        model_name = "Llama 3.1 8B"
    else:
        continue
    # elif "70B" in model_str:
    #     model_name = "Llama 3.1 70B"
    # elif "405B" in model_str:
    #     model_name = "Llama 3.1 405B"
    # else:
    #     model_name = model_str
    
    # Format context length
    if context_len >= 1000000:
        context_str = f"{context_len//1000000}M"
    elif context_len >= 1000:
        context_str = f"{context_len//1000}K"
    else:
        context_str = str(context_len)
    
    # Create label - show full model name for 2K, just context length for others
    if context_len == 2000:
        label = f"Llama-3.1-8B (Context Length={context_str})"
    else:
        label = f"{context_str}"
    
    plt.plot(range(len(batch_sizes)), kv_cache_sizes, marker='o', linewidth=2, markersize=6, 
             color=colors[i % len(colors)], label=label)

# Add horizontal dotted lines for GPU memory limits
gpu_limits = {
    'A100 (80GB)': 80,
    'B200 (180GB)': 180
}

# Colors for GPU limit lines
gpu_colors = ['#D32F2F', '#7B1FA2']  # Professional red and purple

for i, (gpu_name, memory_gb) in enumerate(gpu_limits.items()):
    plt.axhline(y=memory_gb, color=gpu_colors[i], linestyle='--', linewidth=2)

# Set log scale for y-axis
plt.yscale('log')

# Customize the plot
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('KV Cache Memory Size (GB)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=12, frameon=False)

# Set x-axis ticks with equal spacing
plt.xticks(range(len(batch_sizes)), batch_sizes)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('kv_cache_memory_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
