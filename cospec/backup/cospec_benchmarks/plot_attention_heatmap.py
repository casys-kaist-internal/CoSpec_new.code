import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Model configuration
NUM_HEADS = (40, 40)  # OPT-13B configuration
HEAD_SIZE = 128

# Batch sizes and target average query lengths from the benchmark
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
target_avg_query_lens = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]

# Create a matrix to store speedup values
speedup_matrix = np.zeros((len(target_avg_query_lens), len(batch_sizes)))

# Read the benchmark results from the terminal output
# You'll need to manually copy the speedup values from the terminal output
# and fill them in the matrix below
# Example format:
# speedup_matrix = [
#     [1.234, 1.345, ...],  # for target_avg = 1.0
#     [1.345, 1.456, ...],  # for target_avg = 1.1
#     ...
# ]

# Create a DataFrame for better visualization
df = pd.DataFrame(speedup_matrix, 
                 index=[f"{x:.1f}" for x in target_avg_query_lens],
                 columns=[f"{x}" for x in batch_sizes])

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd", 
            cbar_kws={'label': 'Speedup (Normal/Consolidated)'})

# Customize the plot
plt.title(f'Attention Speedup Heatmap\n(NUM_HEADS={NUM_HEADS}, HEAD_SIZE={HEAD_SIZE})')
plt.xlabel('Batch Size')
plt.ylabel('Target Average Query Length')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('attention_speedup_heatmap.png', dpi=300, bbox_inches='tight')
plt.close() 