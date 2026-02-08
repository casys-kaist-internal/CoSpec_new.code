import matplotlib.pyplot as plt
import numpy as np

# Data from test results 2 (cuMemcpyBatchAsync) - only selected chunk sizes
chunk_sizes = ["4KB", "8KB", "16KB", "32KB"]
bandwidth_2 = [
    3.2814,
    6.4145,
    11.4780,
    22.5942
]

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

# Define color
color = '#659250'  # Green

# Create bar plot
rects = ax.bar(chunk_sizes, bandwidth_2, color=color, edgecolor='black', alpha=0.8, width=0.6)

# Set labels and title
ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax.set_xlabel('Chunk Size', fontsize=12)
ax.set_title('cuMemcpyBatchAsync (Scatter-Gather DMA)', fontsize=14, fontweight='bold')

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# y lim auto
ax.set_ylim(0, 28)

# Function to add value labels on top of bars
def autolabel(rects):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + (max(bandwidth_2) * 0.02), f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add value labels
autolabel(rects)

plt.tight_layout()
plt.savefig('h2d_bandwidth_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
