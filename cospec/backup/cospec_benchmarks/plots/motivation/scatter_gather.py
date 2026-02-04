import matplotlib.pyplot as plt
import numpy as np


# Data from test results 1 (cuMemcpyAsync)
chunk_sizes = ["1KB", "2KB", "4KB", "8KB", "16KB", "32KB", "64KB", "128KB", "256KB", "512KB", "1MB"]
transfer_times_1 = [
    180.353022850,
    85.654575960,
    44.740507130,
    24.238608510,
    14.658008760,
    7.118350980,
    4.180063230,
    2.710710950,
    1.980685080,
    1.615810980,
    1.434858720
]
bandwidth_1 = [
    0.177429796,
    0.373593584,
    0.715235523,
    1.320207799,
    2.183106896,
    4.495423180,
    7.655386591,
    11.805021114,
    16.156026177,
    19.804296663,
    22.301847251
]

# Data from test results 2 (cuMemcpyBatchAsync)
transfer_times_2 = [
    39.3346000,
    19.2291000,
    9.5235000,
    4.8718000,
    2.7226000,
    1.3831000,
    1.2858000,
    1.2650000,
    1.2547000,
    1.2497000,
    1.2471000
]
bandwidth_2 = [
    0.7945,
    1.6251,
    3.2814,
    6.4145,
    11.4780,
    22.5942,
    24.3039,
    24.7036,
    24.9064,
    25.0060,
    25.0581
]


# Ensure both result lists have the same length as chunk_sizes
if not (len(transfer_times_2) == len(chunk_sizes) and len(bandwidth_2) == len(chunk_sizes)):
    raise ValueError("The length of the new result lists must be the same as chunk_sizes")

# positions for the bars
x = np.arange(len(chunk_sizes))
width = 0.35  # width of the bars

# Define colors
color1 = '#FEAE00'  # Orange
color2 = '#659250'  # Green

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Add main title for the entire figure
fig.suptitle('H2D Transfer Performance for 32MB Data', fontsize=14, fontweight='bold', y=1.05)

# Left subplot: Bandwidth vs Chunk Size
rects1_bw = ax1.bar(x - width/2, bandwidth_1, width, label='cuMemcpyAsync', color=color1, edgecolor='black', alpha=0.8)
rects2_bw = ax1.bar(x + width/2, bandwidth_2, width, label='cuMemcpyBatchAsync (Scatter-Gather DMA)', color=color2, edgecolor='black', alpha=0.8)

ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax1.set_xlabel('Chunk Size', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(chunk_sizes, rotation=45, ha='center')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Right subplot: Transfer Time vs Chunk Size
rects1_tt = ax2.bar(x - width/2, transfer_times_1, width, label='cuMemcpyAsync', color=color1, edgecolor='black', alpha=0.8)
rects2_tt = ax2.bar(x + width/2, transfer_times_2, width, label='cuMemcpyBatchAsync (Scatter-Gather DMA)', color=color2, edgecolor='black', alpha=0.8)

ax2.set_ylabel('Total Latency (ms)', fontsize=12)
ax2.set_xlabel('Chunk Size', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(chunk_sizes, rotation=45, ha='center')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Create a single legend at the top
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=12)

# Function to add value labels on top of bars
def autolabel_bw(rects1, rects2, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    max_val_bw = max(max(bandwidth_1), max(bandwidth_2))
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + (max_val_bw * 0.02), f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)

    for rect in rects2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + (max_val_bw * 0.02), f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)

def autolabel_tt(rects1, rects2, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    max_val_tt = max(max(transfer_times_1), max(transfer_times_2))
    for rect in rects1:
        height = rect.get_height()
        if height > 100:  # For large values, put label inside
            ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{height:.1f}',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(rect.get_x() + rect.get_width() / 2, height + (max_val_tt * 0.02), f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    for rect in rects2:
        height = rect.get_height()
        if height > 100:  # For large values, put label inside
            ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{height:.1f}',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(rect.get_x() + rect.get_width() / 2, height + (max_val_tt * 0.02), f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

# Add value labels
# autolabel_bw(rects1_bw, rects2_bw, ax1)
# autolabel_tt(rects1_tt, rects2_tt, ax2)

plt.tight_layout()
plt.savefig('h2d_comparison_combined_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
