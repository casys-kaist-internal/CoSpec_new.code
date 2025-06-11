import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the token latencies from the JSON file
with open("token_latencies_without_sv.json", "r") as f:
    token_latencies = json.load(f)

# Convert to milliseconds and filter out zeros/failed requests
latencies_ms = [lat * 1000 for lat in token_latencies if lat > 0]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Token Latency Distribution Analysis', fontsize=16)

# Plot 1: Histogram with KDE
sns.histplot(data=latencies_ms, bins=50, kde=True, ax=ax1)
ax1.set_title('Histogram with Kernel Density Estimate')
ax1.set_xlabel('Latency per Token (ms)')
ax1.set_ylabel('Count')

# Add percentile markers
percentiles = [50, 90, 95, 99]
colors = ['g', 'y', 'orange', 'r']
for p, color in zip(percentiles, colors):
    percentile = np.percentile(latencies_ms, p)
    ax1.axvline(percentile, color=color, linestyle='--', alpha=0.7)
    ax1.text(percentile, ax1.get_ylim()[1]*0.9, 
             f'P{p}: {percentile:.2f}ms',
             color=color, rotation=90)

# Plot 2: Box plot
sns.boxplot(x=latencies_ms, ax=ax2)
ax2.set_title('Box Plot')
ax2.set_xlabel('Latency per Token (ms)')

# Add statistical summary
stats_text = f"""
Statistical Summary:
Mean: {np.mean(latencies_ms):.2f} ms
Median: {np.median(latencies_ms):.2f} ms
Std Dev: {np.std(latencies_ms):.2f} ms
Min: {np.min(latencies_ms):.2f} ms
Max: {np.max(latencies_ms):.2f} ms
"""
ax2.text(0.02, 0.98, stats_text,
         transform=ax2.transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout and display
plt.tight_layout()
plt.savefig("token_latencies_tile.png")

# Print additional statistics
print("\nDetailed Statistics:")
print(f"Total number of requests: {len(latencies_ms)}")
print(f"Number of failed requests: {len(token_latencies) - len(latencies_ms)}")
print("\nPercentiles:")
for p in [50, 75, 90, 95, 99, 99.9]:
    print(f"P{p}: {np.percentile(latencies_ms, p):.2f} ms")