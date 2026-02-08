import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Read the data
df = pd.read_csv('H200_tiling_results.csv')

# Create figure with appropriate size for 2-column paper
# 2-column width is typically around 7.2 inches
plt.figure(figsize=(6, 3))

# Convert latency to milliseconds
df['mean_latency'] = df['mean_latency'] * 1000

# Show until 512
df = df[df['num_tokens'] <= 512]

# Create the plot with a clean style
plt.plot(df['num_tokens'], df['mean_latency'], 'o-', linewidth=1.5, markersize=3, color='#1f77b4')

# Add annotations for specific points
# at 256 
latency_256 = df[df['num_tokens'] == 256]['mean_latency'].values[0]
plt.plot(256, latency_256, 'o', markersize=5, color='red')
plt.annotate(f'{latency_256:.1f}ms', 
            xy=(256, latency_256),
            xytext=(-10, -12),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold')

# at 256 + 8 = 264
latency_264 = df[df['num_tokens'] == 264]['mean_latency'].values[0]
slowdown = latency_264 / latency_256
plt.plot(264, latency_264, 'o', markersize=5, color='red')
plt.annotate(f'{latency_264:.1f}ms', 
            xy=(264, latency_264),
            xytext=(-15, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold')
# Add arrow and slowdown text
plt.annotate('', 
            xy=(264, latency_264),
            xytext=(264, latency_256 - 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
# Add horizontal line
plt.plot([261, 267], [latency_256, latency_256], color='red', lw=1)
plt.annotate(f'{(slowdown-1)*100:.0f}% slowdown',
            xy=(264, (latency_256 + latency_264)/2 - 1),
            xytext=(5, -2),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color='red')

# Customize the plot
plt.xlabel('Number of Tokens', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)

# Title
plt.title('Model: OPT-66B / GPU: H200', fontsize=12, fontweight='bold')

# Set x-axis ticks and grid at multiples of 64
max_tokens = df['num_tokens'].max()
xticks = np.arange(0, max_tokens + 64, 64)
plt.xticks(xticks)
plt.grid(True, linestyle='--', alpha=0.3, which='major')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('tiling_effect_H200.pdf', dpi=300, bbox_inches='tight')
