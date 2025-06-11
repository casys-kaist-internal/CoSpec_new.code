import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Read the data
df = pd.read_csv('tiling_results.csv')

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
# at 192 
latency_192 = df[df['num_tokens'] == 192]['mean_latency'].values[0]
plt.plot(192, latency_192, 'o', markersize=5, color='red')
plt.annotate(f'{latency_192:.1f}ms', 
            xy=(192, latency_192),
            xytext=(-10, -12),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold')

# at 200
latency_200 = df[df['num_tokens'] == 200]['mean_latency'].values[0]
slowdown = latency_200 / latency_192
plt.plot(200, latency_200, 'o', markersize=5, color='red')
plt.annotate(f'{latency_200:.1f}ms', 
            xy=(200, latency_200),
            xytext=(-15, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold')
# Add arrow and slowdown text
plt.annotate('', 
            xy=(200, latency_200),
            xytext=(200, latency_192 - 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
# Add horizontal line
plt.plot([195, 205], [latency_192, latency_192], color='red', lw=1)
plt.annotate(f'{(slowdown-1)*100:.0f}% slowdown',
            xy=(200, (latency_192 + latency_200)/2 - 2),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color='red')

# at 384
latency_384 = df[df['num_tokens'] == 384]['mean_latency'].values[0]
plt.plot(384, latency_384, 'o', markersize=5, color='red')
plt.annotate(f'{latency_384:.1f}ms', 
            xy=(384, latency_384),
            xytext=(-10, -12),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold')

# at 392
latency_392 = df[df['num_tokens'] == 392]['mean_latency'].values[0]
slowdown_384_392 = latency_392 / latency_384
plt.plot(392, latency_392, 'o', markersize=5, color='red')
plt.annotate(f'{latency_392:.1f}ms', 
            xy=(392, latency_392),
            xytext=(-17, 7),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold')
# Add arrow and slowdown text
plt.annotate('', 
            xy=(392, latency_392),
            xytext=(392, latency_384 - 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
# Add horizontal line
plt.plot([387, 397], [latency_384, latency_384], color='red', lw=1)
plt.annotate(f'{(slowdown_384_392-1)*100:.0f}% slowdown',
            xy=(392, (latency_384 + latency_392)/2 - 1),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color='red')

# Customize the plot
plt.xlabel('Number of Tokens', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)

# Set x-axis ticks and grid at multiples of 64
max_tokens = df['num_tokens'].max()
xticks = np.arange(0, max_tokens + 64, 64)
plt.xticks(xticks)
plt.grid(True, linestyle='--', alpha=0.3, which='major')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('tiling_effect.pdf', dpi=300, bbox_inches='tight')
plt.savefig('tiling_effect.png', dpi=300, bbox_inches='tight')
