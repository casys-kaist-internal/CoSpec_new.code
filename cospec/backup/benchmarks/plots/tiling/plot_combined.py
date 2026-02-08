import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Read the data for all three GPUs
df_a100 = pd.read_csv('A100_tiling_results.csv')
df_a6000 = pd.read_csv('A6000_tiling_results.csv')
df_h200 = pd.read_csv('H200_tiling_results.csv')

# Create figure with 3 subplots in a row - optimized for 2-column paper
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.5))

# Convert latency to milliseconds for all dataframes
df_a100['mean_latency'] = df_a100['mean_latency'] * 1000
df_a6000['mean_latency'] = df_a6000['mean_latency'] * 1000
df_h200['mean_latency'] = df_h200['mean_latency'] * 1000

# Show until 512 for all
df_a100 = df_a100[df_a100['num_tokens'] <= 512]
df_a6000 = df_a6000[df_a6000['num_tokens'] <= 512]
df_h200 = df_h200[df_h200['num_tokens'] <= 512]

# Set x-axis ticks and grid at multiples of 64
max_tokens = 512
xticks = np.arange(0, max_tokens + 64, 64)

# Plot 1: A6000
ax1.plot(df_a6000['num_tokens'], df_a6000['mean_latency'], 'o-', linewidth=1.5, markersize=3, color='#1f77b4')

ax1.set_xlabel('Number of Tokens', fontsize=11)
ax1.set_ylabel('Latency (ms)', fontsize=11)
ax1.set_title('Model: OPT-6.7B / GPU: A6000', fontsize=11, fontweight='bold')
ax1.set_xticks(xticks)
ax1.grid(True, linestyle='--', alpha=0.3, which='major')
ax1.tick_params(axis='both', which='major', labelsize=10)

# Plot 2: A100
ax2.plot(df_a100['num_tokens'], df_a100['mean_latency'], 'o-', linewidth=1.5, markersize=3, color='#1f77b4')

# Add annotations for A100 specific points
# at 128 
latency_128 = df_a100[df_a100['num_tokens'] == 128]['mean_latency'].values[0]
ax2.plot(128, latency_128, 'o', markersize=5, color='red')
ax2.annotate(f'{latency_128:.1f}ms', 
            xy=(128, latency_128),
            xytext=(-10, -12),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold')

# at 128 + 8 = 136
latency_136 = df_a100[df_a100['num_tokens'] == 136]['mean_latency'].values[0]
slowdown = latency_136 / latency_128
ax2.plot(136, latency_136, 'o', markersize=5, color='red')
ax2.annotate(f'{latency_136:.1f}ms', 
            xy=(136, latency_136),
            xytext=(-15, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold')
# Add arrow and slowdown text
ax2.annotate('', 
            xy=(136, latency_136),
            xytext=(136, latency_128 - 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
# Add horizontal line
ax2.plot([133, 139], [latency_128, latency_128], color='red', lw=1)
ax2.annotate(f'{(slowdown-1)*100:.0f}% slowdown',
            xy=(136, (latency_128 + latency_136)/2 - 1),
            xytext=(5, -2),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='red')

# at 384
latency_384 = df_a100[df_a100['num_tokens'] == 384]['mean_latency'].values[0]
ax2.plot(384, latency_384, 'o', markersize=5, color='red')
ax2.annotate(f'{latency_384:.1f}ms', 
            xy=(384, latency_384),
            xytext=(-10, -12),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold')

# at 392
latency_392 = df_a100[df_a100['num_tokens'] == 392]['mean_latency'].values[0]
slowdown_384_392 = latency_392 / latency_384
ax2.plot(392, latency_392, 'o', markersize=5, color='red')
ax2.annotate(f'{latency_392:.1f}ms', 
            xy=(392, latency_392),
            xytext=(-17, 7),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold')
# Add arrow and slowdown text
ax2.annotate('', 
            xy=(392, latency_392),
            xytext=(392, latency_384 - 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
# Add horizontal line
ax2.plot([387, 397], [latency_384, latency_384], color='red', lw=1)
ax2.annotate(f'{(slowdown_384_392-1)*100:.0f}% slowdown',
            xy=(392, (latency_384 + latency_392)/2 - 1),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='red')

ax2.set_xlabel('Number of Tokens', fontsize=11)
ax2.set_ylabel('Latency (ms)', fontsize=11)
ax2.set_title('Model: OPT-13B / GPU: A100', fontsize=11, fontweight='bold')
ax2.set_xticks(xticks)
ax2.grid(True, linestyle='--', alpha=0.3, which='major')
ax2.tick_params(axis='both', which='major', labelsize=10)

# Plot 3: H200
ax3.plot(df_h200['num_tokens'], df_h200['mean_latency'], 'o-', linewidth=1.5, markersize=3, color='#1f77b4')

# Add annotations for H200 specific points
# at 256 
latency_256 = df_h200[df_h200['num_tokens'] == 256]['mean_latency'].values[0]
ax3.plot(256, latency_256, 'o', markersize=5, color='red')
ax3.annotate(f'{latency_256:.1f}ms', 
            xy=(256, latency_256),
            xytext=(-10, -12),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold')

# at 256 + 8 = 264
latency_264 = df_h200[df_h200['num_tokens'] == 264]['mean_latency'].values[0]
slowdown = latency_264 / latency_256
ax3.plot(264, latency_264, 'o', markersize=5, color='red')
ax3.annotate(f'{latency_264:.1f}ms', 
            xy=(264, latency_264),
            xytext=(-15, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold')
# Add arrow and slowdown text
ax3.annotate('', 
            xy=(264, latency_264),
            xytext=(264, latency_256 - 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
# Add horizontal line
ax3.plot([261, 267], [latency_256, latency_256], color='red', lw=1)
ax3.annotate(f'{(slowdown-1)*100:.0f}% slowdown',
            xy=(264, (latency_256 + latency_264)/2 - 1),
            xytext=(5, -2),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='red')

ax3.set_xlabel('Number of Tokens', fontsize=11)
ax3.set_ylabel('Latency (ms)', fontsize=11)
ax3.set_title('Model: OPT-66B / GPU: H200', fontsize=11, fontweight='bold')
ax3.set_xticks(xticks)
ax3.grid(True, linestyle='--', alpha=0.3, which='major')
ax3.tick_params(axis='both', which='major', labelsize=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('tiling_effect_combined.pdf', dpi=300, bbox_inches='tight')
plt.show() 