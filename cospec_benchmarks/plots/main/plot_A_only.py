import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import interp1d

# Read the CSV file
df = pd.read_csv('A.csv')

# Rename full_cospec to CoSpec in the config column
df['config'] = df['config'].replace('full_cospec', 'CoSpec')

# Get unique datasets
datasets = sorted(df['dataset'].unique())
dataset = datasets[0]
dataset_df = df[df['dataset'] == dataset]

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Custom colors
ar_color = '#FF0000'  # Red for AR
cospec_color = '#006400'  # Forest green for CoSpec

# Plot Auto Regressive (baseline with spec_tokens=0)
ar_data = dataset_df[(dataset_df['config'] == 'baseline') & (dataset_df['spec_tokens'] == 0)]
ax.plot(ar_data['request_throughput'], ar_data['mean_token_latency'],
        marker='o', label='AR', linewidth=3, color=ar_color, markersize=8)

# Plot CoSpec
cospec_data = dataset_df[dataset_df['config'] == 'CoSpec']
ax.plot(cospec_data['request_throughput'], cospec_data['mean_token_latency'],
        marker='s', label='CoSpec', linewidth=3, color=cospec_color, markersize=8)

# Define the reference lines
reference_latency = 500  # ms - horizontal line
reference_throughput = 2  # req/s - vertical line

# # Draw horizontal line for latency reference
# ax.axhline(y=reference_latency, color='gray', linestyle='--', alpha=0.7, linewidth=2)
# ax.text(11.5, reference_latency + 2, f'{reference_latency}ms', fontsize=12, ha='right', va='bottom')

# # Draw vertical line for throughput reference
# ax.axvline(x=reference_throughput, color='gray', linestyle='--', alpha=0.7, linewidth=2)
# ax.text(reference_throughput + 0.1, 800, f'{reference_throughput} req/s', fontsize=12, ha='left', va='top', rotation=90)

# Function to interpolate and find intersection points
def find_intersection(x_data, y_data, target_value, is_horizontal=True):
    """Find where the curve intersects with a horizontal or vertical line"""
    if len(x_data) < 2:
        return None, None
    
    if is_horizontal:
        # Find intersection with horizontal line (y = target_value)
        # Interpolate x given y
        f_interp = interp1d(y_data, x_data, kind='linear', bounds_error=False, fill_value='extrapolate')
        try:
            x_intersect = f_interp(target_value)
            return x_intersect, target_value
        except:
            return None, None
    else:
        # Find intersection with vertical line (x = target_value)
        # Interpolate y given x
        f_interp = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value='extrapolate')
        try:
            y_intersect = f_interp(target_value)
            return target_value, y_intersect
        except:
            return None, None

# Find intersection points for throughput speedup (horizontal line)
ar_x_throughput, ar_y_throughput = find_intersection(
    ar_data['request_throughput'].values, 
    ar_data['mean_token_latency'].values, 
    reference_latency, 
    is_horizontal=True
)

cospec_x_throughput, cospec_y_throughput = find_intersection(
    cospec_data['request_throughput'].values, 
    cospec_data['mean_token_latency'].values, 
    reference_latency, 
    is_horizontal=True
)

# Find intersection points for latency speedup (vertical line)
ar_x_latency, ar_y_latency = find_intersection(
    ar_data['request_throughput'].values, 
    ar_data['mean_token_latency'].values, 
    reference_throughput, 
    is_horizontal=False
)

cospec_x_latency, cospec_y_latency = find_intersection(
    cospec_data['request_throughput'].values, 
    cospec_data['mean_token_latency'].values, 
    reference_throughput, 
    is_horizontal=False
)

# Draw throughput speedup arrow (horizontal arrow)
if ar_x_throughput is not None and cospec_x_throughput is not None:
    # Calculate speedup
    throughput_speedup = cospec_x_throughput / ar_x_throughput
    
    # Draw horizontal arrow
    arrow1 = FancyArrowPatch((ar_x_throughput, reference_latency), (cospec_x_throughput, reference_latency),
                            arrowstyle='->', mutation_scale=20, 
                            color='black', linewidth=3)
    ax.add_patch(arrow1)
    
    # Add annotation
    mid_x = (ar_x_throughput + cospec_x_throughput) / 2
    ax.annotate(f'{throughput_speedup:.1f}x\nthroughput', 
                xy=(mid_x, reference_latency), xytext=(mid_x, reference_latency - 250),
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Draw latency speedup arrow (vertical arrow)
if ar_y_latency is not None and cospec_y_latency is not None:
    # Calculate speedup
    latency_speedup = ar_y_latency / cospec_y_latency
    
    # Draw vertical arrow
    arrow2 = FancyArrowPatch((reference_throughput, ar_y_latency), (reference_throughput, cospec_y_latency),
                            arrowstyle='->', mutation_scale=20, 
                            color='black', linewidth=3)
    ax.add_patch(arrow2)
    
    # Add annotation
    mid_y = (ar_y_latency + cospec_y_latency) / 2
    ax.annotate(f'{latency_speedup:.1f}x\nlatency', 
                xy=(reference_throughput, mid_y), xytext=(reference_throughput + 0.8, mid_y),
                ha='left', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Customize plot
ax.set_xlabel('Request Throughput (req/s)', fontsize=16)
ax.set_ylabel('Mean Token Latency (ms)', fontsize=16)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=14)

# Set y-axis to log scale
ax.set_yscale('log')

# Set reasonable limits
ax.set_xlim(0, 12)
ax.set_ylim(5, 600)

# Add legend above the plot
ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False, ncol=2)

# Add subplot label at the bottom
ax.text(0.5, -0.25, '(a) OPT-6.7B/OPT-125M (A6000)', transform=ax.transAxes, 
        fontsize=14, fontweight='bold', ha='center')

# Adjust layout and save
plt.tight_layout()
plt.savefig('A_only.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.show() 