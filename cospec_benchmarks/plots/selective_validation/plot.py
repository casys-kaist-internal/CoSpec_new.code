import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import os
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# List of CSV files to process and their corresponding request rates
CSV_FILES = [
    'F.csv',
    'G.csv',
    'H.csv',
]

# Specify request rates for each CSV file
REQUEST_RATES = {
    'F.csv': 10,    
    'G.csv': 10,  
    'H.csv': 10,     
}

# Custom color palette for the plot
blue_palette = ['#E74C3C', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']
green_palette = ['#228B22', '#32CD32', '#90EE90']  # Forest green, Lime green, Light green
orange_palette = ['#FF8C00', '#FFA500', '#FFD700']  # Dark orange, Orange, Gold

# Define validation types and their colors
validation_types = ['Threshold', 'Linear', 'Polynomial', 'Tiled']
type_colors = {
    'Threshold': blue_palette[1],    # Light blue
    'Linear': blue_palette[2],      # Medium blue
    'Polynomial': blue_palette[3],  # Dark blue
    'Tiled': green_palette[0]              # Forest green
}

# Define threshold values and their hatch patterns
threshold_values = ['0.1', '0.3', '0.5']
bar_styles = [
    '',           # No hatch for 0.1 (solid color)
    '....',       # Dots for 0.3
    'xxxx'        # Dense cross for 0.5
]

# Create figure with subplots for each CSV file
n_files = len(CSV_FILES)
# Adjust figure size for better visualization
fig, axes = plt.subplots(1, n_files, figsize=(9, 2.5))  # Wider figure for better spacing

# Set style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')

# Process each CSV file
for idx, csv_file in enumerate(CSV_FILES):
    print(f"Processing file: {csv_file}")
    
    # Get the request rate for this CSV file
    selected_request_rate = REQUEST_RATES[csv_file]
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get unique datasets
    datasets = sorted(df['dataset'].unique())
    
    # For this example, we'll plot the first dataset
    dataset = datasets[0]
    dataset_df = df[df['dataset'] == dataset]
    
    ax = axes[idx]    
    # Get baseline data
    baseline_data = dataset_df[(dataset_df['config'] == 'without_selective_validation') & 
                            (dataset_df['request_rate'] == selected_request_rate)]
    
    # Define the desired order of configurations
    desired_order = [
        'Without Selective Validation',
        'Threshold 0.1',
        'Threshold 0.3',
        'Threshold 0.5',
        'Linear 0.1',
        'Linear 0.3',
        'Linear 0.5',
        'Polynomial 0.1',
        'Polynomial 0.3',
        'Polynomial 0.5',
        'Tiled 0.1',
        'Tiled 0.3',
        'Tiled 0.5'
    ]
    
    # Map old config names to new labels
    config_label_map = {
        'without_selective_validation': 'Without Selective Validation',
        'selective_validation_threshold_0.1': 'Threshold 0.1',
        'selective_validation_threshold_0.3': 'Threshold 0.3',
        'selective_validation_threshold_0.5': 'Threshold 0.5',
        'selective_validation_tile_0.1': 'Tiled 0.1',
        'selective_validation_tile_0.3': 'Tiled 0.3',
        'selective_validation_tile_0.5': 'Tiled 0.5',
        'selective_validation_linear_0.1': 'Linear 0.1',
        'selective_validation_linear_0.3': 'Linear 0.3',
        'selective_validation_linear_0.5': 'Linear 0.5',
        'selective_validation_polynomial_0.1': 'Polynomial 0.1',
        'selective_validation_polynomial_0.3': 'Polynomial 0.3',
        'selective_validation_polynomial_0.5': 'Polynomial 0.5'
    }
    
    # Create reverse mapping for finding config names
    reverse_config_map = {v: k for k, v in config_label_map.items()}
    
    # Get all configurations to plot
    configs = []
    max_speedup = 1.0  # Initialize with baseline value
    min_speedup = 1.0  # Initialize with baseline value
    for label in desired_order:
        if label == 'Without Selective Validation':
            continue
            
        config_name = reverse_config_map.get(label)
        if config_name:
            config_data = dataset_df[(dataset_df['config'] == config_name) & 
                                  (dataset_df['request_rate'] == selected_request_rate)]
            if not config_data.empty:
                configs.append((label, config_data))
                if not baseline_data.empty:
                    speedup = baseline_data['mean_token_latency'].iloc[0] / config_data['mean_token_latency'].iloc[0]
                    max_speedup = max(max_speedup, speedup)
                    min_speedup = min(min_speedup, speedup)
    
    # Calculate bar positions
    n_configs = len(configs)
    bar_width = 0.8 / n_configs
    x = np.arange(1)
    
    # Plot bars for each configuration
    for i, (label, data) in enumerate(configs):
        # Add spacing between validation types
        validation_type = label.split()[0]
        type_index = validation_types.index(validation_type)
        spacing = type_index * 0.04  # Add 0.1 spacing for each validation type
        
        x_pos = x + (i - n_configs/2 + 0.5) * bar_width + spacing
        
        if not data.empty and not baseline_data.empty:
            speedup = baseline_data['mean_token_latency'].iloc[0] / data['mean_token_latency'].iloc[0]
            
            parts = label.split()
            vtype = parts[0]
            threshold = parts[1]
            
            color = type_colors[vtype]
            hatch_idx = threshold_values.index(threshold)
            hatch = bar_styles[hatch_idx]
            
            # Plot bar with improved styling
            ax.bar(x_pos, speedup, bar_width, 
                  color=color,
                  hatch=hatch,
                  edgecolor='black',
                  linewidth=1,
                  alpha=0.9)
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Add horizontal line at y=1 (baseline) with improved styling
    baseline_line = ax.axhline(y=1, color='red',  # Using first color from blue palette for baseline
                             linestyle='--', 
                             alpha=0.7,
                             label='Without Selective Validation')
        
    # Set y-axis limits with some padding
    y_padding = 0.05  # 5% padding
    y_min = max(0.75, min_speedup * (1 - y_padding))  # Ensure minimum is at least 0.75
    y_max = max_speedup * (1 + y_padding)
    ax.set_ylim(y_min, y_max)
    
    # Customize subplot
    ax.set_xlabel('')
    if idx == 0:
        ax.set_ylabel('Mean Latency Speedup', fontsize=12)
    else:
        ax.set_ylabel('')
    
    # Improve grid appearance
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add subplot label with improved styling
    model_pairs = [
        f'(a) OPT-6.7B / OPT-125M\n({selected_request_rate} req/s, A6000)',
        f'(b) OPT-13B / OPT-125M\n({selected_request_rate} req/s, A100)',
        f'(c) OPT-30B / OPT-350M\n({selected_request_rate} req/s, H200)'
    ]
    ax.text(0.5, -0.25, model_pairs[idx], transform=ax.transAxes, 
            fontsize=12, fontweight='bold',
            horizontalalignment='center')

# Create custom legend elements
legend_elements = []
legend_labels = []

# Add baseline to legend elements first
legend_elements.append(Line2D([0], [0], color='red', 
                           linestyle='--', 
                           linewidth=2,
                           label='Without SV'))
legend_labels.append('Without SV')

# Add validation type patches with improved styling
for vtype in validation_types:
    legend_elements.append(Patch(facecolor=type_colors[vtype], 
                             edgecolor='black',
                             linewidth=1,
                             label=vtype))
    legend_labels.append(vtype)

# Add threshold value patches with improved styling
for i, value in enumerate(threshold_values):
    legend_elements.append(Patch(facecolor='white', 
                              edgecolor='black',
                              hatch=bar_styles[i],
                              linewidth=1,
                              label=value))
    legend_labels.append(value)

# Create single legend
fig.legend(legend_elements, 
          legend_labels,
          loc='upper center', 
          bbox_to_anchor=(0.5, 1.15),
          ncol=8,  # Adjust number of columns to fit all items in one row
          fontsize=12,
          frameon=False,
          columnspacing=1.0)

# Adjust layout and save with improved spacing
plt.tight_layout(pad=0.5)
output_path = 'selective_validation.pdf'
plt.savefig(output_path, bbox_inches='tight', format='pdf', dpi=300)
plt.close()

print(f"Combined plot has been saved to '{output_path}'")
