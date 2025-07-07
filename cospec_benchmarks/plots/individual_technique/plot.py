import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import os
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# List of CSV files to process and their corresponding request rates
CSV_FILES = [
    'F.csv',
    'G.csv',
    'H.csv',
]

# Specify request rates for each CSV file
REQUEST_RATES = {
    'F.csv': 10,    # Change this value for OPT-6.7B
    'G.csv': 10,  # Change this value for Llama-13B
    'H.csv': 10,     # Change this value for OPT-30B
}

# Specify GPU names for each CSV file
GPU_NAMES = {
    'F.csv': 'A6000',    # GPU for OPT-6.7B
    'G.csv': 'A100',     # GPU for OPT-13B
    'H.csv': 'H200',     # GPU for OPT-30B
}

# Custom color palette for the plot
blue_palette = ['#E74C3C', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']
green_palette = ['#228B22', '#32CD32', '#90EE90']  # Forest green, Lime green, Light green
orange_palette = ['#FF8C00', '#FFA500', '#FFD700']  # Dark orange, Orange, Gold

# Map configurations to colors
config_colors = {
    'Spec 7': blue_palette[1],
    'Dynamic Colocation': blue_palette[2],  # Medium blue
    'Dynamic Colocation + Selective Validation': blue_palette[4],  # Dark blue
    'Dynamic Colocation + Selective Validation + Consolidated Attention': green_palette[0],  # Forest green
}

# Create figure with subplots for each CSV file
n_files = len(CSV_FILES)
# Adjust figure size for 2-column paper (typically 7.5 inches wide)
fig, axes = plt.subplots(2, n_files, figsize=(8, 3.5))  # Reduced width from 7 to 6.5 inches

# Adjust subplot spacing
plt.subplots_adjust(wspace=0.15)  # Reduce horizontal space between subplots

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
    
    # Process both throughput and latency plots
    for row in range(2):
        ax = axes[row, idx]
        
        # Define the desired order of configurations
        desired_order = [
            'Spec 7',
            'Dynamic Colocation',
            'Dynamic Colocation + Selective Validation',
            'Dynamic Colocation + Selective Validation + Consolidated Attention',
            'AR'
        ]
        
        # Map old config names to new labels
        config_label_map = {
            'baseline': 'Spec 7',
            'colocation': 'Dynamic Colocation',
            'colocation_dynamic': 'Dynamic Colocation',
            'colocation_dynamic_selective': 'Dynamic Colocation + Selective Validation',
            'full_cospec': 'Dynamic Colocation + Selective Validation + Consolidated Attention'
        }
        
        # Create reverse mapping for finding config names
        reverse_config_map = {v: k for k, v in config_label_map.items()}
        
        # Get all configurations to plot
        configs = []
        max_value = 0.0  # Initialize with 0
        min_value = float('inf')  # Initialize with infinity
        
        # First get baseline data (Spec 7)
        baseline_data = dataset_df[(dataset_df['config'] == 'baseline') & 
                                (dataset_df['request_rate'] == selected_request_rate)]
        if not baseline_data.empty:
            configs.append(('Spec 7', baseline_data))
            if row == 0:  # Throughput
                max_value = max(max_value, baseline_data['request_throughput'].iloc[0])
                min_value = min(min_value, baseline_data['request_throughput'].iloc[0])
            else:  # Latency
                max_value = max(max_value, baseline_data['mean_token_latency'].iloc[0] / 1000)  # Convert to seconds
                min_value = min(min_value, baseline_data['mean_token_latency'].iloc[0] / 1000)  # Convert to seconds
        
        # Then get other configurations
        for label in desired_order[1:]:  # Skip baseline as we already added it
            config_name = reverse_config_map.get(label)
            if config_name:
                config_data = dataset_df[(dataset_df['config'] == config_name) & 
                                      (dataset_df['request_rate'] == selected_request_rate)]
                if not config_data.empty:
                    configs.append((label, config_data))
                    if row == 0:  # Throughput
                        max_value = max(max_value, config_data['request_throughput'].iloc[0])
                        min_value = min(min_value, config_data['request_throughput'].iloc[0])
                    else:  # Latency
                        max_value = max(max_value, config_data['mean_token_latency'].iloc[0] / 1000)  # Convert to seconds
                        min_value = min(min_value, config_data['mean_token_latency'].iloc[0] / 1000)  # Convert to seconds
        
        # Also check autoregressive baseline for min/max
        ar_baseline = dataset_df[(dataset_df['spec_tokens'] == 0) & 
                               (dataset_df['request_rate'] == selected_request_rate)]
        if not ar_baseline.empty:
            if row == 0:  # Throughput
                max_value = max(max_value, ar_baseline['request_throughput'].iloc[0])
                min_value = min(min_value, ar_baseline['request_throughput'].iloc[0])
            else:  # Latency
                max_value = max(max_value, ar_baseline['mean_token_latency'].iloc[0] / 1000)  # Convert to seconds
                min_value = min(min_value, ar_baseline['mean_token_latency'].iloc[0] / 1000)  # Convert to seconds
        
        # Calculate bar positions
        n_configs = len(configs)
        bar_width = 0.6 / n_configs  # Reduced bar width to add margin
        x = np.arange(1)  # Only one position since we're showing one throughput
        
        # Plot bars for each configuration
        for i, (label, data) in enumerate(configs):
            # Calculate x positions for this configuration
            x_pos = x + (i - n_configs/2 + 0.5) * bar_width
            
            # Get actual value
            if row == 0:  # Throughput
                value = data['request_throughput'].iloc[0]
            else:  # Latency
                value = data['mean_token_latency'].iloc[0] / 1000  # Convert ms to s
            
            # Plot the bar with edge color
            bar = ax.bar(x_pos, value, bar_width, label=label, alpha=0.9, 
                        color=config_colors[label], edgecolor='black', linewidth=0.5)
            
            # Add speedup annotation on top of the bar
            if label == 'Spec 7':  # Baseline
                ax.text(x_pos, value, '1.00x', 
                       ha='center', va='bottom', fontsize=8)
            else:  # Other configurations
                if row == 0:  # Throughput
                    speedup = value / baseline_data['request_throughput'].iloc[0]
                else:  # Latency
                    speedup = baseline_data['mean_token_latency'].iloc[0] / data['mean_token_latency'].iloc[0]
                ax.text(x_pos, value, f'{speedup:.2f}x', 
                       ha='center', va='bottom', fontsize=8)
        
        # Add red line for autoregressive baseline (spec_tokens = 0)
        if not ar_baseline.empty:
            if row == 0:  # Throughput
                ar_value = ar_baseline['request_throughput'].iloc[0]
            else:  # Latency
                ar_value = ar_baseline['mean_token_latency'].iloc[0] / 1000  # Convert ms to s
            ar_line = ax.axhline(y=ar_value, color='red', linestyle='--', alpha=0.7, label='AR')

        # Remove x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Set y-axis limits based on min and max values with small margin
        margin = (max_value - min_value) * 0.15  # 10% margin
        ax.set_ylim(max(0, min_value - margin), max_value + margin)
        
        # Set y-ticks for latency plot to 0.5 granularity
        if row == 1:
            if idx == 2:
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
            elif idx == 1:
                ax.yaxis.set_major_locator(MultipleLocator(1))
        
        # Customize subplot
        ax.set_xlabel('')  # Remove x-axis label
        # Only show y-label for the leftmost plot
        if idx == 0:
            if row == 0:
                ax.set_ylabel('Request Throughput\n(req/s)', fontsize=10)
            else:
                ax.set_ylabel('Mean Token Latency\n(s/token)', fontsize=10)
        else:
            ax.set_ylabel('')
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.5, linewidth=0.5, zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Add subplot label with model pairs and request rate
        if row == 1:  # Only add model labels to bottom row
            model_pairs = [
                f'(a) OPT-6.7B / OPT-125M\n({selected_request_rate} req/s, {GPU_NAMES["F.csv"]})',
                f'(b) OPT-13B / OPT-125M\n({selected_request_rate} req/s, {GPU_NAMES["G.csv"]})',
                f'(c) OPT-30B / OPT-350M\n({selected_request_rate} req/s, {GPU_NAMES["H.csv"]})'
            ]
            ax.text(0.5, -0.30, model_pairs[idx], transform=ax.transAxes, fontsize=10, fontweight='bold',
                    horizontalalignment='center')  # Center align the text

# Create a single shared legend at the top
handles, labels = axes[0, 0].get_legend_handles_labels()

# Define the desired order of legend items
desired_order = [
    'AR',
    'Spec 7',
    'Dynamic Colocation',
    'Dynamic Colocation + Selective Validation',
    'Dynamic Colocation + Selective Validation + Consolidated Attention'
]

# Create a custom line for AR in the legend
ar_line = Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='AR')

# Reorder handles and labels according to desired order
ordered_handles = []
ordered_labels = []
for label in desired_order:
    if label == 'AR':
        ordered_handles.append(ar_line)
        ordered_labels.append('AR')
    elif label in labels:
        idx = labels.index(label)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])

fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=6, fontsize=12, frameon=False)

# Adjust layout and save
plt.tight_layout(pad=0.5)  # Reduced padding between subplots from 0.5 to 0.2
output_path = 'individual_technique.pdf'
plt.savefig(output_path, bbox_inches='tight', format='pdf', pad_inches=0.05)  # Reduced padding around the figure from 0.1 to 0.05
plt.close()

print(f"Combined plot has been saved to '{output_path}'")
