import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import os
import numpy as np

# List of CSV files to process and their corresponding request rates
CSV_FILES = [
    'selective_validation_opt_A6000.csv',
    'selective_validation_llama_A6000.csv',
    'selective_validation_opt_A100.csv',
]

# Specify request rates for each CSV file
REQUEST_RATES = {
    'selective_validation_opt_A6000.csv': 12,    # Change this value for OPT-6.7B
    'selective_validation_llama_A6000.csv': 6,  # Change this value for Llama-13B
    'selective_validation_opt_A100.csv': 5,     # Change this value for OPT-30B
}

# Custom blueish and reddish color palette for the plot
blue_palette = ['#E74C3C', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']
green_color = '#228B22'  # Green color for CoSpec

# Map configurations to colors
config_colors = {
    'Without Selective Validation': 'red',  # Reddish
    'Threshold 0.1': blue_palette[2],  # Light blue
    'Threshold 0.3': blue_palette[3],  # Medium blue
    'Threshold 0.5': blue_palette[4],  # Dark blue
    'Linear': '#FF8C00',  # Darkest blue
    'Polynomial': '#8B4513',
    'Tile': green_color,  # Medium blue
}

# Create output directory for plots
output_dir = "combined_plots"
os.makedirs(output_dir, exist_ok=True)

# Create figure with subplots for each CSV file
n_files = len(CSV_FILES)
# Adjust figure size for 2-column paper (typically 7.5 inches wide)
fig, axes = plt.subplots(1, n_files, figsize=(7, 2))  # Reduced height for better fit

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
    
    # Get unique temperatures
    temperatures = sorted(dataset_df['temperature'].unique())
    # Move temperature -1 to the end if it exists
    if -1 in temperatures:
        temperatures.remove(-1)
        temperatures.append(-1)
    
    # Plot for temperature 0 (or first temperature)
    temp = temperatures[0]
    ax = axes[idx]
    
    # Get all configurations for this temperature
    temp_data = dataset_df[dataset_df['temperature'] == temp]
    
    # Get baseline data (colocation_consolidated)
    baseline_data = temp_data[(temp_data['config'] == 'colocation_consolidated') & 
                            (temp_data['request_rate'] == selected_request_rate)]
    
    # Define the desired order of configurations
    desired_order = [
        'Without Selective Validation',
        'Threshold 0.1',
        'Threshold 0.3',
        'Threshold 0.5',
        'Linear',
        'Polynomial',
        'Tile'
    ]
    
    # Map old config names to new labels
    config_label_map = {
        'colocation_consolidated_threshold_0.1': 'Threshold 0.1',
        'colocation_consolidated_threshold_0.3': 'Threshold 0.3',
        'colocation_consolidated_threshold_0.5': 'Threshold 0.5',
        'colocation_consolidated_tile': 'Tile',
        'colocation_consolidated_linear': 'Linear',
        'colocation_consolidated_polynomial': 'Polynomial'
    }
    
    # Create reverse mapping for finding config names
    reverse_config_map = {v: k for k, v in config_label_map.items()}
    
    # Get all configurations to plot
    configs = []
    max_speedup = 1.0  # Initialize with baseline value
    for label in desired_order:
        if label == 'Without Selective Validation':
            # This is the baseline, we'll handle it separately
            continue
            
        # Find the corresponding config name
        config_name = reverse_config_map.get(label)
        if config_name:
            config_data = temp_data[(temp_data['config'] == config_name) & 
                                  (temp_data['request_rate'] == selected_request_rate)]
            if not config_data.empty:
                configs.append((label, config_data))
                # Calculate speedup and update max_speedup if needed
                if not baseline_data.empty:
                    speedup = baseline_data['mean_token_latency'].iloc[0] / config_data['mean_token_latency'].iloc[0]
                    max_speedup = max(max_speedup, speedup)
    
    # Calculate bar positions
    n_configs = len(configs)
    bar_width = 0.8 / n_configs  # Adjust bar width based on number of configs
    x = np.arange(1)  # Only one position since we're showing one throughput
    
    # Plot bars for each configuration
    for i, (label, data) in enumerate(configs):
        # Calculate x positions for this configuration
        x_pos = x + (i - n_configs/2 + 0.5) * bar_width
        
        # Calculate speedup value
        if not data.empty and not baseline_data.empty:
            # Speedup = baseline_latency / current_latency
            speedup = baseline_data['mean_token_latency'].iloc[0] / data['mean_token_latency'].iloc[0]
            ax.bar(x_pos, speedup, bar_width, label=label, alpha=0.9, color=config_colors[label])
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Add horizontal line at y=1 (baseline) with red color and add to legend
    baseline_line = ax.axhline(y=1, color=config_colors['Without Selective Validation'], 
                             linestyle='--', label='Without Selective Validation')
    
    # Set y-axis limits based on maximum speedup value
    ax.set_ylim(0.75, max_speedup * 1.1)
    
    # Customize subplot
    ax.set_xlabel('')  # Remove x-axis label
    # Only show y-label for the leftmost plot
    if idx == 0:
        ax.set_ylabel('Mean Latency Speedup', fontsize=10)
    else:
        ax.set_ylabel('')
    ax.grid(True, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Add subplot label with model pairs and request rate
    model_pairs = [
        f'(a) OPT-6.7B / OPT-125M\n({selected_request_rate} req/s)',
        f'(b) Llama-13B / Vicuna-68M\n({selected_request_rate} req/s)',
        f'(c) OPT-30B / OPT-350M\n({selected_request_rate} req/s)'
    ]
    ax.text(0.5, -0.25, model_pairs[idx], transform=ax.transAxes, fontsize=10, fontweight='bold',
            horizontalalignment='center')  # Center align the text

# Create a single shared legend at the top
handles, labels = ax.get_legend_handles_labels()

# Define the desired order of legend items
desired_order = [
    'Without Selective Validation',
    'Threshold 0.1',
    'Threshold 0.3',
    'Threshold 0.5',
    'Linear',
    'Polynomial',
    'Tile'
]

# Reorder handles and labels according to desired order
ordered_handles = []
ordered_labels = []
for label in desired_order:
    if label in labels:
        idx = labels.index(label)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])

fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=4, fontsize=10, frameon=False)

# Adjust layout and save
plt.tight_layout(pad=0.5)  # Reduce padding between subplots
output_path = os.path.join(output_dir, 'main.pdf')
plt.savefig(output_path, bbox_inches='tight', format='pdf', pad_inches=0.1)  # Reduce padding around the figure
plt.close()

print(f"Combined plot has been saved to '{output_path}'")
