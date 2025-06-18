import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import os
from matplotlib import gridspec

# List of CSV files to process
CSV_FILES = [
    'A.csv', # OPT-6.7B / OPT-125M
    'B.csv', # Llama-13B / Vicuna-68M
    'C.csv', # OPT-13B / OPT-125M
    'D.csv', # 
    'E.csv', # OPT-30B / OPT-350M
]

# Replace with your desired value
y_max_values = {
    'A.csv': 1000, 
    'B.csv': 2000, 
    'C.csv': 1500, 
    'D.csv': 400, 
    'E.csv': 800, 
}

model_pairs = [
    '(a) OPT-6.7B / OPT-125M',
    '(b) Llama-13B / Vicuna-68M',
    '(c) OPT-13B / OPT-125M',
    '(d) Llama-70B / Llama-1B',
    '(e) OPT-30B / OPT-350M'
]

# Create figure with subplots for each CSV file
n_files = len(CSV_FILES)
fig = plt.figure(figsize=(15, 6.5))

# Create GridSpec with specific widths
gs = gridspec.GridSpec(2, 6, figure=fig)

# Create axes for first row (3 plots)
axes = []
for i in range(3):
    axes.append(fig.add_subplot(gs[0, i*2:(i+1)*2]))

# Create axes for second row (2 plots centered)
axes.append(fig.add_subplot(gs[1, 1:3]))  # First plot in second row
axes.append(fig.add_subplot(gs[1, 3:5]))  # Second plot in second row

# Create a list to store all lines and labels for the shared legend
all_lines = []
all_labels = []

# Custom color palette for the plot
ar_color = '#FF0000'  # Red for AR
cospec_color = '#006400'  # Forest green for CoSpec

# Specific colors and markers for each configuration
config_colors = {
    # Spec tokens with distinct colors and markers
    'Spec 1': ('#4B0082', 'o'),  # Indigo with circle
    'Spec 3': ('#FF8C00', 's'),  # Dark orange with square
    'Spec 5': ('#9400D3', '^'),  # Dark violet with triangle up
    'Spec 7': ('#1E90FF', 'D'),  # Dodger blue with diamond
    # Other configs with distinct colors and markers
    'Spec 2': ('#8B008B', 'v'),  # Dark magenta with triangle down
    'Spec 4': ('#228B22', '>'),  # Forest green with triangle right
    'Spec 6': ('#B8860B', '<'),  # Dark goldenrod with triangle left
    'Spec 8': ('#00CED1', 'p'),  # Dark turquoise with pentagon
    'DisableByBatch': ('#8B4513', '*'),  # Saddle brown with star
    'default': ('#696969', 'x')  # Dim gray with x for any other configs
}

# Process each CSV file
for idx, csv_file in enumerate(CSV_FILES):
    print(f"Processing file: {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Rename full_cospec to CoSpec in the config column
    df['config'] = df['config'].replace('full_cospec', 'CoSpec')
    
    # Get unique datasets
    datasets = sorted(df['dataset'].unique())
    
    # For this example, we'll plot the first dataset
    dataset = datasets[0]
    dataset_df = df[df['dataset'] == dataset]
    ax = axes[idx]
    
    # Plot Auto Regressive (baseline with spec_tokens=0)
    ar_data = dataset_df[(dataset_df['config'] == 'baseline') & (dataset_df['spec_tokens'] == 0)]
    line = ax.plot(ar_data['request_throughput'], ar_data['mean_token_latency'],
            marker='o', label='AR', linewidth=2, color=ar_color)
    if idx == 0:  # Only add to legend from first plot
        all_lines.extend(line)
        all_labels.append('AR')
    
    # Plot baseline with other spec_tokens for current temperature
    baseline_data = dataset_df[(dataset_df['config'] == 'baseline') & 
                      (dataset_df['spec_tokens'] > 0)]
    for spec_tokens in sorted(baseline_data['spec_tokens'].unique()):
        spec_data = baseline_data[baseline_data['spec_tokens'] == spec_tokens]
        spec_label = f'Spec {spec_tokens}'
        color, marker = config_colors.get(spec_label, config_colors['default'])
        line = ax.plot(spec_data['request_throughput'], spec_data['mean_token_latency'],
                marker=marker, label=spec_label, linewidth=2, color=color)
        if idx == 0:  # Only add to legend from first plot
            all_lines.extend(line)
            all_labels.append(spec_label)
    
    # Plot other configs
    other_configs = [config for config in dataset_df['config'].unique() if config != 'baseline']
    # Sort configs to put CoSpec last
    other_configs.sort(key=lambda x: x == 'CoSpec')
    
    # First plot all configs except CoSpec
    for config in other_configs:
        if config == 'CoSpec':
            continue
        config_data = dataset_df[(dataset_df['config'] == config)]
        # Rename disablebatch48 to DisableByBatch
        display_name = 'DisableByBatch' if config == 'disablebatch48' else config
        color, marker = config_colors.get(display_name, config_colors['default'])
        line = ax.plot(config_data['request_throughput'], config_data['mean_token_latency'],
                marker=marker, label=display_name, linewidth=2, color=color)
        if idx == 0:  # Only add to legend from first plot
            all_lines.extend(line)
            all_labels.append(display_name)
    
    # Then plot CoSpec last
    if 'CoSpec' in other_configs:
        config_data = dataset_df[(dataset_df['config'] == 'CoSpec')]
        line = ax.plot(config_data['request_throughput'], config_data['mean_token_latency'],
                marker='o', label='CoSpec', linewidth=2, color=cospec_color)
        if idx == 0:  # Only add to legend from first plot
            all_lines.extend(line)
            all_labels.append('CoSpec')
    
    # Calculate y-axis limits based on the data
    all_latencies = dataset_df['mean_token_latency']
    min_latency = all_latencies.min()
    max_latency = all_latencies.max()
    
    # Set y-axis limits with specific values for each plot
    y_min = max(1, min_latency * 0.8)  # Don't go below 1ms
    # Set specific y_max values for each plot
    y_max = y_max_values[csv_file]
    
    # y axis log scale with specific limits
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    
    # Customize subplot
    ax.set_xlabel('Request Throughput (req/s)', fontsize=14)
    # Show y-label for the leftmost plot in each row
    if idx == 0 or idx == 3:  # First plot in first row or first plot in second row
        ax.set_ylabel('Mean Token Latency (ms)', fontsize=14)
    else:
        ax.set_ylabel('')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add subplot label with model pairs
    ax.text(0.15, -0.4, model_pairs[idx], transform=ax.transAxes, fontsize=14, fontweight='bold')

# Create a single shared legend at the top
fig.legend(all_lines, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=7, fontsize=16, frameon=False)  # Increased font size and moved legend up

# Adjust layout and save
plt.tight_layout()
plt.savefig('main.pdf', bbox_inches='tight', pad_inches=0.5, format='pdf')  # Added pad_inches to ensure legend is visible
plt.close()
