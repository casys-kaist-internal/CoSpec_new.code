import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import os
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot benchmark results from CSV file')
parser.add_argument('csv', type=str, help='Path to the benchmark results CSV file')
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.csv)

# Create output directory for plots using the input CSV filename
output_dir = os.path.splitext(os.path.basename(args.csv))[0]
output_dir = "plot_" + output_dir
os.makedirs(output_dir, exist_ok=True)

# Get unique datasets
datasets = sorted(df['dataset'].unique())

# Plot for each dataset
for dataset in datasets:
    print(f"Plotting results for dataset: {dataset}")
    dataset_df = df[df['dataset'] == dataset]
    
    # Get unique temperatures and calculate grid dimensions
    temperatures = sorted(dataset_df['temperature'].unique())
    # Move temperature -1 to the end if it exists
    if -1 in temperatures:
        temperatures.remove(-1)
        temperatures.append(-1)
    n_temps = len(temperatures)
    n_cols = 5
    n_rows = math.ceil(n_temps / n_cols)

    # Create figure with subplots for each temperature
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 5*n_rows))
    axes = axes.flatten()
    
    # Plot for each temperature
    for idx, temp in enumerate(temperatures):
        ax = axes[idx]
        
        # Get all configurations for this temperature
        temp_data = dataset_df[dataset_df['temperature'] == temp]
        
        # Get unique request rates
        throughputs = sorted(temp_data['request_throughput'].unique())
        
        # Get baseline data (colocation_consolidated)
        baseline_data = temp_data[temp_data['config'] == 'colocation_consolidated']
        
        # Get all configurations to plot (excluding baseline)
        configs = []
        # Add Auto Regressive (baseline with spec_tokens=0)
        ar_data = temp_data[(temp_data['config'] == 'baseline') & (temp_data['spec_tokens'] == 0)]
        if not ar_data.empty:
            configs.append(('Auto Regressive', ar_data))
        
        # Add baseline with other spec_tokens
        spec_data = temp_data[(temp_data['config'] == 'baseline') & (temp_data['spec_tokens'] > 0)]
        for spec_tokens in sorted(spec_data['spec_tokens'].unique()):
            spec_config = spec_data[spec_data['spec_tokens'] == spec_tokens]
            configs.append((f'baseline (spec_tokens={spec_tokens})', spec_config))
        
        # Add other configs (excluding colocation_consolidated)
        other_configs = [config for config in temp_data['config'].unique() 
                        if config not in ['baseline', 'colocation_consolidated']]
        for config in other_configs:
            config_data = temp_data[temp_data['config'] == config]
            configs.append((config, config_data))
        
        # Calculate bar positions
        n_configs = len(configs)
        bar_width = 0.8 / n_configs  # Adjust bar width based on number of configs
        x = np.arange(len(throughputs))
        
        # Plot bars for each configuration
        for i, (label, data) in enumerate(configs):
            # Calculate x positions for this configuration
            x_pos = x + (i - n_configs/2 + 0.5) * bar_width
            
            # Calculate throughput speedup values for each throughput
            speedup_values = []
            for tp in throughputs:
                tp_data = data[data['request_throughput'] == tp]
                tp_baseline = baseline_data[baseline_data['request_throughput'] == tp]
                
                if not tp_data.empty and not tp_baseline.empty:
                    # Throughput speedup = current_throughput / baseline_throughput
                    # We use request_rate as throughput since it represents the achieved throughput
                    speedup = tp_data['request_throughput'].iloc[0] / tp_baseline['request_throughput'].iloc[0]
                    speedup_values.append(speedup)
                else:
                    speedup_values.append(0)
            
            ax.bar(x_pos, speedup_values, bar_width, label=label, alpha=0.7)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels([f'{tp:.1f}' for tp in throughputs], rotation=45)
        
        # Add horizontal line at y=1 (baseline)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Customize subplot
        ax.set_xlabel('Request Rate (req/s)', fontsize=10)
        ax.set_ylabel('Throughput Speedup vs Colocation Consolidated', fontsize=10)
        # Display "Random" for temperature -1
        temp_title = "Random" if temp == -1 else f"Temperature = {temp}"
        ax.set_title(temp_title, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Remove any empty subplots
    for idx in range(len(temperatures), len(axes)):
        fig.delaxes(axes[idx])

    # Add dataset name to the figure
    fig.suptitle(f'Dataset: {dataset}', fontsize=16, y=1.02)

    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'throughput_speedup_{dataset}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

print(f"Plots have been saved to the '{output_dir}' directory")
