import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot benchmark results from CSV file')
parser.add_argument('csv', type=str, help='Path to the benchmark results CSV file')
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.csv)

# Create output directory for plots
output_dir = 'benchmark_plots'
os.makedirs(output_dir, exist_ok=True)

# Get unique datasets
datasets = sorted(df['dataset'].unique())

# Plot for each dataset
for dataset in datasets:
    print(f"Plotting results for dataset: {dataset}")
    dataset_df = df[df['dataset'] == dataset]
    
    # Get unique temperatures and calculate grid dimensions
    temperatures = sorted(dataset_df['temperature'].unique())
    n_temps = len(temperatures)
    n_cols = 5
    n_rows = math.ceil(n_temps / n_cols)

    # Create figure with subplots for each temperature
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 5*n_rows))
    axes = axes.flatten()
    
    # Plot for each temperature
    for idx, temp in enumerate(temperatures):
        ax = axes[idx]
        
        # Plot Auto Regressive (baseline with spec_tokens=0) for all temperatures
        ar_data = dataset_df[(dataset_df['config'] == 'baseline') & (dataset_df['spec_tokens'] == 0)]
        ax.plot(ar_data['request_throughput'], ar_data['mean_token_latency'],
                marker='o', label='Auto Regressive', linewidth=2, color='black')
        
        # Plot baseline with other spec_tokens for current temperature
        baseline_data = dataset_df[(dataset_df['config'] == 'baseline') & 
                          (dataset_df['temperature'] == temp) & 
                          (dataset_df['spec_tokens'] > 0)]
        for spec_tokens in sorted(baseline_data['spec_tokens'].unique()):
            spec_data = baseline_data[baseline_data['spec_tokens'] == spec_tokens]
            ax.plot(spec_data['request_throughput'], spec_data['mean_token_latency'],
                    marker='o', label=f'baseline (spec_tokens={spec_tokens})', linewidth=2)
        
        # Plot other configs
        other_configs = [config for config in dataset_df['config'].unique() if config != 'baseline']
        for config in other_configs:
            config_data = dataset_df[(dataset_df['config'] == config) & (dataset_df['temperature'] == temp)]
            ax.plot(config_data['request_throughput'], config_data['mean_token_latency'],
                    marker='o', label=config, linewidth=2)
        
        # Customize subplot
        ax.set_xlabel('Request Throughput (req/s)', fontsize=10)
        ax.set_ylabel('Mean Token Latency (ms)', fontsize=10)
        ax.set_title(f'Temperature = {temp}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Remove any empty subplots
    for idx in range(len(temperatures), len(axes)):
        fig.delaxes(axes[idx])

    # Add dataset name to the figure
    fig.suptitle(f'Dataset: {dataset}', fontsize=16, y=1.02)

    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'mean_token_latency_{dataset}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

print(f"Plots have been saved to the '{output_dir}' directory")
