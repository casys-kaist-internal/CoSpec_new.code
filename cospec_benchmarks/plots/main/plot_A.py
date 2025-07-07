import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import os
from matplotlib import gridspec

# List of CSV files to process - only A.csv
CSV_FILES = [
    'A.csv', # OPT-6.7B / OPT-125M
]

# Replace with your desired value
y_max_values = {
    'A.csv': 600, 
}

model_pairs = [
    'OPT-6.7B / OPT-125M',
]

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

# Create a list to store all lines and labels for the legend
all_lines = []
all_labels = []

# Custom color palette for the plot
ar_color = '#FF0000'  # Red for AR

# Specific colors and markers for each configuration
config_colors = {
    # Spec tokens with distinct colors and markers
    'Spec 1': ('#D7E2F9', 'o'),  # Indigo with circle
    'Spec 3': ('#88BCFF', 's'),  # Dark orange with square
    'Spec 5': ('#3864B9', '^'),  # Dark violet with triangle up
    'Spec 7': ('#1B345F', 'D'),  # Dodger blue with diamond
    # Other configs with distinct colors and markers

    'DisableByBatch': ('#8B4513', '*'),  # Saddle brown with star
    'default': ('#696969', 'x')  # Dim gray with x for any other configs
}

blue_palette = ['#E74C3C', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']
green_palette = ['#228B22', '#32CD32', '#90EE90']  # Forest green, Lime green, Light green
orange_palette = ['#FF8C00', '#FFA500', '#FFD700']  # Dark orange, Orange, Gold


# Process A.csv file
csv_file = 'A.csv'
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

# Store AR data for speedup calculation
ar_data = dataset_df[(dataset_df['config'] == 'baseline') & (dataset_df['spec_tokens'] == 0)]

# Plot Auto Regressive (baseline with spec_tokens=0) on first subplot
# Plot black outline first
ax1.plot(ar_data['request_throughput'], ar_data['mean_token_latency'],
        marker='o', linewidth=4, color='black', markeredgecolor='black', markeredgewidth=1)
# Plot colored line on top
line = ax1.plot(ar_data['request_throughput'], ar_data['mean_token_latency'],
        marker='o', label='AR', linewidth=2, color=ar_color, markeredgecolor='black', markeredgewidth=1)
all_lines.extend(line)
all_labels.append('AR')

# Plot baseline with other spec_tokens for current temperature on first subplot
baseline_data = dataset_df[(dataset_df['config'] == 'baseline') & 
                  (dataset_df['spec_tokens'] > 0)]
for spec_tokens in sorted(baseline_data['spec_tokens'].unique()):
    spec_data = baseline_data[baseline_data['spec_tokens'] == spec_tokens]
    spec_label = f'Spec {spec_tokens}'
    color, marker = config_colors.get(spec_label, config_colors['default'])
    # Plot black outline first
    ax1.plot(spec_data['request_throughput'], spec_data['mean_token_latency'],
            marker=marker, linewidth=3, color='black', markeredgecolor='black', markeredgewidth=1)
    # Plot colored line on top
    line = ax1.plot(spec_data['request_throughput'], spec_data['mean_token_latency'],
            marker=marker, label=spec_label, linewidth=2, color=color, markeredgecolor='black', markeredgewidth=1)
    all_lines.extend(line)
    all_labels.append(spec_label)

# Plot other configs (excluding CoSpec) on first subplot
other_configs = [config for config in dataset_df['config'].unique() if config != 'baseline' and config != 'CoSpec']

for config in other_configs:
    config_data = dataset_df[(dataset_df['config'] == config)]
    # Rename disablebatch48 to DisableByBatch
    display_name = 'DisableByBatch' if config == 'disablebatch48' else config
    color, marker = config_colors.get(display_name, config_colors['default'])
    # Plot black outline first
    ax1.plot(config_data['request_throughput'], config_data['mean_token_latency'],
            marker=marker, linewidth=3, color='black', markeredgecolor='black', markeredgewidth=1)
    # Plot colored line on top
    line = ax1.plot(config_data['request_throughput'], config_data['mean_token_latency'],
            marker=marker, label=display_name, linewidth=2, color=color, markeredgecolor='black', markeredgewidth=1)
    all_lines.extend(line)
    all_labels.append(display_name)

# Calculate y-axis limits based on the data for first subplot
all_latencies = dataset_df['mean_token_latency']
min_latency = all_latencies.min()
max_latency = all_latencies.max()

# Set y-axis limits with specific values for each plot
y_min = max(1, min_latency * 0.8)  # Don't go below 1ms
# Set specific y_max values for each plot
y_max = y_max_values[csv_file]

# y axis log scale with specific limits for first subplot
ax1.set_yscale('log')
ax1.set_ylim(y_min, y_max)

# Customize first subplot
ax1.set_xlabel('Request Throughput (req/s)', fontsize=12)
ax1.set_ylabel('Mean Token Latency (ms)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='both', which='major', labelsize=10)


# Second subplot: Bar graph showing speedup compared to AR
# Get unique request throughput values
request_rates = sorted(ar_data['request_rate'].unique())

# show only 5 request rates
request_rates = request_rates[:5]

# Create speedup data for bar plot
speedup_data = {}
config_names = []

# Calculate speedup for each configuration at each request rate
for config in ['baseline'] + other_configs:
    if config == 'baseline':
        # Handle baseline with different spec_tokens
        for spec_tokens in sorted(baseline_data['spec_tokens'].unique()):
            spec_data = baseline_data[baseline_data['spec_tokens'] == spec_tokens]
            spec_label = f'Spec {spec_tokens}'
            config_names.append(spec_label)
            
            speedups = []
            print(f"\n{spec_label} speedups:")
            for request_rate in request_rates:
                ar_latency = ar_data[ar_data['request_rate'] == request_rate]['mean_token_latency'].iloc[0]
                spec_data_at_rate = spec_data[spec_data['request_rate'] == request_rate]
                if len(spec_data_at_rate) > 0:
                    spec_latency = spec_data_at_rate['mean_token_latency'].iloc[0]
                    speedup = ar_latency / spec_latency
                    print(f"  Request rate {request_rate:.0f} req/s: AR={ar_latency:.2f}ms, {spec_label}={spec_latency:.2f}ms, Speedup={speedup:.3f}x")
                else:
                    speedup = 1.0  # No speedup if no data
                    print(f"  Request rate {request_rate:.0f} req/s: No data available")
                speedups.append(speedup)
            speedup_data[spec_label] = speedups
    else:
        # Handle other configs
        config_data = dataset_df[(dataset_df['config'] == config)]
        display_name = 'DisableByBatch' if config == 'disablebatch48' else config
        config_names.append(display_name)
        
        speedups = []
        print(f"\n{display_name} speedups:")
        for request_rate in request_rates:
            ar_latency = ar_data[ar_data['request_rate'] == request_rate]['mean_token_latency'].iloc[0]
            config_data_at_rate = config_data[config_data['request_rate'] == request_rate]
            if len(config_data_at_rate) > 0:
                config_latency = config_data_at_rate['mean_token_latency'].iloc[0]
                speedup = ar_latency / config_latency
                print(f"  Request rate {request_rate:.0f} req/s: AR={ar_latency:.2f}ms, {display_name}={config_latency:.2f}ms, Speedup={speedup:.3f}x")
            else:
                speedup = 1.0  # No speedup if no data
                print(f"  Request rate {request_rate:.0f} req/s: No data available")
            speedups.append(speedup)
        speedup_data[display_name] = speedups

print(f"\nRequest rates used: {[f'{r:.0f}' for r in request_rates]}")
print(f"Configurations: {config_names}")

# Create bar plot
x = range(len(request_rates))
width = 0.8 / len(config_names)  # Adjust bar width based on number of configs

for i, config_name in enumerate(config_names):
    speedups = speedup_data[config_name]
    color, _ = config_colors.get(config_name, config_colors['default'])
    ax2.bar([xi + i * width for xi in x], speedups, width, label=config_name, color=color, alpha=0.8, 
            edgecolor='black', linewidth=1)

# Add red dotted line at speedup 1.0 to show AR baseline
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='AR Baseline')
# Add "AR" text on the dotted line
ax2.text(ax2.get_xlim()[1] * 0.98, 1.05, 'Auto Regressive', color='red', fontsize=10, 
         ha='right', va='bottom', fontweight='bold')

# Customize second subplot
ax2.set_xlabel('Request Rate (req/s)', fontsize=12)
ax2.set_ylabel('Latency Speedup', fontsize=12)
ax2.set_xticks([xi + width * (len(config_names) - 1) / 2 for xi in x])
ax2.set_xticklabels([f'{r:.0f}' for r in request_rates], fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
ax2.tick_params(axis='both', which='major', labelsize=10)

# Create a single shared legend at the top of both subplots
fig.legend(all_lines, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=5, fontsize=14, frameon=False)

# margin between the subplots
plt.subplots_adjust(wspace=0.3)

# Adjust layout and save
plt.tight_layout()
# remove padding around the figure
plt.savefig('motivation.pdf', bbox_inches='tight', format='pdf')  # Added pad_inches to ensure legend is visible
plt.close()
