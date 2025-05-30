import pandas as pd
import matplotlib.pyplot as plt
import math

# Read the CSV data into a pandas DataFrame
csv_file_name = "result.csv"
data = pd.read_csv(csv_file_name)

# Convert appropriate columns to numeric and boolean types
numeric_cols = [
    'temperature', 'request_rate', 'draft_size', 'budget_token', 'budget_seq',
    'selective_validation_threshold', 'p50_ttft', 'p99_ttft', 'p50_tpot', 'p99_tpot',
    'p50_token_latency', 'p99_token_latency', 'token_throughput',
    'request_throughput', 'token_latency'
]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)

boolean_cols = ['colocate', 'consolidated_attention', 'preempt_flag']
data[boolean_cols] = data[boolean_cols].astype(bool)

# Filter data for request rate 12
data = data[data['request_rate'] == 12]

# Ensure that the data is sorted by selective_validation_threshold for plotting
data = data.sort_values(by='selective_validation_threshold')

# Get unique temperatures
temperatures = [0, 0.25, 0.5, 0.75, -1]  # Including -1 for random

# Define color palette for temperatures
colors = ['#D7E2F9', '#72A0FF', '#3B82F6', '#3864B9', '#1B345F']  # Updated bluish palette
colors = ['#D7E2F9', '#A0C4FF', '#4F8EFF', '#1C6DD0', '#003366']  # More distinguishable bluish palette

temp_color_map = dict(zip(temperatures, colors))

# Plot settings
markers = ['o', 's', '^', 'D', 'v']  # Different markers for temperatures
marker_map = dict(zip(temperatures, markers))

# Initialize the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(6, 2.8), sharex=True)
fig.subplots_adjust(hspace=0.3)

# Set x-ticks with a granularity of 0.1
x_ticks = [round(x * 0.1, 1) for x in range(0, 11)]  # Adjust this based on the range of your selective_validation_threshold values

# Plot Token Throughput vs. Drop Threshold
ax_throughput = axes[0]
ax_throughput.set_xticks(x_ticks)  # Set x-ticks with 0.1 granularity

for temp in temperatures:
    # Filter data for this temperature
    temp_data = data[data['temperature'] == temp]
    if temp_data.empty:
        continue  # Skip if no data for this temperature

    label = f'Temp {temp}' if temp != -1 else 'Random'

    # Plot the thicker black line as the "edge"
    ax_throughput.plot(
        temp_data['selective_validation_threshold'], temp_data['request_throughput'],
        marker=marker_map[temp], color='black', linestyle='-', linewidth=2, alpha=0.7
    )

    # Overlay the actual colored line with a thinner width
    ax_throughput.plot(
        temp_data['selective_validation_threshold'], temp_data['request_throughput'],
        marker=marker_map[temp], label=f'{label}', color=temp_color_map[temp],
        linestyle='-', linewidth=1.5, markersize=6, markeredgewidth=1, markeredgecolor='black'
    )

ax_throughput.set_ylabel('Request Throughput\n(requests/s)', fontsize=10)
ax_throughput.grid(True, linestyle='--', linewidth=0.5)
ax_throughput.set_xlabel('Drop Threshold', fontsize=10)

# Plot Token Latency vs. Drop Threshold
ax_latency = axes[1]
ax_latency.set_xticks(x_ticks)  # Set x-ticks with 0.1 granularity

for temp in temperatures:
    # Filter data for this temperature
    temp_data = data[data['temperature'] == temp]
    if temp_data.empty:
        continue  # Skip if no data for this temperature
    label = f'Temp {temp}' if temp != -1 else 'Random'

    # Plot the thicker black line as the "edge"
    ax_latency.plot(
        temp_data['selective_validation_threshold'], temp_data['token_latency'],
        marker=marker_map[temp], color='black', linestyle='-', linewidth=2, alpha=0.7
    )

    # Overlay the actual colored line with a thinner width
    ax_latency.plot(
        temp_data['selective_validation_threshold'], temp_data['token_latency'],
        marker=marker_map[temp], label=f'{label}', color=temp_color_map[temp],
        linestyle='-', linewidth=1.5, markersize=6, markeredgewidth=1, markeredgecolor='black'
    )

ax_latency.set_xlabel('Drop Threshold', fontsize=10)
ax_latency.set_ylabel('Token Latency (s/token)', fontsize=10)
ax_latency.grid(True, linestyle='--', linewidth=0.5)


# Add a single legend for temperatures with bigger font size and closer spacing
handles, labels = ax_throughput.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(temperatures),
           bbox_to_anchor=(0.5, 1.05), fontsize=11, frameon=False, columnspacing=0.5)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, 0.95])

# space between subplots
plt.subplots_adjust(wspace=0.36)

# Save and show the plot
plt.savefig('token_throughput_latency_vs_selective_validation_threshold_request_rate_12.pdf', bbox_inches='tight', format='pdf')
plt.show()