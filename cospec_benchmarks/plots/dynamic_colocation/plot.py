import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# List of datasets to include in the plot
datasets_to_plot = ['colocation', 'without_colocation', 'dynamic_colocation']

# Read the latency CSV data into pandas DataFrames
latency_data = {}
for dataset in datasets_to_plot:
    csv_file = f"{dataset}.csv"
    if os.path.exists(csv_file):
        latency_data[dataset] = pd.read_csv(csv_file)
    else:
        print(f"Warning: {csv_file} not found")

# Define colors for each dataset
dataset_colors = {
    'colocation': '#228B22',
    'without_colocation': '#3864B9',
    'dynamic_colocation': 'red'
}

# Sort data by arrival_time_s and apply a moving average
window_size_latency = 30  # Smoothing window size
for key, df in latency_data.items():
    df.sort_values('arrival_time_s', inplace=True)
    df['Smoothed_Latency'] = df['token_latency_ms'].rolling(window=window_size_latency, center=True).mean()

# Set figure size
plt.figure(figsize=(6.5, 2.5))

# Plot the smoothed latencies for selected datasets
for key in datasets_to_plot:
    if key in latency_data:
        plt.plot(latency_data[key]['arrival_time_s'], latency_data[key]['Smoothed_Latency'],
                 label=f'{key.replace("_", " ").title()}', color=dataset_colors[key], linestyle='-', linewidth=1.2)

# Highlight request rate regions
plt.axvspan(0, 60, color='#e0f3f8', alpha=0.5)
plt.axvspan(60, 120, color='#3690c0', alpha=0.3)
plt.axvspan(120, 180, color='#e0f3f8', alpha=0.5)

# Annotate request rate regions
plt.text(30, 140, 'Low (4 req/s)', fontsize=10, color='black', ha='center', fontweight='bold')
plt.text(90, 140, 'High (10 req/s)', fontsize=10, color='black', ha='center', fontweight='bold')
plt.text(153, 140, 'Low (4 req/s)', fontsize=10, color='black', ha='center', fontweight='bold')

# List of inflection point times
# inflection_times = [61.268914790358394, 137.8050911310129]

# Highlight inflection points on dynamic colocation
# if 'dynamic_colocation' in datasets_to_plot:
#     dynamic_colocation_df = latency_data['dynamic_colocation']
    
#     for i, inflection_time in enumerate(inflection_times):
#         # Find the closest time in the dynamic colocation data
#         closest_row = dynamic_colocation_df.iloc[(dynamic_colocation_df['arrival_time_s'] - inflection_time).abs().argsort()[:1]]
#         inflection_latency = closest_row['Smoothed_Latency'].values[0]
        
#         annotation_text = "Coloc On" if i == 0 else "Coloc Off"
#         annotation_cords = (inflection_time - 13, inflection_latency) if i == 0 else (inflection_time - 13, inflection_latency - 0.012)
#         # Mark and annotate the inflection point
#         plt.scatter([inflection_time], [inflection_latency], color='black', zorder=5)
#         plt.text(annotation_cords[0], annotation_cords[1], annotation_text, fontsize=10, color='red', ha='center', va='bottom', fontweight='bold')

# Add labels and legend
plt.yscale('log')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Token Latency (ms)', fontsize=12)
plt.legend(loc='upper center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.3), frameon=False)
plt.xlim(0, 180)
plt.ylim(10, 220)

# Customize ticks
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Add the zoomed-in subplot
x_min, x_max = 25, 35  # Define x-range for the zoomed region
y_min, y_max = 11, 21  # Define y-range for the zoomed region

# Add a transparent box to annotate the zoomed region
buffer = 0.005
plt.gca().add_patch(plt.Rectangle(
    (25, 11),
    10,
    10,
    edgecolor='black',
    facecolor='none',
    linestyle='-',
    linewidth=1,
    zorder=10
))

plt.tick_params(axis='both', which='major', labelsize=10)  # Changed from 12 to 14


# Draw connecting lines
plt.plot([25, 10], [11, 31], color='black', linestyle=':', linewidth=1.2)
plt.plot([35, 32], [20, 30], color='black', linestyle=':', linewidth=1.2)

# Create inset axes
zoom_ax = plt.axes([0.15, 0.4, 0.1, 0.2])
for key in datasets_to_plot:
    if key in latency_data:
        zoom_ax.plot(latency_data[key]['arrival_time_s'], latency_data[key]['Smoothed_Latency'],
                     label=f'{key.replace("_", " ").title()}', color=dataset_colors[key])
zoom_ax.set_xlim(x_min, x_max)
zoom_ax.set_ylim(y_min, y_max)
zoom_ax.set_yscale('log')
zoom_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
zoom_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


# Adjust layout
plt.tight_layout()


# Save the plot
plt.savefig('dynamic_colocation.pdf', bbox_inches='tight', format='pdf')
plt.show()

# Calculate speedup in the sections
low_request_rate_latency_first = {}
low_request_rate_latency_third = {}
high_request_rate_latency = {}

for key in datasets_to_plot:
    if key in latency_data:
        # Average latency for the first low request rate segment (0–60 seconds)
        low_request_rate_latency_first[key] = latency_data[key][
            (latency_data[key]['arrival_time_s'] >= 0) & (latency_data[key]['arrival_time_s'] < 60)
        ]['Smoothed_Latency'].mean()
        
        # Average latency for the third low request rate segment (120–180 seconds)
        low_request_rate_latency_third[key] = latency_data[key][
            (latency_data[key]['arrival_time_s'] >= 120) & (latency_data[key]['arrival_time_s'] < 180)
        ]['Smoothed_Latency'].mean()
        
        # Average latency for the high request rate segment (60–120 seconds)
        high_request_rate_latency[key] = latency_data[key][
            (latency_data[key]['arrival_time_s'] >= 60) & (latency_data[key]['arrival_time_s'] < 120)
        ]['Smoothed_Latency'].mean()

# Calculate speedups
speedup_low_request_rate_first = {}
colocation_latency_first = low_request_rate_latency_first['colocation']
for key in datasets_to_plot:
    if key in low_request_rate_latency_first:
        speedup_low_request_rate_first[key] = colocation_latency_first / low_request_rate_latency_first[key]

speedup_low_request_rate_third = {}
colocation_latency_third = low_request_rate_latency_third['colocation']
for key in datasets_to_plot:
    if key in low_request_rate_latency_third:
        speedup_low_request_rate_third[key] = colocation_latency_third / low_request_rate_latency_third[key]

speedup_high_request_rate = {}
without_colocation_latency_high = high_request_rate_latency['without_colocation']
for key in datasets_to_plot:
    if key in high_request_rate_latency:
        speedup_high_request_rate[key] = without_colocation_latency_high / high_request_rate_latency[key]

# Print the speedup values
print("Speedup for First Low Request Rate (Compared to Colocation):")
for key, value in speedup_low_request_rate_first.items():
    print(f"{key.replace('_', ' ').title()}: {value:.2f}")

print("\nSpeedup for High Request Rate (Compared to Without Colocation):")
for key, value in speedup_high_request_rate.items():
    print(f"{key.replace('_', ' ').title()}: {value:.2f}")

print("\nSpeedup for Third Low Request Rate (Compared to Colocation):")
for key, value in speedup_low_request_rate_third.items():
    print(f"{key.replace('_', ' ').title()}: {value:.2f}")
