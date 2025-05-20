import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('tiling_results.csv')

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['num_tokens'], df['mean_latency'], 'b-o', linewidth=2, markersize=6)

# Customize the plot
plt.title('Mean Latency vs Number of Tokens', fontsize=14)
plt.xlabel('Number of Tokens', fontsize=12)
plt.ylabel('Mean Latency (seconds)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('latency_plot.png', dpi=300, bbox_inches='tight')
plt.close()
