import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# Read all CSV files in the directory
csv_files = glob.glob('*.csv')

# Create a figure
plt.figure(figsize=(100, 6))

# Calculate max tokens once
max_tokens = max([pd.read_csv(f, header=None, names=['metric', 'value'])[pd.read_csv(f, header=None, names=['metric', 'value'])['metric'] == 'target_num_tokens']['value'].max() for f in csv_files])
max_tokens = int(max_tokens)

# Process each CSV file
for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file, header=None, names=['metric', 'value'])
    
    # Filter for target_num_tokens
    token_data = df[df['metric'] == 'target_num_tokens']['value'].values
    
    # Remove first 100 numbers (you can adjust this number)
    token_data = token_data[100:]
    
    # Count occurrences of each token value
    value_counts = pd.Series(token_data).value_counts().sort_index()
    
    # Plot the actual counts as a line
    plt.plot(value_counts.index, value_counts.values, 
             marker='o', linestyle='-', alpha=0.7,
             label=os.path.splitext(csv_file)[0])

# Add vertical lines at 8-token intervals for better granularity
for x in range(0, max_tokens + 8, 8):
    plt.axvline(x=x, color='gray', linestyle='--', alpha=0.3)

# Set x-axis ticks to multiples of 8
plt.xticks(np.arange(0, max_tokens + 8, 8))

plt.xlabel('Number of Tokens')
plt.ylabel('Count')  # Changed from 'Density' to 'Count'
plt.title('Distribution of Token Numbers')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('token_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
