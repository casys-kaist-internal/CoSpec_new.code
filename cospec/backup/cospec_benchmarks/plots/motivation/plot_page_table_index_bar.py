import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('page_table_index.csv')

# Extract the data we need
context_lengths = df.iloc[6, 1:].values.astype(int)  # Context lengths (row 7, 0-indexed)
page_table_after = df.iloc[10, 1:].values.astype(float)  # Page table index after (row 11)
page_table_before = df.iloc[12, 1:].values.astype(float)  # Page table index before (row 13)

# Convert from bytes to MB
page_table_after_mb = page_table_after / (1024 * 1024)
page_table_before_mb = page_table_before / (1024 * 1024)

# Convert context lengths to readable format
context_labels = []
for ctx_len in context_lengths:
    if ctx_len >= 1000000:
        context_labels.append(f"{ctx_len//1000000}M")
    elif ctx_len >= 1000:
        context_labels.append(f"{ctx_len//1000}K")
    else:
        context_labels.append(str(ctx_len))

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 4))

# Set up bar positions
x = np.arange(len(context_lengths))
width = 0.35  # Width of the bars

# Create bars
bars1 = ax.bar(x - width/2, page_table_before_mb, width, label='Llama-3.1-70B (Page Size=16)', 
               color='#FEAE00', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, page_table_after_mb, width, label='MicroPage', 
               color='#659250', alpha=0.8, edgecolor='black', linewidth=0.5)

# Customize the plot
ax.set_xlabel('Context Length', fontsize=14, fontweight='bold')
ax.set_ylabel('Page Table Size per Request (MB)', fontsize=12, fontweight='bold')

# Set x-axis ticks
ax.set_xticks(x)
ax.set_xticklabels(context_labels, fontsize=12)

# Add legend on top without box
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=12, 
          frameon=False, ncol=2)

# Add grid
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# # Set y-axis to log scale for better visualization
ax.set_yscale('log')

# y limit automatically but y minimum not change only change y max
y_min = page_table_before_mb.min()
y_max = page_table_after_mb.max()

ax.set_ylim(y_min, y_max * 10)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('page_table_index_bar_chart.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print the data for verification
print("Context Lengths:", context_labels)
print("Page Table Index Before (MB):", page_table_before_mb)
print("Page Table Index After (MB):", page_table_after_mb) 