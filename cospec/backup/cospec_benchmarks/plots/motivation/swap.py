import json
import os
import matplotlib.pyplot as plt

# Directory containing JSON files
directory = "data_swap/data_swap"  # Change this to the actual path
output_directory = "output"  # Directory to save plots
os.makedirs(output_directory, exist_ok=True)

# Read JSON files
files = sorted([f for f in os.listdir(directory) if f.endswith(".json")], reverse=True)

print(files)

# Page sizes corresponding to file numbers (assuming typical page sizes)
page_sizes = ["1", "2", "4", "8", "16", "32", "64", "128"]

benchmark_duration_values = []

for file in files:
    with open(os.path.join(directory, file), "r") as f:
        data = json.load(f)
        benchmark_duration_values.append(data["benchmark_summary"]["benchmark_duration_seconds"])

# Plot single graph
plt.figure(figsize=(8, 4))
plt.plot(page_sizes, benchmark_duration_values, marker='o', linestyle='-', linewidth=2, markersize=8, color='#659250')
plt.xlabel("Page Size", fontsize=12, fontweight='bold')
plt.ylabel("Benchmark Duration (seconds)", fontsize=12, fontweight='bold')
plt.title("Llama-2-7B on ShareGPT Dataset (100 Requests)", fontsize=14, fontweight='bold')
plt.xticks(page_sizes)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = os.path.join(output_directory, "benchmark_duration_vs_page_size.pdf")
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to: {output_path}")