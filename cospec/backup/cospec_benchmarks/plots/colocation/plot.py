import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

def plot_speedup_heatmap_from_csv(csv_file):
    """Plot heatmap of speedup ratio between colocation and non-colocation modes from CSV data"""
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Group results by batch_size and num_speculative_tokens
    results_dict = {}
    for _, row in df.iterrows():
        key = (row['batch_size'], row['num_speculative_tokens'])
        if key not in results_dict:
            results_dict[key] = {'colocation': [], 'non_colocation': []}
        
        if row['colocation_mode']:
            results_dict[key]['colocation'].append(row['mean_step_time'])
        else:
            results_dict[key]['non_colocation'].append(row['mean_step_time'])
    
    # Get unique batch sizes and spec token numbers
    batch_sizes = sorted(set(k[0] for k in results_dict.keys()))
    spec_tokens = sorted(set(k[1] for k in results_dict.keys()), reverse=True)
    
    # Create speedup matrix
    speedup_matrix = np.zeros((len(spec_tokens), len(batch_sizes)))
    
    for i, num_spec_tokens in enumerate(spec_tokens):
        for j, batch_size in enumerate(batch_sizes):
            key = (batch_size, num_spec_tokens)
            if key in results_dict:
                colocation_times = results_dict[key]['colocation']
                non_colocation_times = results_dict[key]['non_colocation']
                
                if colocation_times and non_colocation_times:
                    avg_colocation = np.mean(colocation_times)
                    avg_non_colocation = np.mean(non_colocation_times)
                    speedup_matrix[i, j] = avg_non_colocation / avg_colocation
    
    # Create heatmap
    plt.figure(figsize=(8, 3))
    ax = sns.heatmap(speedup_matrix, 
               xticklabels=batch_sizes,
               yticklabels=spec_tokens,
               cmap='YlGnBu',
               center=1.0, 
               annot=False,
               fmt='.2f',  
               cbar_kws={'label': 'Step Latency Speedup'})
    # Increase colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Step Latency Speedup', fontsize=14)
    
    # Add smooth contour line at 1.0
    smoothed_matrix = gaussian_filter(speedup_matrix, sigma=0.7)
    plt.contour(smoothed_matrix, levels=[1.0], colors='red', linewidths=2, linestyles='dashed')
    
    # Add text annotations
    plt.text(14, 4, 'Colocation Better', 
             ha='right', va='center', color='white', fontweight='bold', fontsize=14)
    plt.text(0.5, 6.2, 'Non-colocation Better', 
             ha='left', va='center', color='black', fontweight='bold', fontsize=14)
    
    plt.xlabel('Batch Size', fontsize=14)
    plt.ylabel('Speculation Length', fontsize=14)
    
    # Rotate x-axis tick labels
    plt.xticks(rotation=45)
    
    # Save plot
    plot_file = os.path.join(os.path.dirname(csv_file), "speedup_heatmap.pdf")
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    

if __name__ == "__main__":
    # Plot the heatmap
    csv_file = os.path.join(os.path.dirname(__file__), "results.csv")
    plot_speedup_heatmap_from_csv(csv_file)