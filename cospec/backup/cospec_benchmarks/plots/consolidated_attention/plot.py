import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

"""
Model Configurations:
OPT-6.7B: 32 heads, 128 head size
OPT-13B: 40 heads, 128 head size
OPT-30B: 56 heads, 128 head size
"""

def read_speedup_data(version):
    models = ['opt_6.7b', 'opt_13b', 'opt_30b']
    data = {}
    
    for model in models:
        file_path = f'results/{model}_v{version}_speedup.csv'
        df = pd.read_csv(file_path, index_col=0)
        # Select only batch sizes that are multiples of 32
        df = df[['32', '64', '96', '128']]
        df = df.loc[[2, 4, 6, 8]]
        # Transpose to make batch size the y-axis
        df = df.T
        # Reverse the order of batch sizes
        df = df.iloc[::-1]
        data[model] = df
    
    return data

def find_global_min_max():
    versions = [1, 2]
    models = ['opt_6.7b', 'opt_13b', 'opt_30b']
    all_values = []
    
    for version in versions:
        for model in models:
            file_path = f'results/{model}_v{version}_speedup.csv'
            df = pd.read_csv(file_path, index_col=0)
            all_values.extend(df.values.flatten())
    
    return min(all_values), max(all_values)

def plot_heatmaps(version):
    data = read_speedup_data(version)
    models = ['opt_6.7b', 'opt_13b', 'opt_30b']
    display_names = ['(a) OPT-6.7B (A6000)', '(b) OPT-13B (A100)', '(c) OPT-30B (H200)']
    model_configs = {
        'opt_6.7b': '# Heads: 32, Head Size: 128\nSeq Len: 1024',
        'opt_13b': '# Heads: 40, Head Size: 128\nSeq Len: 1024',
        'opt_30b': '# Heads: 56, Head Size: 128\nSeq Len: 1024'
    }
    
    # Find global max
    _, vmax = find_global_min_max()
    # Set fixed color scale
    vmin = 0  # Fixed minimum at -3
    vmax = max(vmax, 2.0)  # Ensure symmetric scale
    center = 1.0  # Center of the colormap
    
    # Create figure with a single colorbar
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])
    
    # Create subplots
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])
    
    # Create custom colormap with stronger colors
    colors = ["red", "white", "green"]  # Bright red, white, bright green
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=n_bins)
    
    for idx, (model, display_name) in enumerate(zip(models, display_names)):
        sns.heatmap(data[model], 
                   ax=axes[idx],
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   center=center,
                   cbar=idx==0,  # Only show colorbar for first plot
                   cbar_ax=cbar_ax if idx==0 else None,
                   cbar_kws={'label': 'Latency Speedup'},
                   xticklabels=[1, 3, 5, 7],  # Speculative window sizes (subtracted 1)
                   yticklabels=data[model].index if idx==0 else [],  # Only show y-labels for first plot
                   annot=True,  # Show values
                   fmt='.2f',  # Format values to 2 decimal places
                   square=True,  # Make cells square
                   linewidths=0.5,  # Add borders
                   linecolor='black',  # Border color
                   annot_kws={'size': 12})  # Increase annotation font size
        
        # Move x-axis labels to top
        axes[idx].xaxis.tick_top()
        axes[idx].xaxis.set_label_position('top')
        
        # Increase tick label font sizes
        axes[idx].tick_params(axis='both', which='major', labelsize=12)
        
        # Place the bold model name above the axis
        axes[idx].text(0.5, -0.34, display_name, fontsize=14, fontweight='bold', ha='center', va='bottom', transform=axes[idx].transAxes)
        # Set the config as the normal title
        axes[idx].text(0.5, -0.2, model_configs[model], fontsize=12, fontweight='normal', ha='center', va='bottom', transform=axes[idx].transAxes)
        axes[idx].set_xlabel('Speculation Size', fontsize=14)
        if idx == 0:  # Only show y-label for first plot
            axes[idx].set_ylabel('Batch Size', fontsize=14)
            # Set colorbar label font size
            cbar_ax.yaxis.label.set_size(14)
        else:
            axes[idx].set_ylabel('')
    
    # plt.tight_layout()
    plt.savefig(f'speedup_heatmaps_v{version}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create plots for both versions
    plot_heatmaps(1)
    plot_heatmaps(2)

if __name__ == "__main__":
    main() 