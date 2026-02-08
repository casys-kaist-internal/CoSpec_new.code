import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import os

# Set matplotlib to use PDF backend for better quality
plt.switch_backend('Agg')

def load_data(data_dir='data'):
    """Load all CSV files from the data directory."""
    data = {}
    
    # Load calibration data
    calib_path = os.path.join(data_dir, 'calibration_data.csv')
    if os.path.exists(calib_path):
        data['calibration'] = pd.read_csv(calib_path)
    
    # Load ROC data
    roc_path = os.path.join(data_dir, 'roc_data.csv')
    if os.path.exists(roc_path):
        data['roc'] = pd.read_csv(roc_path)
    
    # Load metrics
    metrics_path = os.path.join(data_dir, 'metrics.csv')
    if os.path.exists(metrics_path):
        data['metrics'] = pd.read_csv(metrics_path)
    
    return data

def plot_calibration_curve(ax, calib_data, metrics_data=None):
    """Plot calibration curve."""
    # Plot calibration curve
    ax.plot(calib_data['bin_center'], calib_data['actual_mean'], 
            'o-', color='#1f77b4', linewidth=3, markersize=8, 
            label='Model Calibration')
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], '--', color='red', linewidth=2, 
            label='Perfect Calibration')
    
    ax.set_xlabel('Predicted Probability', fontsize=16)
    ax.set_ylabel('Actual Probability', fontsize=16)
    ax.set_title('Calibration Curve', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add text label for perfect calibration line
    ax.text(0.08, 0.18, 'Perfect Calibration (ECE=0)', rotation=28, fontsize=14, 
            color='red', fontweight='bold')
    
    # Add ECE value if available
    if metrics_data is not None and 'ECE' in metrics_data['metric'].values:
        ece_value = metrics_data[metrics_data['metric'] == 'ECE']['value'].iloc[0]
        ax.text(0.58, 0.1, f'ECE={ece_value:.4f}', 
                transform=ax.transAxes, fontsize=14, fontweight='bold', color='#1f77b4',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

def plot_roc_curve(ax, roc_data, metrics_data=None):
    """Plot ROC curve."""
    # Plot ROC curve
    ax.plot(roc_data['fpr'], roc_data['tpr'], 
            color='#1f77b4', linewidth=3, label='ROC Curve')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], '--', color='red', linewidth=2, 
            label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title('ROC Curve', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add text label for random classifier line
    ax.text(0.01, 0.1, 'Random Classifier (AUROC=0.5)', rotation=28, fontsize=14, 
            color='red', fontweight='bold')
    
    # Add AUROC value if available
    if metrics_data is not None and 'AUROC' in metrics_data['metric'].values:
        auroc_value = metrics_data[metrics_data['metric'] == 'AUROC']['value'].iloc[0]
        ax.text(0.48, 0.1, f'AUROC={auroc_value:.4f}', color='#1f77b4',
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

def create_plots(data):
    """Create subplots for calibration and ROC curves."""
    # Set up the figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    
    # Plot calibration curve
    if 'calibration' in data:
        plot_calibration_curve(ax1, data['calibration'], data.get('metrics'))
    else:
        ax1.text(0.5, 0.5, 'Calibration data not found', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    
    # Plot ROC curve
    if 'roc' in data:
        plot_roc_curve(ax2, data['roc'], data.get('metrics'))
    else:
        ax2.text(0.5, 0.5, 'ROC data not found', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def main():
    """Main function to load data and create plots."""
    # Load data
    data = load_data()
    
    if not data:
        print("No data files found in the data directory!")
        return
    
    # Create plots
    fig = create_plots(data)
    
    # Save the plot in PDF format
    output_path = 'calibration_roc_plots.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plots saved to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)

if __name__ == "__main__":
    main()
