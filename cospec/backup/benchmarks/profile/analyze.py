import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers from a DataFrame column using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def plot_model_and_calibration(df, num_tokens_col, total_seq_len_col, time_col, title, output_path):
    """Create plots showing the model and calibration plot."""
    # Prepare data for regression
    x = np.column_stack((
        df[num_tokens_col].values,
        df[total_seq_len_col].values,
        df[total_seq_len_col].values ** 2  # Quadratic term for total_seq_len
    ))
    y = df[time_col].values.reshape(-1, 1)
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(x, y)
    y_pred = reg.predict(x)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Model fit using num_tokens as x-axis
    sns.scatterplot(data=df, x=num_tokens_col, y=time_col, alpha=0.5, label='Data points', ax=ax1)
    sort_idx = np.argsort(x[:, 0])
    ax1.plot(x[sort_idx, 0], y_pred[sort_idx], 'r-', 
             label=f'Model (R²={r2:.3f})')
    ax1.set_title(f'{title}\nModel Fit')
    ax1.set_xlabel(num_tokens_col)
    ax1.set_ylabel(time_col)
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Calibration plot
    sns.scatterplot(x=y.flatten(), y=y_pred.flatten(), alpha=0.5, ax=ax2)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax2.set_title('Calibration Plot')
    ax2.set_xlabel('Actual Time')
    ax2.set_ylabel('Predicted Time')
    ax2.legend()
    ax2.grid(True)
    
    # Add model equation to the figure
    plt.figtext(0.5, 0.01, 
                f"Equation: {time_col} = {reg.intercept_[0]:.3f} + {reg.coef_[0][0]:.3f}*{num_tokens_col} + {reg.coef_[0][1]:.3f}*{total_seq_len_col} + {reg.coef_[0][2]:.3f}*{total_seq_len_col}²", 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Print model details
    print(f"\nModel Details for {title}:")
    print(f"R² Score: {r2:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Equation: {time_col} = {reg.intercept_[0]:.3f} + {reg.coef_[0][0]:.3f}*{num_tokens_col} + {reg.coef_[0][1]:.3f}*{total_seq_len_col} + {reg.coef_[0][2]:.3f}*{total_seq_len_col}²")
    
    return reg

def analyze_profile_data(csv_path: str):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Print initial data info
    print("\nInitial Data Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Create output directory for plots
    output_dir = Path(csv_path).parent / "analysis_plots"
    output_dir.mkdir(exist_ok=True)
    
    # Convert 'None' strings to actual NaN values
    df = df.replace('None', np.nan)
    
    # Convert numeric columns to float
    numeric_columns = [
        'draft_time', 'target_decode_time', 'target_prefill_time',
        'draft_num_tokens', 'draft_max_seq_len', 'draft_total_seq_len',
        'target_num_tokens', 'target_max_seq_len', 'target_total_seq_len'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Print data info after conversion
    print("\nData Info after conversion:")
    print(df.info())
    
    # Print number of non-NaN values for each column
    print("\nNumber of non-NaN values for each column:")
    print(df.count())
    
    # Create plots for draft model
    draft_df = df[df['draft_time'].notna()]
    if not draft_df.empty:
        # Remove outliers from draft time
        draft_df = remove_outliers(draft_df, 'draft_time')
        print("\nDraft model data points after outlier removal:", len(draft_df))
        
        # Plot model and calibration
        plot_model_and_calibration(
            draft_df, 'draft_num_tokens', 'draft_total_seq_len', 'draft_time',
            'Draft Time Model (Outliers Removed)',
            output_dir / 'draft_time_model.png'
        )
        
        # Calculate correlations
        print("\nDraft Model Correlations (Outliers Removed):")
        draft_corr = draft_df[['draft_time', 'draft_num_tokens', 'draft_total_seq_len']].corr()
        print(draft_corr)
    
    # Create plots for target decode model
    target_decode_df = df[df['target_decode_time'].notna()]
    if not target_decode_df.empty:
        # Remove outliers from target decode time
        target_decode_df = remove_outliers(target_decode_df, 'target_decode_time')
        print("\nTarget decode model data points after outlier removal:", len(target_decode_df))
        
        # Plot model and calibration
        plot_model_and_calibration(
            target_decode_df, 'target_num_tokens', 'target_total_seq_len', 'target_decode_time',
            'Target Decode Time Model (Outliers Removed)',
            output_dir / 'target_decode_time_model.png'
        )
        
        # Calculate correlations
        print("\nTarget Decode Model Correlations (Outliers Removed):")
        target_decode_corr = target_decode_df[['target_decode_time', 'target_num_tokens', 'target_total_seq_len']].corr()
        print(target_decode_corr)
    
    # Create plots for target prefill model
    target_prefill_df = df[df['target_prefill_time'].notna()]
    if not target_prefill_df.empty:
        # Remove outliers from target prefill time
        target_prefill_df = remove_outliers(target_prefill_df, 'target_prefill_time')
        print("\nTarget prefill model data points after outlier removal:", len(target_prefill_df))
        
        # Plot model and calibration
        plot_model_and_calibration(
            target_prefill_df, 'target_num_tokens', 'target_total_seq_len', 'target_prefill_time',
            'Target Prefill Time Model (Outliers Removed)',
            output_dir / 'target_prefill_time_model.png'
        )
        
        # Calculate correlations
        print("\nTarget Prefill Model Correlations (Outliers Removed):")
        target_prefill_corr = target_prefill_df[['target_prefill_time', 'target_num_tokens', 'target_total_seq_len']].corr()
        print(target_prefill_corr)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_profile.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    analyze_profile_data(csv_path)