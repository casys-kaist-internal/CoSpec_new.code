import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text # For adjusting text labels to prevent overlap
from sklearn.linear_model import LinearRegression

def create_presentation_chart():
    """
    Generates a visually effective, presentation-ready chart showing the top GPU
    Tensor Core performance growth over time, including a trend line for Tensor
    Core performance speedup, and annotating all top performance dots.
    """
    try:
        # --- 1. Load and Clean Data ---
        df = pd.read_csv('hardware.csv')

        # Convert relevant columns to numeric, coercing errors to NaN
        df['Release year'] = pd.to_numeric(df['Release year'], errors='coerce')
        # We only need Tensor Performance for this version
        df['FP16 Tensor Performance (FLOP/s)'] = pd.to_numeric(df['FP16 Tensor Performance (FLOP/s)'], errors='coerce')

        # Drop rows where essential plotting data is missing (release year or name)
        df.dropna(subset=['Release year', 'Name of the hardware'], inplace=True)

        # Cast year to integer for clean axis labels
        df['Release year'] = df['Release year'].astype(int)

        # Convert FLOP/s to TFLOP/s for more readable y-axis labels
        df['FP16 Tensor Performance (TFLOP/s)'] = df['FP16 Tensor Performance (FLOP/s)'] / 1e12

        # --- Filter for Top Tensor Core Performance per Year ---
        # Ensure we have a copy to avoid SettingWithCopyWarning
        df_tensor_filtered = df.dropna(subset=['FP16 Tensor Performance (TFLOP/s)']).copy()
        
        # Group by year and get the index of the maximum Tensor Performance for each year
        idx = df_tensor_filtered.groupby('Release year')['FP16 Tensor Performance (TFLOP/s)'].idxmax()
        # Select only these top-performing GPUs for each year
        df_tensor_top = df_tensor_filtered


        fig, ax = plt.subplots(figsize=(8, 4)) # Increased figure size for better visual appeal and space


        # --- 3. Plot Data Points (Only Top Tensor Core Performance) ---
        ax.scatter(
            df_tensor_top['Release year'],
            df_tensor_top['FP16 Tensor Performance (TFLOP/s)'],
            c='#1f77b4', # A distinct orange color
            marker='o',
        )

        # # --- 4. Add Linear Trend Line for Tensor Core Performance ---
        # if not df_tensor_top.empty:
        #     # Perform linear regression for Top Tensor Core performance
        #     slope, intercept = np.polyfit(
        #         df_tensor_top['Release year'],
        #         df_tensor_top['FP16 Tensor Performance (TFLOP/s)'],
        #         1
        #     )

        #     # Generate points for the trend line
        #     trend_years = np.array([df_tensor_top['Release year'].min(), df_tensor_top['Release year'].max()])
        #     trend_performance = slope * trend_years + intercept

        #     # Plot the trend line
        #     ax.plot(
        #         trend_years,
        #         trend_performance,
        #         color='red',
        #         linestyle='--', # Dashed line for trend
        #     )
        # else:
        #     print("Not enough data to draw Tensor Core trend line.")

        # --- 4. Add Exponential Trend Line for Tensor Core Performance ---
        if not df_tensor_top.empty:
            # Perform log-linear regression for exponential growth modeling
            X = df_tensor_top['Release year'].values.reshape(-1, 1)
            y = np.log(df_tensor_top['FP16 Tensor Performance (TFLOP/s)'])
            
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            intercept = reg.intercept_

            # Generate points for the exponential trend line
            trend_years = np.array([df_tensor_top['Release year'].min(), 
                                df_tensor_top['Release year'].max()])
            trend_performance = np.exp(intercept + slope * trend_years)

            # Plot the exponential trend line
            ax.plot(
                trend_years,
                trend_performance,
                color='red',
                linestyle='--',
                label='Exponential Trend'
            )
            
            # Calculate annual speedup factor from continuous growth rate
            annual_speedup = np.exp(slope)
            
            # Format the speedup annotation
            speedup_text = f"{annual_speedup:.2f}x/year"
            print(f"Continuous annual speedup: {annual_speedup:.2f}x/year")

            # # Calculate midpoint for annotation
            # mid_year = np.mean(trend_years)
            # mid_perf = np.exp(intercept + slope * mid_year)
            
            # ax.annotate(
            #     speedup_text,
            #     xy=(mid_year, mid_perf),
            #     xytext=(mid_year + 0.5, mid_perf * 1.2),
            #     arrowprops=dict(
            #         arrowstyle='->',
            #         color='red',
            #         lw=1.5,
            #         connectionstyle="arc3,rad=0.2"
            #     ),
            #     fontsize=12,
            #     fontweight='bold',
            #     color='red',
            #     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8)
            # )


        # --- 5. Add Labels for All Plotted Tensor Core GPUs ---
        texts = []
        for _, gpu in df_tensor_top.iterrows(): # Iterate over the filtered DataFrame
            performance_to_label = gpu['FP16 Tensor Performance (TFLOP/s)']
            
            label_text = gpu['Name of the hardware']

            # skip 2080 Ti
            if gpu['Name of the hardware'] == 'NVIDIA GeForce RTX 2080 Ti':
                continue

            if pd.notna(performance_to_label):
                label_text += f"\n({performance_to_label:.1f} TFLOP/s)" # No need for "Tensor" as it's implied

            if pd.notna(performance_to_label): # Only add text if a performance value exists
                # texts.append(
                #     ax.text(
                #         gpu['Release year'],
                #         performance_to_label, # Y-coordinate based on the chosen performance
                #         label_text,
                #         fontsize=8,
                #         color='#222222'
                #     )
                # )
                text = plt.annotate(
                    gpu['Name of the hardware'], 
                    xy=(gpu['Release year'], performance_to_label),
                    xytext=(gpu['Release year'], performance_to_label * 1.1),
                    fontsize=8,  # Small font size
                    color='#222222',
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.7
                    ),
                    ha='left',
                    va='bottom',
                    arrowprops=dict(
                        arrowstyle='->',
                        color='gray',
                        lw=0.6,
                        alpha=0.6,
                        connectionstyle="arc3,rad=0.1"
                    )
                )
                texts.append(text)

        # # Use adjust_text to prevent labels from overlapping and draw connecting lines
        # adjust_text(
        #     texts,
        #     arrowprops=dict(
        #         arrowstyle='-', # Simple straight line
        #         color='gray', # Lighter color for arrows
        #         lw=0.7, # Line width
        #         connectionstyle="arc3,rad=0.0" # Ensures a straight line connection
        #     ),
        #     # Add some additional force to prevent overlaps (use the plotted points)
        #     add_objects=[ax.scatter(df_tensor_top['Release year'], df_tensor_top['FP16 Tensor Performance (TFLOP/s)'], s=1)]
        # )

        # --- 6. Customize Axes and Title ---
        ax.set_xlabel('Release Year', fontsize=12, labelpad=10)
        ax.set_ylabel('FP16 Tensor Core Perf (TFLOP/s)', fontsize=12, labelpad=10)
        # ax.set_title('Top GPU Tensor Core Performance Growth Over Time', fontsize=16, pad=15, weight='bold')

        # Set x-axis ticks to show only the years present in the data points
        ax.set_xticks(df_tensor_top['Release year'].unique()) 

        # set y lim to 0 to auto
        # ax.set_ylim(0, df_tensor_top['FP16 Tensor Performance (TFLOP/s)'].max() * 1.1)

        # log scale 
        ax.set_yscale('log')

        # Add a light grid for readability
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='lightgray', fancybox=True, shadow=True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

        # --- 7. Save the Output ---\
        # in pdf format
        output_filename = 'top_gpu_tensor_performance.pdf'
        plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight') # High DPI for presentations

        print(f"Presentation-ready chart saved as '{output_filename}'")

    except FileNotFoundError:
        print("Error: 'hardware.csv' not found. Please ensure it's in the same directory.")
    except KeyError as ke:
        print(f"Error: Missing expected column in 'hardware.csv'. Please check your CSV for: {ke}")
        print("Expected columns include 'Release year', 'FP16 Tensor Performance (FLOP/s)', 'Name of the hardware'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function to create the chart
create_presentation_chart()