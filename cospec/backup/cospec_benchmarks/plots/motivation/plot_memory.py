import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text # For adjusting text labels to prevent overlap
from sklearn.linear_model import LinearRegression

def create_memory_plots():
    """
    Generates two separate presentation-ready charts showing:
    1. GPU Memory size growth over time
    2. GPU Memory bandwidth growth over time
    Both include trend lines and annotated data points for GPUs with FP16 Tensor Performance only.
    """
    try:
        # --- 1. Load and Clean Data ---
        df = pd.read_csv('hardware.csv')

        # Convert relevant columns to numeric, coercing errors to NaN
        df['Release year'] = pd.to_numeric(df['Release year'], errors='coerce')
        df['Memory size per board (Byte)'] = pd.to_numeric(df['Memory size per board (Byte)'], errors='coerce')
        df['Memory Bandwidth (Byte/s)'] = pd.to_numeric(df['Memory Bandwidth (Byte/s)'], errors='coerce')
        df['FP16 Tensor Performance (FLOP/s)'] = pd.to_numeric(df['FP16 Tensor Performance (FLOP/s)'], errors='coerce')

        # Release year from 2017 to 2025
        df = df[df['Release year'] >= 2017]

        # Drop rows where essential plotting data is missing
        df.dropna(subset=['Release year', 'Name of the hardware'], inplace=True)

        # Cast year to integer for clean axis labels
        df['Release year'] = df['Release year'].astype(int)

        # Convert memory size to GB for readability
        df['Memory size (GB)'] = df['Memory size per board (Byte)'] / (1000**3)
        
        # Convert memory bandwidth to GB/s for readability
        df['Memory Bandwidth (GB/s)'] = df['Memory Bandwidth (Byte/s)'] / (1000**3)

        # Convert FP16 Tensor Performance to TFLOP/s for readability
        df['FP16 Tensor Performance (TFLOP/s)'] = df['FP16 Tensor Performance (FLOP/s)'] / 1e12

        # Filter data for each plot
        df_memory_size = df.dropna(subset=['Memory size (GB)']).copy()
        df_memory_bandwidth = df.dropna(subset=['Memory Bandwidth (GB/s)']).copy()

        # Identify GPUs with FP16 Tensor Performance
        df_memory_size['has_tensor_perf'] = df_memory_size['FP16 Tensor Performance (TFLOP/s)'].notna()
        df_memory_bandwidth['has_tensor_perf'] = df_memory_bandwidth['FP16 Tensor Performance (TFLOP/s)'].notna()

        # --- Plot 1: Memory Size ---
        fig1, ax1 = plt.subplots(figsize=(8, 4))

        # Plot all GPUs (smaller, lighter points)
        ax1.scatter(
            df_memory_size['Release year'],
            df_memory_size['Memory size (GB)'],
            c='#1f77b4',
            marker='o',
        )

        # # Plot GPUs with Tensor Performance (larger, colored points)
        tensor_size = df_memory_size[df_memory_size['has_tensor_perf']]
        # ax1.scatter(
        #     tensor_size['Release year'],
        #     tensor_size['Memory size (GB)'],
        #     c='#2E86AB',  # Blue color
        #     marker='o',
        #     s=80,
        #     label='GPUs with Tensor Performance'
        # )

        # Add exponential trend line for memory size
        if not df_memory_size.empty:
            X = df_memory_size['Release year'].values.reshape(-1, 1)
            y = np.log(df_memory_size['Memory size (GB)'])
            
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            intercept = reg.intercept_

            trend_years = np.array([df_memory_size['Release year'].min(), 
                                   df_memory_size['Release year'].max()])
            trend_size = np.exp(intercept + slope * trend_years)

            ax1.plot(
                trend_years,
                trend_size,
                color='red',
                linestyle='--',
                label='Exponential Trend'
            )
            
            annual_growth = np.exp(slope)
            print(f"Memory size annual growth: {annual_growth:.2f}x/year")

        # Add labels for GPUs with Tensor Performance only
        texts1 = []
        for _, gpu in tensor_size.iterrows():
            memory_size = gpu['Memory size (GB)']
            tensor_perf = gpu['FP16 Tensor Performance (TFLOP/s)']
            label_text = gpu['Name of the hardware']
            if pd.notna(memory_size):
                label_text += f"\n({memory_size:.1f} GB)"

            if pd.notna(memory_size):
                # texts1.append(
                #     ax1.text(
                #         gpu['Release year'],
                #         memory_size,
                #         label_text,
                #         fontsize=8,
                #         color='#222222',
                #     )
                # )
                text = plt.annotate(
                    gpu['Name of the hardware'], 
                    xy=(gpu['Release year'], memory_size),
                    xytext=(gpu['Release year'], memory_size * 1.1),
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
                texts1.append(text)

        # if texts1:  # Only adjust if there are texts to adjust
        #     adjust_text(
        #         texts1,
        #         arrowprops=dict(
        #             arrowstyle='-',
        #             color='gray',
        #             lw=0.7,
        #             connectionstyle="arc3,rad=0.0"
        #         ),
        #         add_objects=[ax1.scatter(tensor_size['Release year'], tensor_size['Memory size (GB)'], s=1)]
        #     )

        # Customize memory size plot
        ax1.set_xlabel('Release Year', fontsize=12, labelpad=10)
        ax1.set_ylabel('Memory Size (GB)', fontsize=12, labelpad=10)
        ax1.set_xticks(df_memory_size['Release year'].unique())
        # ax1.set_ylim(0, df_memory_size['Memory size (GB)'].max() * 1.1)
        ax1.set_yscale('log')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Save memory size plot
        plt.tight_layout()
        output_filename1 = 'gpu_memory_size_analysis.pdf'
        plt.savefig(output_filename1, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Memory size chart saved as '{output_filename1}'")

        # --- Plot 2: Memory Bandwidth ---
        fig2, ax2 = plt.subplots(figsize=(8, 4))

        # Plot all GPUs (smaller, lighter points)
        ax2.scatter(
            df_memory_bandwidth['Release year'],
            df_memory_bandwidth['Memory Bandwidth (GB/s)'],
            c='#1f77b4',
            marker='o',
        )

        # Plot GPUs with Tensor Performance (larger, colored points)
        tensor_bandwidth = df_memory_bandwidth[df_memory_bandwidth['has_tensor_perf']]

        # Add exponential trend line for memory bandwidth
        if not df_memory_bandwidth.empty:
            X = df_memory_bandwidth['Release year'].values.reshape(-1, 1)
            y = np.log(df_memory_bandwidth['Memory Bandwidth (GB/s)'])
            
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            intercept = reg.intercept_

            trend_years = np.array([df_memory_bandwidth['Release year'].min(), 
                                   df_memory_bandwidth['Release year'].max()])
            trend_bandwidth = np.exp(intercept + slope * trend_years)

            ax2.plot(
                trend_years,
                trend_bandwidth,
                color='red',
                linestyle='--',
                label='Exponential Trend'
            )
            
            annual_growth = np.exp(slope)
            print(f"Memory bandwidth annual growth: {annual_growth:.2f}x/year")

        # Add labels for GPUs with Tensor Performance only
        texts2 = []
        for _, gpu in tensor_bandwidth.iterrows():
            bandwidth = gpu['Memory Bandwidth (GB/s)']
            tensor_perf = gpu['FP16 Tensor Performance (TFLOP/s)']
            label_text = gpu['Name of the hardware']

            if label_text == 'NVIDIA GeForce RTX 3090 Ti':
                continue

            if label_text == 'AMD Radeon Instinct MI250X':
                continue

            if pd.notna(bandwidth):
                label_text += f"\n({bandwidth:.1f} GB/s)"

            if pd.notna(bandwidth):
                # texts2.append(
                #     ax2.text(
                #         gpu['Release year'],
                #         bandwidth,
                #         label_text,
                #         fontsize=8,
                #         color='#222222',
                #     )
                # )
                text = plt.annotate(
                    gpu['Name of the hardware'], 
                    xy=(gpu['Release year'], bandwidth),
                    xytext=(gpu['Release year'], bandwidth * 1.1),
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
                texts2.append(text)

        # if texts2:  # Only adjust if there are texts to adjust
        #     adjust_text(
        #         texts2,
        #         arrowprops=dict(
        #             arrowstyle='-',
        #             color='gray',
        #             lw=0.7,
        #             connectionstyle="arc3,rad=0.0"
        #         ),
        #         add_objects=[ax2.scatter(tensor_bandwidth['Release year'], tensor_bandwidth['Memory Bandwidth (GB/s)'], s=1)]
        #     )

        # Customize memory bandwidth plot
        ax2.set_xlabel('Release Year', fontsize=12, labelpad=10)
        ax2.set_ylabel('Memory Bandwidth (GB/s)', fontsize=12, labelpad=10)
        ax2.set_xticks(df_memory_bandwidth['Release year'].unique())
        # ax2.set_ylim(0, df_memory_bandwidth['Memory Bandwidth (GB/s)'].max() * 1.1)
        ax2.set_yscale('log')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Save memory bandwidth plot
        plt.tight_layout()
        output_filename2 = 'gpu_memory_bandwidth_analysis.pdf'
        plt.savefig(output_filename2, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Memory bandwidth chart saved as '{output_filename2}'")

        print("Both memory analysis charts have been created successfully!")

    except FileNotFoundError:
        print("Error: 'hardware.csv' not found. Please ensure it's in the same directory.")
    except KeyError as ke:
        print(f"Error: Missing expected column in 'hardware.csv'. Please check your CSV for: {ke}")
        print("Expected columns include 'Release year', 'Memory size per board (Byte)', 'Memory Bandwidth (Byte/s)', 'FP16 Tensor Performance (FLOP/s)', 'Name of the hardware'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function to create the charts
create_memory_plots()