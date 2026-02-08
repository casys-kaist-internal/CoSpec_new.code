import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from datetime import datetime
import re

def create_context_length_chart():
    """
    Generates a chart showing the evolution of LLM context length over time.
    """
    try:
        # --- 1. Load and Clean Data ---
        df = pd.read_csv('model_context_length.csv')
        
        # Clean the data - remove empty rows and rows with missing essential data
        df = df.dropna(subset=['LLM', 'Input Context Size'])
        
        # Clean the release date column - extract year from various formats
        def extract_year(date_str):
            if pd.isna(date_str):
                return None
            # Look for 4-digit year patterns
            year_match = re.search(r'20\d{2}', str(date_str))
            if year_match:
                return int(year_match.group())
            return None
        
        df['Release Year'] = df['Release Date'].apply(extract_year)
        
        # Clean context size - handle comma-separated numbers and convert to numeric
        def clean_context_size(size_str):
            if pd.isna(size_str):
                return None
            # Remove commas and convert to numeric
            size_str = str(size_str).replace(',', '')
            try:
                return float(size_str)
            except:
                return None
        
        df['Context Size'] = df['Input Context Size'].apply(clean_context_size)
        
        # Filter out rows with missing year or context size
        df = df.dropna(subset=['Release Year', 'Context Size'])
        
        # Convert context size to thousands for better readability
        df['Context Size'] = df['Context Size']
        
        # Sort by release year
        df = df.sort_values('Release Year')
        
        print(f"Loaded {len(df)} models with valid data")
        print("\nData preview:")
        print(df[['LLM', 'Release Year', 'Context Size', 'Company']].head(10))
        
        # --- 2. Create the Plot ---
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Color map for companies
        companies = df['Company'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(companies)))
        company_colors = dict(zip(companies, colors))
        
        # --- 3. Plot Data Points ---
        for company in companies:
            company_data = df[df['Company'] == company]
            ax.scatter(
                company_data['Release Year'],
                company_data['Context Size'],
                c='#1f77b4',
                marker='o'
            )
        
        # --- 4. Add Trend Line ---
        if not df.empty:
            # Perform log-linear regression for exponential growth modeling
            X = df['Release Year'].values.reshape(-1, 1)
            y = np.log(df['Context Size'])
            
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            
            # Generate points for the exponential trend line
            trend_years = np.array([df['Release Year'].min(), df['Release Year'].max()])
            trend_size = np.exp(intercept + slope * trend_years)
            
            # Plot the exponential trend line
            ax.plot(
                trend_years,
                trend_size,
                color='red',
                linestyle='--',
            )
        
        # --- 5. Add Labels for Notable Models ---
        texts = []
        notable_models = ['GPT-1', 'T5 (base)', 'GPT-2', 'GPT-3', 'GPT-4', 'Claude 1.2', 'GPT-3.5 Turbo',  'Claude 2.1', 'Gemini 1.0', 'Gemini 1.5', 'Gemini 1.5 Pro 2M', 'Llama 4 Scout', 'DeepSeek-R1', 'DeepSeek-V3',  'Claude Sonnet 3.7',  'GPT-Neo',  'BLOOM', 'Llama', 'Qwen3']
        
        for _, model in df.iterrows():    
            if model['LLM'] not in notable_models:
                continue
            
            # label_text = f"{model['LLM']}\n({model['Context Size']:.0f}K)"
            
            # texts.append(
            #     ax.text(
            #         model['Release Year'],
            #         model['Context Size'],
            #         label_text,
            #         fontsize=9,
            #         fontweight='bold',
            #         color='black',
            #         ha='center',
            #         va='bottom'
            #     )
            # )

            # Create text with background box for better readability
            text = plt.annotate(
                model['LLM'], 
                xy=(model['Release Year'], model['Context Size']),
                xytext=(model['Release Year'], model['Context Size'] * 1.3),
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


        
        # --- 6. Customize Axes and Title ---
        ax.set_xlabel('Release Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Context Length', fontsize=14, fontweight='bold')
        
        # Set y-axis to log scale for better visualization
        ax.set_yscale('log')
        
        # Set x-axis ticks
        years = sorted(df['Release Year'].unique())
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        
        # Add grid
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # --- 7. Save the Output ---
        output_filename = 'model_context_length.pdf'
        plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
        
        print(f"\nChart saved as '{output_filename}'")
        
        # Print some statistics
        print(f"\nStatistics:")
        print(f"Total models analyzed: {len(df)}")
        print(f"Year range: {df['Release Year'].min()} - {df['Release Year'].max()}")
        print(f"Context length range: {df['Context Size'].min():,.0f} - {df['Context Size'].max():,.0f} tokens")
        print(f"Average annual growth rate: {np.exp(slope):.2f}x per year")
        
    except FileNotFoundError:
        print("Error: 'model_context_length.csv' not found. Please ensure it's in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_context_length_chart() 