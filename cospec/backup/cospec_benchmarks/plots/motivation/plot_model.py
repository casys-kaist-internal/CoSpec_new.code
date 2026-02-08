import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression
from adjustText import adjust_text

# Read the CSV file
df = pd.read_csv('notable_ai_models.csv')

# Clean and prepare the data
# Convert publication date to datetime
df['Publication date'] = pd.to_datetime(df['Publication date'], errors='coerce')

# from 2017 to 2025
df = df[df['Publication date'] >= pd.Timestamp('2017-01-01')]

# Clean parameters column - remove non-numeric values and convert to float
df['Parameters'] = pd.to_numeric(df['Parameters'], errors='coerce')

# Remove rows with missing data
df_clean = df.dropna(subset=['Publication date', 'Parameters', 'Model'])

# Create the scatter plot
plt.figure(figsize=(8, 4))  # Increased figure size for better text placement

# Create scatter plot
scatter = plt.scatter(df_clean['Publication date'], df_clean['Parameters'], 
                     c='#1f77b4', marker='o', s=10)

# Add exponential trend line
if not df_clean.empty:
    # Convert dates to years for regression
    X = df_clean['Publication date'].dt.year.values.reshape(-1, 1)
    y = np.log(df_clean['Parameters'])
    
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    trend_years = np.array([df_clean['Publication date'].dt.year.min(), 
                           df_clean['Publication date'].dt.year.max()])
    trend_params = np.exp(intercept + slope * trend_years)

    plt.plot(
        pd.to_datetime(trend_years, format='%Y'),
        trend_params,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Exponential Trend'
    )
    
    annual_growth = np.exp(slope)
    print(f"Model parameters annual growth: {annual_growth:.2f}x/year")

# Annotate notable models with automatic positioning
notable_models = ['Llama 4 Behemoth (preview)', 'DeepSeek-V3', 'Llama 3.2 11B', 'Llama 3.1-405B', 'Llama 3-70B', 'Llama 2-7B',  'GPT-3.5 Turbo', 'OPT-175B',  'InstructGPT 6B', 'GPT-2 (1.5B)', 'GPT-1']

texts = []

# Get plot limits for boundary checking
x_min, x_max = df_clean['Publication date'].min(), df_clean['Publication date'].max()
y_min, y_max = df_clean['Parameters'].min(), df_clean['Parameters'].max()

# Calculate offsets for text positioning
x_range = (x_max - x_min).total_seconds() * 1e9  # Convert to nanoseconds for calculation
y_range = y_max - y_min
x_offset = x_range * 0.02  # 2% of x range
y_offset = y_range * 0.1   # 10% of y range

for idx, row in df_clean.iterrows():
    if row['Model'] in notable_models:
        x_pos = row['Publication date']
        y_pos = row['Parameters']

        # change Llama 4 Behemoth (preview) to Llama 4 Behemoth
        if row['Model'] == 'Llama 4 Behemoth (preview)':
            row['Model'] = 'Llama 4 Behemoth'

        # change GPT-2 (1.5B) to GPT-2
        if row['Model'] == 'GPT-2 (1.5B)':
            row['Model'] = 'GPT-2'
        
        # Calculate automatic position with boundary checking
        text_x = x_pos + pd.Timedelta(days=30)  # Offset by 30 days
        text_y = y_pos * 1.2  # Offset upward by 20%
        
        # Ensure text stays within plot boundaries
        if text_x > x_max:
            text_x = x_pos - pd.Timedelta(days=30)  # Move left if too far right
        
        if text_y > y_max:
            text_y = y_pos * 0.8  # Move down if too high
        
        # Create text with background box for better readability
        text = plt.annotate(
            row['Model'], 
            xy=(x_pos, y_pos),
            xytext=(text_x, text_y),
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

# Use adjust_text to fine-tune positions while respecting boundaries
# if texts:
#     adjust_text(
#         texts,
#         arrowprops=dict(
#             arrowstyle='->',
#             color='gray',
#             lw=0.6,
#             alpha=0.6,
#             connectionstyle="arc3,rad=0.1"
#         ),
#         expand_points=(1.1, 1.1),  # Minimal expansion
#         force_points=(0.2, 0.2),   # Gentle force
#         force_text=(0.2, 0.2),     # Gentle force between text
#         min_arrow_len=5,           # Short arrows
#         max_iter=30,               # Few iterations
#         add_objects=[scatter],
#         # Constrain movement to prevent overflow
#         only_move={'points':'xy', 'text':'xy', 'objects':'xy'}
#     )

# Customize the plot
plt.xlabel('Publication Year', fontsize=12, fontweight='bold')
plt.ylabel('Model Parameters', fontsize=12, fontweight='bold')

# Use log scale for y-axis since parameters vary by orders of magnitude
plt.yscale('log')
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot in pdf format
plt.savefig('ai_models_parameters_vs_date.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()
