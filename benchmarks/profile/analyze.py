import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def read_csv_file(filename):
    """Read and parse a CSV file from the same directory"""
    file_path = os.path.join(os.path.dirname(__file__), filename)
    
    with open(file_path, 'r') as csvfile:
        csv_dict_reader = csv.DictReader(csvfile)  # Changed to DictReader for named access
        data = [row for row in csv_dict_reader]
    
    return data

def filter_outliers(data, value_index):
    """Remove outliers using IQR method on specified value index"""
    values = [item[value_index] for item in data]
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return [d for d in data if lower_bound <= d[value_index] <= upper_bound]


def __main__():
    data = read_csv_file("cospec_profiler_results.csv")
    
    # Prepare storage for processed data
    processed_sets = []
    
    # Process data in groups of 3 consecutive entries (draft, target, step)
    for i in range(0, len(data)-2, 3):
        try:
            draft = data[i]
            target = data[i+1]
            step = data[i+2]
            
            # Verify we have a complete set
            if (draft['name'] == 'draft' and 
                target['name'] == 'target' and 
                step['name'] == 'step'):

                assert draft['running_queue_size'] == target['running_queue_size'] == step['running_queue_size']
                assert draft['num_lookahead_slots'] == target['num_lookahead_slots'] == step['num_lookahead_slots']
                assert draft['total_seq_len'] == target['total_seq_len'] == step['total_seq_len']
                
                draft['running_queue_size'] = int(draft['running_queue_size'])
                draft['num_lookahead_slots'] = int(draft['num_lookahead_slots'])
                draft['total_seq_len'] = int(draft['total_seq_len'])
                draft['duration'] = float(draft['duration'])

                target['running_queue_size'] = int(target['running_queue_size'])
                target['num_lookahead_slots'] = int(target['num_lookahead_slots'])
                target['total_seq_len'] = int(target['total_seq_len'])
                target['duration'] = float(target['duration'])

                step['running_queue_size'] = int(step['running_queue_size'])
                step['num_lookahead_slots'] = int(step['num_lookahead_slots'])
                step['total_seq_len'] = int(step['total_seq_len'])
                step['duration'] = float(step['duration'])

                processed_sets.append({
                    'draft': draft,
                    'target': target,
                    'step': step,
                })
        except (KeyError, IndexError):
            continue
    
    # Prepare plot data using consecutive sets
    draft_data = []
    target_data = []
    step_data = []
    
    for s in processed_sets:
        # Draft model analysis
        draft_data.append((
            s['draft']['running_queue_size'],
            s['draft']['total_seq_len'],
            s['draft']['duration'] / s['draft']['num_lookahead_slots']
        ))
        
        # Target model analysis
        target_data.append((
            s['target']['running_queue_size'] * (s['target']['num_lookahead_slots'] + 1),
            s['target']['total_seq_len'],
            s['target']['duration']
        ))
        
        # Pre/post processing analysis
        step_data.append((
            s['step']['running_queue_size'],
            s['step']['num_lookahead_slots'],  # Added lookahead slots
            s['step']['duration'] - (s['draft']['duration'] + s['target']['duration'])
        ))

    draft_data = filter_outliers(draft_data, 2)  # Filter on duration/slot
    target_data = filter_outliers(target_data, 2)  # Filter on duration
    step_data = filter_outliers(step_data, 2)    # Filter on overhead

    # Train regression models
    def train_model(features, target):
        """Efficient linear regression training with numpy"""
        X = np.array(features)
        y = np.array(target)
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept term
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        return coeffs[0], coeffs[1:]  # intercept, coefficients

    # Draft model: Predict duration/slot from queue_size and seq_len
    X_draft = [[d[0], d[1]] for d in draft_data]
    y_draft = [d[2] for d in draft_data]
    draft_intercept, draft_coeffs = train_model(X_draft, y_draft)
    
    # Target model: Predict duration from adj_queue_size and seq_len
    X_target = [[t[0], t[1]] for t in target_data]
    y_target = [t[2] for t in target_data]
    target_intercept, target_coeffs = train_model(X_target, y_target)
    
    # Pre/post model: Predict overhead from queue_size and lookahead
    X_prepost = [[s[0], s[1]] for s in step_data]
    y_prepost = [s[2] for s in step_data]
    prepost_intercept, prepost_coeffs = train_model(X_prepost, y_prepost)

    # Print model equations
    print("\nRegression Models:")
    print(f"Draft: y = {draft_intercept:.2e} + {draft_coeffs[0]:.2e}*Q + {draft_coeffs[1]:.2e}*L")
    print(f"Target: y = {target_intercept:.2e} + {target_coeffs[0]:.2e}*Q_adj + {target_coeffs[1]:.2e}*L")
    print(f"PrePost: y = {prepost_intercept:.2e} + {prepost_coeffs[0]:.2e}*Q + {prepost_coeffs[1]:.2e}*N")

    # Calculate R² scores
    def calculate_r2(features, target, intercept, coeffs):
        pred = intercept + np.dot(features, coeffs)
        return r2_score(target, pred)

    print("\nR² Scores:")
    print(f"Draft: {calculate_r2(X_draft, y_draft, draft_intercept, draft_coeffs):.3f}")
    print(f"Target: {calculate_r2(X_target, y_target, target_intercept, target_coeffs):.3f}")
    print(f"PrePost: {calculate_r2(X_prepost, y_prepost, prepost_intercept, prepost_coeffs):.3f}")

    # Update subplot specs for three 3D plots
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            'Draft Model Latency',
            'Target Model Latency',
            'Pre/Post Processing Overhead'
        )
    )

    # Common marker settings
    marker_size = 3  # Reduced from 5
    marker_opacity = 0.8

    # Draft Model 3D Plot
    fig.add_trace(
        go.Scatter3d(
            x=[d[0] for d in draft_data],
            y=[d[1] for d in draft_data],
            z=[d[2] for d in draft_data],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=[d[2] for d in draft_data],
                colorscale='Viridis',
                opacity=marker_opacity
            ),
            name='Draft'
        ),
        row=1, col=1
    )

    # Target Model 3D Plot
    fig.add_trace(
        go.Scatter3d(
            x=[t[0] for t in target_data],
            y=[t[1] for t in target_data],
            z=[t[2] for t in target_data],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=[t[2] for t in target_data],
                colorscale='Plasma',
                opacity=marker_opacity
            ),
            name='Target'
        ),
        row=1, col=2
    )

    # Pre/Post Processing 3D Plot
    fig.add_trace(
        go.Scatter3d(
            x=[s[0] for s in step_data],  # Queue Size
            y=[s[1] for s in step_data],  # Lookahead Slots
            z=[s[2] for s in step_data],  # Overhead
            mode='markers',
            marker=dict(
                size=marker_size,
                color=[s[0] for s in step_data],  # Color by Queue Size
                colorscale='Ice',
                opacity=marker_opacity,
                colorbar=dict(title='Queue Size')
            ),
            name='Overhead'
        ),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        height=600,
        width=1600,
        scene1=dict(
            xaxis_title='Queue Size',
            yaxis_title='Sequence Length',
            zaxis_title='Duration (s)'
        ),
        scene2=dict(
            xaxis_title='Adjusted Queue Size',
            yaxis_title='Sequence Length',
            zaxis_title='Duration (s)'
        ),
        scene3=dict(
            xaxis_title='Queue Size',
            yaxis_title='Lookahead Slots',
            zaxis_title='Overhead (s)'
        ),
        showlegend=False
    )
    
    # Update 2D plot axis labels
    fig.update_xaxes(title_text="Running Queue Size", row=1, col=3)
    fig.update_yaxes(title_text="Overhead Duration (s)", row=1, col=3)

    # Save and show
    fig.write_html("latency_analysis_interactive.html")
    print("Saved interactive plot to latency_analysis_interactive.html")

if __name__ == "__main__":
    __main__()
