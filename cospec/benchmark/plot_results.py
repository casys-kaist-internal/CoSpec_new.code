#!/usr/bin/env python3
"""Generate publication-quality plots from CoSpec evaluation results.

Creates plots following OSDI/SOSP standards:
- Figure 1: Latency-Throughput curves (main result)
- Figure 2: Model generality (bar chart)
- Figure 3: SM ratio ablation
- Figure 4: Gamma sensitivity

Usage:
    python plot_results.py --results-dir results/evaluation_20240101_120000
    python plot_results.py --results-csv results.csv --output-dir plots/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color scheme (colorblind-friendly)
COLORS = {
    'ar': '#808080',  # Gray
    'vanilla_sd': '#1f77b4',  # Blue
    'cospec': '#d62728',  # Red
}

MARKERS = {
    'ar': 'o',
    'vanilla_sd': 's',
    'cospec': '^',
}

LINESTYLES = {
    'ar': '--',
    'vanilla_sd': ':',
    'cospec': '-',
}

LABELS = {
    'ar': 'AR Baseline',
    'vanilla_sd': 'Vanilla SD',
    'cospec': 'CoSpec',
}


def load_results(results_dir: Optional[str] = None,
                 results_csv: Optional[str] = None) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    if results_csv:
        csv_path = Path(results_csv)
    elif results_dir:
        csv_path = Path(results_dir) / 'results.csv'
    else:
        raise ValueError("Must specify either --results-dir or --results-csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across repeats (mean and std)."""
    # Group by config, gamma, sm_ratio, request_rate
    group_cols = ['config', 'gamma', 'sm_ratio', 'request_rate']
    agg_cols = [col for col in df.columns if col not in group_cols + ['repeat']]

    agg_funcs = {col: ['mean', 'std'] for col in agg_cols}
    agg_df = df.groupby(group_cols).agg(agg_funcs)

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()

    return agg_df


def plot_latency_throughput(df: pd.DataFrame, output_dir: Path,
                            metric: str = 'p99_ttft_ms',
                            ylabel: str = 'P99 TTFT (ms)'):
    """Plot latency vs throughput curves (Figure 1 - Main Result).

    Following OSDI/SOSP style:
    - X-axis: Request throughput (req/s)
    - Y-axis: Latency metric (P99 TTFT or E2E)
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    agg_df = aggregate_results(df)

    # Filter to main configs (experiment 1)
    main_configs = ['ar', 'vanilla_sd', 'cospec']

    for config in main_configs:
        config_df = agg_df[
            (agg_df['config'] == config) |
            (agg_df['config'].str.startswith(config) & ~agg_df['config'].str.contains('ablation'))
        ]
        if config_df.empty:
            continue

        # Sort by request rate
        config_df = config_df.sort_values('request_rate')

        x = config_df['request_rate'].values
        y = config_df[f'{metric}_mean'].values
        yerr = config_df[f'{metric}_std'].values if f'{metric}_std' in config_df.columns else None

        ax.errorbar(
            x, y, yerr=yerr,
            label=LABELS.get(config, config),
            color=COLORS.get(config, 'black'),
            marker=MARKERS.get(config, 'o'),
            linestyle=LINESTYLES.get(config, '-'),
            linewidth=1.5,
            markersize=6,
            capsize=3,
        )

    ax.set_xlabel('Request Throughput (req/s)')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add annotation showing improvement
    # (Will be customized based on actual data)

    fig.tight_layout()
    output_path = output_dir / f'fig1_latency_throughput_{metric}.pdf'
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_latency_throughput_dual(df: pd.DataFrame, output_dir: Path):
    """Plot both TTFT and E2E latency in 1x2 subplot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    agg_df = aggregate_results(df)
    main_configs = ['ar', 'vanilla_sd', 'cospec']

    metrics = [
        ('p99_ttft_ms', 'P99 TTFT (ms)'),
        ('mean_e2e_ms', 'Mean E2E Latency (ms)'),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for config in main_configs:
            config_df = agg_df[
                (agg_df['config'] == config) |
                (agg_df['config'].str.startswith(config) & ~agg_df['config'].str.contains('ablation'))
            ]
            if config_df.empty:
                continue

            config_df = config_df.sort_values('request_rate')
            x = config_df['request_rate'].values

            y_col = f'{metric}_mean'
            if y_col not in config_df.columns:
                # Try without _mean suffix
                y_col = metric
            if y_col not in config_df.columns:
                continue

            y = config_df[y_col].values
            yerr_col = f'{metric}_std'
            yerr = config_df[yerr_col].values if yerr_col in config_df.columns else None

            ax.errorbar(
                x, y, yerr=yerr,
                label=LABELS.get(config, config),
                color=COLORS.get(config, 'black'),
                marker=MARKERS.get(config, 'o'),
                linestyle=LINESTYLES.get(config, '-'),
                linewidth=1.5,
                markersize=6,
                capsize=3,
            )

        ax.set_xlabel('Request Throughput (req/s)')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    output_path = output_dir / 'fig1_latency_throughput_dual.pdf'
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_sm_ratio_ablation(df: pd.DataFrame, output_dir: Path):
    """Plot SM ratio ablation (Figure 3)."""
    fig, ax1 = plt.subplots(figsize=(5, 3.5))

    # Filter to ablation data
    ablation_df = df[df['config'].str.contains('ablation')]
    if ablation_df.empty:
        print("No ablation data found for SM ratio plot")
        return

    agg_df = aggregate_results(ablation_df)
    cospec_df = agg_df[agg_df['config'] == 'cospec_ablation'].sort_values('sm_ratio')

    if cospec_df.empty:
        print("No CoSpec ablation data found")
        return

    x = cospec_df['sm_ratio'].values

    # Throughput on left y-axis
    y1 = cospec_df['request_throughput_mean'].values
    yerr1 = cospec_df['request_throughput_std'].values if 'request_throughput_std' in cospec_df.columns else None

    line1 = ax1.errorbar(
        x, y1, yerr=yerr1,
        color=COLORS['cospec'],
        marker=MARKERS['cospec'],
        linestyle='-',
        linewidth=1.5,
        markersize=8,
        capsize=3,
        label='Throughput (req/s)',
    )
    ax1.set_xlabel('Target SM Ratio')
    ax1.set_ylabel('Request Throughput (req/s)', color=COLORS['cospec'])
    ax1.tick_params(axis='y', labelcolor=COLORS['cospec'])

    # P99 TTFT on right y-axis
    ax2 = ax1.twinx()

    metric = 'p99_ttft_ms' if 'p99_ttft_ms_mean' in cospec_df.columns else 'mean_ttft_ms'
    y2 = cospec_df[f'{metric}_mean'].values if f'{metric}_mean' in cospec_df.columns else cospec_df[metric].values
    yerr2 = cospec_df[f'{metric}_std'].values if f'{metric}_std' in cospec_df.columns else None

    line2 = ax2.errorbar(
        x, y2, yerr=yerr2,
        color=COLORS['vanilla_sd'],
        marker=MARKERS['vanilla_sd'],
        linestyle='--',
        linewidth=1.5,
        markersize=8,
        capsize=3,
        label='P99 TTFT (ms)',
    )
    ax2.set_ylabel('P99 TTFT (ms)', color=COLORS['vanilla_sd'])
    ax2.tick_params(axis='y', labelcolor=COLORS['vanilla_sd'])

    # Mark optimal ratio
    if len(y1) > 0:
        best_idx = np.argmax(y1)
        ax1.axvline(x=x[best_idx], color='green', linestyle=':', alpha=0.7, label=f'Optimal ({x[best_idx]})')

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in [line1, line2]]
    ax1.legend(lines, labels, loc='upper right')

    fig.tight_layout()
    output_path = output_dir / 'fig3_sm_ratio_ablation.pdf'
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_gamma_ablation(df: pd.DataFrame, output_dir: Path):
    """Plot gamma (speculation length) ablation (Figure 4)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Filter to ablation data
    ablation_df = df[df['config'].str.contains('ablation')]
    if ablation_df.empty:
        print("No ablation data found for gamma plot")
        return

    agg_df = aggregate_results(ablation_df)

    vanilla_df = agg_df[agg_df['config'] == 'vanilla_sd_ablation'].sort_values('gamma')
    cospec_df = agg_df[agg_df['config'] == 'cospec_ablation'].sort_values('gamma')

    # Get unique gamma values
    gammas = sorted(set(vanilla_df['gamma'].tolist() + cospec_df['gamma'].tolist()))
    x = np.arange(len(gammas))
    width = 0.35

    # Vanilla SD bars
    if not vanilla_df.empty:
        vanilla_vals = []
        vanilla_errs = []
        for g in gammas:
            row = vanilla_df[vanilla_df['gamma'] == g]
            if not row.empty:
                vanilla_vals.append(row['request_throughput_mean'].values[0])
                vanilla_errs.append(row['request_throughput_std'].values[0] if 'request_throughput_std' in row.columns else 0)
            else:
                vanilla_vals.append(0)
                vanilla_errs.append(0)

        ax.bar(x - width/2, vanilla_vals, width, yerr=vanilla_errs,
               label=LABELS['vanilla_sd'], color=COLORS['vanilla_sd'],
               capsize=3, alpha=0.8)

    # CoSpec bars
    if not cospec_df.empty:
        cospec_vals = []
        cospec_errs = []
        for g in gammas:
            row = cospec_df[cospec_df['gamma'] == g]
            if not row.empty:
                cospec_vals.append(row['request_throughput_mean'].values[0])
                cospec_errs.append(row['request_throughput_std'].values[0] if 'request_throughput_std' in row.columns else 0)
            else:
                cospec_vals.append(0)
                cospec_errs.append(0)

        ax.bar(x + width/2, cospec_vals, width, yerr=cospec_errs,
               label=LABELS['cospec'], color=COLORS['cospec'],
               capsize=3, alpha=0.8)

    ax.set_xlabel('Speculation Length (γ)')
    ax.set_ylabel('Request Throughput (req/s)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(g)) for g in gammas])
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    output_path = output_dir / 'fig4_gamma_ablation.pdf'
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_throughput_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot max throughput comparison bar chart."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    agg_df = aggregate_results(df)
    main_configs = ['ar', 'vanilla_sd', 'cospec']

    # Get max throughput for each config (at any latency)
    max_throughputs = []
    labels = []
    colors = []

    for config in main_configs:
        config_df = agg_df[
            (agg_df['config'] == config) |
            (agg_df['config'].str.startswith(config) & ~agg_df['config'].str.contains('ablation'))
        ]
        if config_df.empty:
            continue

        max_tp = config_df['request_throughput_mean'].max()
        max_throughputs.append(max_tp)
        labels.append(LABELS.get(config, config))
        colors.append(COLORS.get(config, 'black'))

    x = np.arange(len(labels))
    bars = ax.bar(x, max_throughputs, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, max_throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Max Throughput (req/s)')
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    output_path = output_dir / 'fig_max_throughput.pdf'
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close(fig)


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate a summary table in LaTeX format."""
    agg_df = aggregate_results(df)
    main_configs = ['ar', 'vanilla_sd', 'cospec']

    rows = []
    for config in main_configs:
        config_df = agg_df[
            (agg_df['config'] == config) |
            (agg_df['config'].str.startswith(config) & ~agg_df['config'].str.contains('ablation'))
        ]
        if config_df.empty:
            continue

        # Get metrics at moderate load (e.g., 4 req/s)
        moderate_df = config_df[config_df['request_rate'] == 4.0]
        if moderate_df.empty:
            moderate_df = config_df.iloc[[len(config_df)//2]]  # Middle rate

        row = {
            'Config': LABELS.get(config, config),
            'Throughput (req/s)': f"{config_df['request_throughput_mean'].max():.1f}",
        }

        # Add latency metrics if available
        for metric, name in [
            ('mean_ttft_ms', 'Mean TTFT (ms)'),
            ('p99_ttft_ms', 'P99 TTFT (ms)'),
            ('mean_tpot_ms', 'Mean TPOT (ms)'),
        ]:
            col = f'{metric}_mean'
            if col in moderate_df.columns:
                row[name] = f"{moderate_df[col].values[0]:.1f}"
            elif metric in moderate_df.columns:
                row[name] = f"{moderate_df[metric].values[0]:.1f}"

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / 'summary_table.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate LaTeX table
    latex_path = output_dir / 'summary_table.tex'
    with open(latex_path, 'w') as f:
        f.write(summary_df.to_latex(index=False, escape=False))
    print(f"Saved: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate CoSpec evaluation plots")
    parser.add_argument('--results-dir', type=str, default=None,
                        help="Directory containing results.csv")
    parser.add_argument('--results-csv', type=str, default=None,
                        help="Path to results CSV file")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory for plots (default: results-dir/plots)")

    args = parser.parse_args()

    # Load data
    df = load_results(args.results_dir, args.results_csv)
    print(f"Loaded {len(df)} result rows")
    print(f"Configs: {df['config'].unique()}")
    print(f"Request rates: {sorted(df['request_rate'].unique())}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.results_dir:
        output_dir = Path(args.results_dir) / 'plots'
    else:
        output_dir = Path('plots')

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate all plots
    print("\nGenerating plots...")

    plot_latency_throughput(df, output_dir, 'p99_ttft_ms', 'P99 TTFT (ms)')
    plot_latency_throughput(df, output_dir, 'mean_ttft_ms', 'Mean TTFT (ms)')
    plot_latency_throughput_dual(df, output_dir)
    plot_sm_ratio_ablation(df, output_dir)
    plot_gamma_ablation(df, output_dir)
    plot_throughput_comparison(df, output_dir)
    generate_summary_table(df, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
