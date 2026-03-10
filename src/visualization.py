"""
Publication-quality figure generation for the paper.

Generates all 16+ figures at 300 DPI suitable for IEEE Access submission.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def setup_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def generate_all_figures(results, all_preds, y_test, inv_results,
                          pareto_solutions, output_dir='figures'):
    """
    Generate all publication-quality figures.
    
    Parameters
    ----------
    results : dict
        Evaluation results from evaluate_with_stats().
    all_preds : dict
        All model predictions.
    y_test : np.ndarray
        True test values.
    inv_results : dict
        Inventory optimization results.
    pareto_solutions : list
        Pareto-optimal solutions.
    output_dir : str
        Directory to save figures.
    """
    setup_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure: Model Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(results.keys())
    r2_vals = [results[m]['R2'] for m in models]
    colors = ['#A5A5A5' if r2 < 0.85 else '#5B9BD5' if r2 < 0.92 else '#70AD47' for r2 in r2_vals]
    ax.barh(models, r2_vals, color=colors, edgecolor='white')
    ax.set_xlabel('$R^2$ Score')
    ax.set_title('Forecasting Model Performance Comparison')
    for i, (m, v) in enumerate(zip(models, r2_vals)):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300)
    plt.close()
    
    # Figure: Pareto Frontier (Parallel Coordinates)
    if pareto_solutions:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = ['Service\nLevel', 'Cost\n(inverted)', 'Delivery\nEff.', 'Production\nSmooth.', 'Composite']
        colors_s = ['#4472C4', '#ED7D31', '#70AD47', '#7030A0']
        
        data = np.array([[s['service_level'], s['total_cost'], s['delivery_efficiency'],
                          s['production_smoothness'], s['composite_score']] for s in pareto_solutions])
        mins, maxs = data.min(0), data.max(0)
        norm = (data - mins) / (maxs - mins + 1e-10)
        norm[:, 1] = 1 - norm[:, 1]  # Invert cost
        
        x = np.arange(len(labels))
        for i, sol in enumerate(pareto_solutions):
            c = colors_s[i % len(colors_s)]
            ax.plot(x, norm[i], 'o-', color=c, linewidth=2.5, markersize=8,
                   label=sol['name'], alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Normalized Score')
        ax.set_title('Pareto-Optimal Solutions (Parallel Coordinates)')
        ax.legend(loc='lower left')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_pcp.png', dpi=300)
        plt.close()

    print(f"  Figures saved to {output_dir}/")
