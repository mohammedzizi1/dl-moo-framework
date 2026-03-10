#!/usr/bin/env python3
"""
Main execution script for the Integrated Deep Learning and Multi-Objective
Pareto Optimization Framework for Retail Supply Chains.

Usage:
    python main.py                          # Full pipeline
    python main.py --data path/to/train.csv # Custom data path

Reference:
    Zizi, M., Chafi, A., & El Hammoumi, M. (2026).
    "An Integrated Deep Learning and Multi-Objective Pareto Optimization
    Framework for Retail Supply Chains." IEEE Access.
"""

import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_pipeline import load_and_split_data, engineer_features, select_features
from src.training import train_advanced_models
from src.evaluation import evaluate_with_stats
from src.inventory_optimization import optimize_inventory
from src.pareto_optimization import multi_objective_optimization
from src.financial_analysis import calculate_realistic_roi
from src.visualization import generate_all_figures


def main(data_path='train.csv'):
    """Run the complete multi-objective supply chain optimization pipeline."""
    
    print("=" * 80)
    print("  INTEGRATED DEEP LEARNING & MULTI-OBJECTIVE PARETO OPTIMIZATION")
    print("  FOR RETAIL SUPPLY CHAINS")
    print("  Zizi, Chafi & El Hammoumi (2026) — IEEE Access")
    print("=" * 80)

    # ── Phase 1: Data Pipeline ──
    train, val, test, df = load_and_split_data(data_path)
    train, val, test = engineer_features(train, val, test)
    selected_features, all_features = select_features(train, val, test, k=40)

    # ── Phase 2: Forecasting ──
    print("\n[4/10] Training baseline models...")
    from src.data_pipeline import select_features  # reimport for baselines
    test_sorted = test.copy().sort_values(['store', 'item', 'date'])
    baselines = {}
    
    # Naive
    test_sorted['naive_pred'] = test_sorted.groupby(['store', 'item'])['sales'].shift(1)
    baselines['Naive (t-1)'] = test_sorted['naive_pred'].fillna(test_sorted['sales'].mean()).values
    
    # Moving Average
    test_sorted['ma7_pred'] = test_sorted.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    baselines['Moving Avg (7d)'] = test_sorted['ma7_pred'].fillna(test_sorted['sales'].mean()).values
    
    # Exponential Smoothing
    baselines['Exp Smoothing'] = test_sorted.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).ewm(span=14, min_periods=1).mean()
    ).fillna(test_sorted['sales'].mean()).values
    
    print("  ✓ Naive, Moving Average, Exponential Smoothing trained")

    # Deep learning models + ensemble
    all_preds, y_test = train_advanced_models(train, val, test, selected_features, baselines)

    # ── Evaluation ──
    results = evaluate_with_stats(y_test, all_preds)

    # ── Phase 3: Optimization ──
    print("\n[8/10] Single-objective inventory optimization...")
    inv_results = {}
    for name, pred in all_preds.items():
        inv_results[name] = optimize_inventory(y_test, pred)
    
    # Print inventory results
    print(f"\n  {'Model':25s} | {'Total Cost':>12s} | {'Service Level':>13s}")
    print("  " + "-" * 60)
    for name, res in inv_results.items():
        print(f"  {name:25s} | ${res['total_cost']:>11,.0f} | {res['service_level']:>12.2f}%")

    # Multi-objective Pareto optimization
    print("\n[9/10] Multi-objective Pareto optimization...")
    best_pred = all_preds['Weighted Ensemble']
    pareto_solutions, baseline_scores, del_std, prod_std = \
        multi_objective_optimization(y_test, best_pred, test, baselines)

    # Financial analysis
    naive_cost = inv_results.get('Naive (t-1)', {}).get('total_cost', 0)
    best_cost = min(r['total_cost'] for r in inv_results.values())
    savings = naive_cost - best_cost
    
    financials = calculate_realistic_roi(savings, test)
    
    print(f"\n  Financial Summary:")
    print(f"    Conservative Annual Savings: ${financials['conservative_annual']:,.0f}")
    print(f"    ROI: {financials['roi_percent']:.1f}%")
    print(f"    Payback Period: {financials['payback_months']:.1f} months")

    # ── Generate figures ──
    print("\n[10/10] Generating publication-quality figures...")
    generate_all_figures(results, all_preds, y_test, inv_results, pareto_solutions)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("  ✓ PIPELINE COMPLETE!")
    print("=" * 80)
    
    best_r2 = max(r['R2'] for r in results.values())
    best_model = [k for k, v in results.items() if v['R2'] == best_r2][0]
    
    print(f"\n  Best Model: {best_model} (R² = {best_r2:.4f})")
    print(f"  Pareto Solutions: {len(pareto_solutions)} non-dominated")
    print(f"  Annual Savings: ${financials['conservative_annual']:,.0f}")
    print(f"  ROI: {financials['roi_percent']:.1f}% | Payback: {financials['payback_months']:.1f} months")
    print("=" * 80)

    return {
        'results': results,
        'all_preds': all_preds,
        'y_test': y_test,
        'inv_results': inv_results,
        'pareto_solutions': pareto_solutions,
        'financials': financials,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Integrated DL + MOO Supply Chain Optimization Framework'
    )
    parser.add_argument('--data', type=str, default='train.csv',
                        help='Path to dataset CSV (default: train.csv)')
    args = parser.parse_args()
    
    output = main(data_path=args.data)
