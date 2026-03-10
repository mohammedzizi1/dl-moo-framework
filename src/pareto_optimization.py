"""
Multi-objective Pareto optimization engine.

Implements the four-objective optimization (Eqs. 7-14 in the paper):
1. Total inventory cost minimization
2. Service level maximization
3. Delivery efficiency maximization
4. Production smoothness maximization

Produces the Pareto frontier of non-dominated solutions.
"""

import numpy as np
from scipy import stats


def calculate_baseline_metrics(test_df, naive_pred):
    """
    Calculate baseline performance under naive forecasting.
    
    These serve as normalized reference points for delivery efficiency
    and production smoothness scores (Eqs. de_score, ps_score).
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with store/item/date columns.
    naive_pred : np.ndarray
        Naive forecast predictions.
    
    Returns
    -------
    dict, float, float
        Baseline metrics, delivery std dev, production std dev.
    """
    sigma_e = np.std(test_df['sales'].values - naive_pred)
    z_alpha = stats.norm.ppf(0.95)
    safety_stock = z_alpha * sigma_e * np.sqrt(3)
    order_qty = naive_pred + safety_stock / 3

    # Delivery std dev: variability across stores (Eq. f3)
    test_df_temp = test_df.copy()
    test_df_temp['order_qty'] = order_qty
    store_orders = test_df_temp.groupby('store')['order_qty'].sum()
    delivery_std = store_orders.std()

    # Production std dev: variability across days (Eq. f4)
    daily_production = test_df_temp.groupby('date')['order_qty'].sum()
    production_std = daily_production.std()

    y_true = test_df['sales'].values
    fulfilled = np.minimum(order_qty, y_true)
    service_level = np.sum(fulfilled) / np.sum(y_true)

    excess = np.maximum(order_qty - y_true, 0)
    shortage = np.maximum(y_true - order_qty, 0)
    total_cost = 0.5 * np.sum(excess) + 10.0 * np.sum(shortage)

    baseline = {
        'delivery_std': delivery_std,
        'production_std': production_std,
        'service_level': service_level * 100,
        'total_cost': total_cost
    }

    return baseline, delivery_std, production_std


def calculate_supply_chain_objectives(y_true, y_pred, test_df, weights,
                                      delivery_baseline_std, production_baseline_std):
    """
    Evaluate all four supply chain objectives for a given weight configuration.
    
    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Actual and predicted demand.
    test_df : pd.DataFrame
        Test data with store/item/date columns.
    weights : dict
        Decision variable weights: service, holding, stockout, production.
    delivery_baseline_std, production_baseline_std : float
        Baseline standard deviations for normalization.
    
    Returns
    -------
    dict
        Objective values: service_level, total_cost, delivery_efficiency,
        production_smoothness, composite_score.
    """
    errors = y_true - y_pred
    sigma_e = np.std(errors)
    z_alpha = stats.norm.ppf(min(weights['service'], 0.999))
    safety_stock = z_alpha * sigma_e * np.sqrt(3)
    order_qty = y_pred + safety_stock / 3

    # Objective 1: Total cost (Eq. f1)
    excess = np.maximum(order_qty - y_true, 0)
    shortage = np.maximum(y_true - order_qty, 0)
    holding_cost = weights['holding'] * 0.5 * np.sum(excess)
    stockout_cost = weights['stockout'] * 10.0 * np.sum(shortage)
    total_cost = holding_cost + stockout_cost

    # Objective 2: Service level (Eq. f2)
    fulfilled = np.minimum(order_qty, y_true)
    service_level = np.sum(fulfilled) / np.sum(y_true) * 100

    # Objective 3: Delivery efficiency (Eqs. f3, de_score)
    test_df_temp = test_df.copy()
    test_df_temp['order_qty'] = order_qty
    store_orders = test_df_temp.groupby('store')['order_qty'].sum()
    delivery_std = store_orders.std()
    delivery_eff = max(0, (delivery_baseline_std - delivery_std) / delivery_baseline_std * 100)

    # Objective 4: Production smoothness (Eqs. f4, ps_score)
    daily_prod = test_df_temp.groupby('date')['order_qty'].sum()
    prod_std = daily_prod.std()
    prod_smooth = max(0, (production_baseline_std - prod_std) / production_baseline_std * 100)

    # Composite score (Eq. composite)
    composite = (service_level / 100 + (1 - total_cost / 1e7) +
                 delivery_eff / 100 + prod_smooth / 100) / 4 * 100

    return {
        'service_level': round(service_level, 2),
        'total_cost': round(total_cost, 2),
        'delivery_efficiency': round(delivery_eff, 2),
        'production_smoothness': round(prod_smooth, 2),
        'composite_score': round(composite, 2)
    }


def is_pareto_optimal(solutions):
    """
    Identify Pareto-optimal solutions via non-dominated sorting.
    
    A solution is Pareto-optimal if no other solution improves all objectives.
    
    Parameters
    ----------
    solutions : list of dict
        Each dict contains objective values.
    
    Returns
    -------
    list of bool
        True for Pareto-optimal solutions.
    """
    n = len(solutions)
    is_optimal = [True] * n

    for i in range(n):
        if not is_optimal[i]:
            continue
        for j in range(n):
            if i == j or not is_optimal[j]:
                continue

            si, sj = solutions[i], solutions[j]

            # Check if j dominates i (higher service, lower cost, higher eff/smooth)
            j_better = (
                sj['service_level'] >= si['service_level'] and
                sj['total_cost'] <= si['total_cost'] and
                sj['delivery_efficiency'] >= si['delivery_efficiency'] and
                sj['production_smoothness'] >= si['production_smoothness']
            )
            j_strictly_better = (
                sj['service_level'] > si['service_level'] or
                sj['total_cost'] < si['total_cost'] or
                sj['delivery_efficiency'] > si['delivery_efficiency'] or
                sj['production_smoothness'] > si['production_smoothness']
            )

            if j_better and j_strictly_better:
                is_optimal[i] = False
                break

    return is_optimal


def multi_objective_optimization(y_true, y_pred, test_df, baselines):
    """
    Run the full multi-objective Pareto optimization pipeline.
    
    Evaluates 15 weight configurations, applies Pareto dominance filtering,
    and returns the non-dominated solution set.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual demand values.
    y_pred : np.ndarray
        Ensemble forecast predictions.
    test_df : pd.DataFrame
        Test data with store/item/date columns.
    baselines : dict
        Baseline model predictions.
    
    Returns
    -------
    pareto_solutions : list of dict
        Pareto-optimal solution set.
    baseline_scores : dict
        Baseline performance metrics.
    delivery_baseline_std : float
        Baseline delivery standard deviation.
    production_baseline_std : float
        Baseline production standard deviation.
    """
    print("\n  Calculating baselines for multi-objective metrics...")

    naive_pred = baselines.get('Naive (t-1)', y_true * 0.9)
    baseline_scores, delivery_baseline_std, production_baseline_std = \
        calculate_baseline_metrics(test_df, naive_pred)

    # 15 weight configurations spanning different strategic priorities
    weight_configs = [
        {'name': 'A', 'label': 'Balanced',          'service': 0.90, 'holding': 1.0, 'stockout': 1.0, 'production': 1.0},
        {'name': 'B', 'label': 'Cost-Service',       'service': 0.92, 'holding': 1.2, 'stockout': 0.8, 'production': 0.8},
        {'name': 'C', 'label': 'High-Service',        'service': 0.95, 'holding': 0.8, 'stockout': 1.5, 'production': 0.7},
        {'name': 'D', 'label': 'Delivery-Focused',     'service': 0.88, 'holding': 1.0, 'stockout': 0.9, 'production': 1.3},
        {'name': 'E', 'label': 'Production-Focused',   'service': 0.87, 'holding': 0.9, 'stockout': 0.8, 'production': 1.5},
        {'name': 'F', 'label': 'Cost-Min',             'service': 0.85, 'holding': 1.5, 'stockout': 0.5, 'production': 0.5},
        {'name': 'G', 'label': 'Service-Max',          'service': 0.97, 'holding': 0.5, 'stockout': 2.0, 'production': 0.5},
        {'name': 'H', 'label': 'Balanced-High',        'service': 0.93, 'holding': 1.1, 'stockout': 1.1, 'production': 1.1},
        {'name': 'I', 'label': 'Delivery-Service',     'service': 0.91, 'holding': 0.9, 'stockout': 1.2, 'production': 1.2},
        {'name': 'J', 'label': 'Low-Cost-Service',     'service': 0.92, 'holding': 1.3, 'stockout': 0.7, 'production': 0.6},
        {'name': 'K', 'label': 'Prod-Delivery',        'service': 0.89, 'holding': 0.8, 'stockout': 0.8, 'production': 1.4},
        {'name': 'L', 'label': 'Max-Efficiency',       'service': 0.86, 'holding': 1.0, 'stockout': 0.7, 'production': 1.6},
        {'name': 'M', 'label': 'Conservative',         'service': 0.94, 'holding': 1.0, 'stockout': 1.3, 'production': 0.9},
        {'name': 'N', 'label': 'Aggressive-Service',   'service': 0.96, 'holding': 0.6, 'stockout': 1.8, 'production': 0.6},
        {'name': 'O', 'label': 'Lean-Operations',      'service': 0.88, 'holding': 1.4, 'stockout': 0.6, 'production': 1.2},
    ]

    all_solutions = []
    for config in weight_configs:
        result = calculate_supply_chain_objectives(
            y_true, y_pred, test_df, config,
            delivery_baseline_std, production_baseline_std
        )
        result['name'] = f"Solution {config['name']}"
        result['label'] = config['label']
        all_solutions.append(result)

    # Pareto dominance filtering
    is_optimal = is_pareto_optimal(all_solutions)
    pareto_solutions = [s for s, opt in zip(all_solutions, is_optimal) if opt]

    print(f"  Found {len(pareto_solutions)} Pareto-optimal solutions from {len(all_solutions)} candidates")
    for sol in pareto_solutions:
        print(f"    {sol['name']:15s} | SL={sol['service_level']:.1f}% | "
              f"Cost=${sol['total_cost']:,.0f} | DE={sol['delivery_efficiency']:.1f}% | "
              f"PS={sol['production_smoothness']:.1f}%")

    return pareto_solutions, baseline_scores, delivery_baseline_std, production_baseline_std
