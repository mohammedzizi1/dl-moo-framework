"""
Single-objective inventory optimization using the newsvendor model.

Implements safety stock calculation (Eq. 24), order quantity determination
(Eq. 25), and asymmetric cost evaluation (Eq. 26) from the paper.
"""

import numpy as np
from scipy import stats


def optimize_inventory(y_true, y_pred, target_service=0.95, h=0.5, p=10.0, lead_time=3):
    """
    Apply newsvendor-based inventory optimization.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual demand values.
    y_pred : np.ndarray
        Forecasted demand values.
    target_service : float
        Target service level (e.g., 0.95 for 95%).
    h : float
        Holding cost per unit per day.
    p : float
        Stockout penalty per unit.
    lead_time : int
        Replenishment lead time in days.
    
    Returns
    -------
    dict
        Dictionary with safety_stock, holding_cost, stockout_cost,
        total_cost, and service_level.
    """
    errors = y_true - y_pred
    sigma_e = np.std(errors)
    
    # Safety stock (Eq. 24): SS = z_alpha * sigma_e * sqrt(LT)
    z_alpha = stats.norm.ppf(target_service)
    safety_stock = z_alpha * sigma_e * np.sqrt(lead_time)
    
    # Order quantity (Eq. 25): Q_t = y_hat_t + SS/LT
    order_qty = y_pred + safety_stock / lead_time
    
    # Cost structure (Eq. 26)
    excess = np.maximum(order_qty - y_true, 0)
    shortage = np.maximum(y_true - order_qty, 0)
    holding_cost = h * np.sum(excess)
    stockout_cost = p * np.sum(shortage)
    total_cost = holding_cost + stockout_cost
    
    # Service level (Eq. 27): SL = sum(min(Q,y)) / sum(y)
    fulfilled = np.minimum(order_qty, y_true)
    service_level = np.sum(fulfilled) / np.sum(y_true)

    return {
        'safety_stock': round(safety_stock, 2),
        'holding_cost': round(holding_cost, 2),
        'stockout_cost': round(stockout_cost, 2),
        'total_cost': round(total_cost, 2),
        'service_level': round(service_level * 100, 2)
    }
