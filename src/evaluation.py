"""
Forecast evaluation metrics including point and probabilistic measures.

Implements R², MAE, RMSE, MAPE, 95% CI (Section III-E of the paper),
plus CRPS and Pinball Loss for probabilistic evaluation.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats


def evaluate_with_stats(y_true, predictions):
    """
    Evaluate all models using point and probabilistic metrics.
    
    Metrics: R², MAE, RMSE, MAPE, 95% CI, CRPS, Pinball Loss.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    predictions : dict
        Model name → prediction array mapping.
    
    Returns
    -------
    results : dict
        Model name → metrics dict mapping.
    """
    print("\n[7/10] Evaluating forecasting performance...")
    results = {}

    for name, y_pred in predictions.items():
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        residuals = y_true - y_pred
        ci_low = np.percentile(residuals, 2.5)
        ci_high = np.percentile(residuals, 97.5)

        results[name] = {
            'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
            'CI_low': ci_low, 'CI_high': ci_high
        }

        print(f"  {name:25s} | R²={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.1f}%")

    return results


def compute_crps(y_true, q10, q50, q90):
    """
    Approximate Continuous Ranked Probability Score using quantile predictions.
    
    Parameters
    ----------
    y_true, q10, q50, q90 : np.ndarray
        Actual values and quantile predictions.
    
    Returns
    -------
    float
        Mean CRPS value.
    """
    crps = np.zeros(len(y_true))
    quantiles = [0.1, 0.5, 0.9]
    preds = [q10, q50, q90]
    for q, pred in zip(quantiles, preds):
        error = y_true - pred
        crps += np.where(error >= 0, q * error, (q - 1) * error)
    return np.mean(crps)


def compute_pinball_loss(y_true, y_pred, tau=0.5):
    """
    Pinball loss at quantile tau (Eq. pinball in paper).
    
    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Actual values and quantile predictions.
    tau : float
        Target quantile level.
    
    Returns
    -------
    float
        Mean pinball loss.
    """
    error = y_true - y_pred
    return np.mean(np.where(error >= 0, tau * error, (tau - 1) * error))
