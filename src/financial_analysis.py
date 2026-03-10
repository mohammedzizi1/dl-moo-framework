"""
Conservative financial impact assessment and ROI projections.

Implements the financial analysis from Table 7 of the paper, including
annualized savings with 85% realization factor.
"""

import numpy as np


def calculate_realistic_roi(savings, test_df, implementation_cost=500000, realization_factor=0.85):
    """
    Calculate conservative financial projections.
    
    Parameters
    ----------
    savings : float
        Total cost savings during the test period.
    test_df : pd.DataFrame
        Test data (used to determine test period duration).
    implementation_cost : float
        Estimated implementation cost (default: $500K).
    realization_factor : float
        Conservative realization factor (default: 85%).
    
    Returns
    -------
    dict
        Financial metrics: annualized savings, ROI, payback period, etc.
    """
    test_days = (test_df['date'].max() - test_df['date'].min()).days
    test_months = test_days / 30.44  # Average days per month

    annualized_linear = savings * (12 / test_months)
    conservative_annual = annualized_linear * realization_factor
    net_benefit = conservative_annual - implementation_cost
    roi = (net_benefit / implementation_cost) * 100
    payback_months = implementation_cost / (conservative_annual / 12)

    return {
        'test_period_months': round(test_months, 1),
        'test_period_savings': round(savings, 2),
        'annualized_linear': round(annualized_linear, 2),
        'conservative_annual': round(conservative_annual, 2),
        'implementation_cost': implementation_cost,
        'net_first_year': round(net_benefit, 2),
        'roi_percent': round(roi, 1),
        'payback_months': round(payback_months, 1)
    }
