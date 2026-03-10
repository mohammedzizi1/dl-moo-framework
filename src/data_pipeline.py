"""
Data loading, feature engineering, and feature selection pipeline.

This module handles:
- Loading and chronologically splitting the Kaggle Store Item Demand dataset
- Engineering temporal, lag, rolling, EWMA, trend, and entity-encoding features
- Selecting top-k features via univariate F-statistics
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def load_and_split_data(filepath='train.csv'):
    """
    Load the dataset and split chronologically into train/val/test sets.
    
    Splits: 65% train | 20% validation | 15% test (by date quantile).
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file (Kaggle Store Item Demand dataset).
    
    Returns
    -------
    train, val, test : pd.DataFrame
        Chronologically split subsets.
    df : pd.DataFrame
        Full dataset.
    """
    print("\n[1/10] Loading and analyzing data...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)

    train_split = df['date'].quantile(0.65)
    val_split = df['date'].quantile(0.85)

    train = df[df['date'] < train_split].copy()
    val = df[(df['date'] >= train_split) & (df['date'] < val_split)].copy()
    test = df[df['date'] >= val_split].copy()

    print(f"  Train: {len(train):,} ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"  Val:   {len(val):,} ({val['date'].min().date()} to {val['date'].max().date()})")
    print(f"  Test:  {len(test):,} ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"  Stores: {df['store'].nunique()} | Items: {df['item'].nunique()}")

    return train, val, test, df


def _add_features(df):
    """Add all engineered features to a dataframe."""
    # Temporal features
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # Cyclical encodings (Eq. 4 in paper)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    df = df.sort_values(['store', 'item', 'date'])

    # Lag features (Eq. 5)
    for lag in [1, 7, 14, 28, 56]:
        df[f'lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)

    # Rolling window statistics (Eq. 6)
    for w in [7, 28]:
        g = df.groupby(['store', 'item'])['sales']
        df[f'roll_mean_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f'roll_std_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        df[f'roll_max_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        df[f'roll_min_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).min())

    # EWMA (Eq. 7)
    df['ewm_14'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).ewm(span=14, min_periods=1).mean()
    )

    # Trend feature (Eq. 8)
    df['trend_28'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(28, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
        )
    )

    return df


def engineer_features(train, val, test):
    """
    Apply feature engineering to all data splits.
    
    Creates 31 features per observation: temporal, lag, rolling, EWMA,
    trend, and entity-encoding features.
    
    Parameters
    ----------
    train, val, test : pd.DataFrame
        Data splits from load_and_split_data().
    
    Returns
    -------
    train, val, test : pd.DataFrame
        Data splits with engineered features added.
    """
    print("\n[2/10] Engineering features...")

    train = _add_features(train)
    val = _add_features(val)
    test = _add_features(test)

    # Entity encodings (computed on train only to prevent leakage)
    global_mean = train['sales'].mean()
    store_enc = train.groupby('store')['sales'].mean()
    item_enc = train.groupby('item')['sales'].mean()
    si_enc = train.groupby(['store', 'item'])['sales'].mean()
    store_std = train.groupby('store')['sales'].std()
    item_std = train.groupby('item')['sales'].std()

    for df in [train, val, test]:
        df['store_mean'] = df['store'].map(store_enc).fillna(global_mean)
        df['item_mean'] = df['item'].map(item_enc).fillna(global_mean)
        df['si_mean'] = df.set_index(['store', 'item']).index.map(si_enc).fillna(global_mean)
        df['store_std'] = df['store'].map(store_std).fillna(train['sales'].std())
        df['item_std'] = df['item'].map(item_std).fillna(train['sales'].std())
        df.fillna(df.median(numeric_only=True), inplace=True)

    print(f"  Features created: {train.shape[1] - 3} (store, item, date excluded)")
    return train, val, test


def select_features(train, val, test, k=40):
    """
    Select top-k features using univariate F-statistics (Eq. 9 in paper).
    
    Parameters
    ----------
    train : pd.DataFrame
        Training data with engineered features.
    k : int
        Number of features to select.
    
    Returns
    -------
    selected_features : list
        Names of selected features.
    all_features : list
        Names of all available features.
    """
    print("\n[3/10] Selecting top features...")
    features = [col for col in train.columns if col not in ['date', 'sales', 'store', 'item']]
    X_tr = train[features].values
    y_tr = train['sales'].values

    selector = SelectKBest(f_regression, k=min(k, len(features)))
    selector.fit(X_tr, y_tr)
    selected_features = [features[i] for i in selector.get_support(indices=True)]
    print(f"  Selected {len(selected_features)} features from {len(features)}")
    return selected_features, features
