"""
Model training pipeline with ensemble weight optimization.

Handles training of all deep learning models and cross-validation-based
optimization of ensemble weights (Section III-D of the paper).
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow import keras

from .models import build_quantile_model, build_lstm, build_transformer


def train_advanced_models(train, val, test, selected_features, baselines):
    """
    Train all deep learning models and compute ensemble predictions.
    
    Trains Quantile Regression, BiLSTM, and Transformer models, then
    optimizes ensemble weights via grid search on validation MAE (Eq. ensemble_weights).
    
    Parameters
    ----------
    train, val, test : pd.DataFrame
        Data splits with engineered features.
    selected_features : list
        Feature names to use.
    baselines : dict
        Baseline predictions (used for comparison).
    
    Returns
    -------
    all_predictions : dict
        Model name → prediction array mapping for all models.
    y_test : np.ndarray
        True test values.
    """
    print("\n[5/10] Training advanced deep learning models...")

    # Prepare data
    X_train = train[selected_features].values
    y_train = train['sales'].values
    X_val = val[selected_features].values
    y_val = val['sales'].values
    X_test = test[selected_features].values
    y_test = test['sales'].values

    # Robust scaling (Eq. 10)
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    all_predictions = {}

    # Include baseline predictions
    for name, pred in baselines.items():
        all_predictions[name] = pred

    # Early stopping callback
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0)

    # --- Quantile Regression ---
    print("  Training Quantile Regression model...")
    qmodel = build_quantile_model(X_train_s.shape[1])
    qmodel.fit(X_train_s, [y_train]*3, validation_data=(X_val_s, [y_val]*3),
               epochs=100, batch_size=256, callbacks=[es], verbose=0)
    q10, q50, q90 = qmodel.predict(X_test_s, verbose=0)
    all_predictions['Quantile (p10)'] = np.maximum(q10.flatten(), 0)
    all_predictions['Quantile (p50)'] = np.maximum(q50.flatten(), 0)
    all_predictions['Quantile (p90)'] = np.maximum(q90.flatten(), 0)
    print("    ✓ Quantile model trained (p10, p50, p90)")

    # --- BiLSTM ---
    print("  Training BiLSTM model...")
    X_train_3d = X_train_s.reshape(-1, 1, X_train_s.shape[1])
    X_val_3d = X_val_s.reshape(-1, 1, X_val_s.shape[1])
    X_test_3d = X_test_s.reshape(-1, 1, X_test_s.shape[1])

    lstm_model = build_lstm(X_train_s.shape[1])
    lstm_model.compile(optimizer=keras.optimizers.Adam(0.001), loss='huber')
    lstm_model.fit(X_train_3d, y_train, validation_data=(X_val_3d, y_val),
                   epochs=100, batch_size=256, callbacks=[es], verbose=0)
    lstm_pred = np.maximum(lstm_model.predict(X_test_3d, verbose=0).flatten(), 0)
    all_predictions['Advanced LSTM'] = lstm_pred
    print("    ✓ BiLSTM model trained")

    # --- Transformer ---
    print("  Training Transformer model...")
    trans_model = build_transformer(X_train_s.shape[1])
    trans_model.compile(optimizer=keras.optimizers.Adam(0.001), loss='huber')
    trans_model.fit(X_train_3d, y_train, validation_data=(X_val_3d, y_val),
                    epochs=100, batch_size=256, callbacks=[es], verbose=0)
    trans_pred = np.maximum(trans_model.predict(X_test_3d, verbose=0).flatten(), 0)
    all_predictions['Transformer'] = trans_pred
    print("    ✓ Transformer model trained")

    # --- Cross-Validation-Optimized Weighted Ensemble (Eq. ensemble_weights) ---
    print("  Optimizing ensemble weights via grid search on validation MAE...")
    
    # Get validation predictions for weight optimization
    q_val = qmodel.predict(X_val_s, verbose=0)[1].flatten()  # q50
    lstm_val = lstm_model.predict(X_val_3d, verbose=0).flatten()
    trans_val = trans_model.predict(X_val_3d, verbose=0).flatten()
    
    best_mae = float('inf')
    best_weights = (1/3, 1/3, 1/3)
    
    for w1 in np.arange(0, 1.01, 0.01):
        for w2 in np.arange(0, 1.01 - w1, 0.01):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            ens_val = w1 * q_val + w2 * lstm_val + w3 * trans_val
            mae = np.mean(np.abs(y_val - ens_val))
            if mae < best_mae:
                best_mae = mae
                best_weights = (w1, w2, w3)
    
    w1, w2, w3 = best_weights
    print(f"    Optimal weights: QR={w1:.2f}, LSTM={w2:.2f}, Trans={w3:.2f}")
    
    ensemble_pred = (w1 * all_predictions['Quantile (p50)'] +
                     w2 * lstm_pred +
                     w3 * trans_pred)
    all_predictions['Weighted Ensemble'] = np.maximum(ensemble_pred, 0)
    print("    ✓ Weighted Ensemble created")

    return all_predictions, y_test
