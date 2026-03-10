"""
Deep learning model architectures for demand forecasting.

Implements three complementary architectures:
- Quantile Regression Neural Network (probabilistic forecasting)
- Bidirectional LSTM (sequential temporal dependencies)
- Transformer with self-attention (non-sequential feature interactions)

Each model exploits different inductive biases as described in Section III-C
of the paper.
"""

import tensorflow as tf
from tensorflow import keras


def quantile_loss(q):
    """
    Pinball loss for quantile regression (Eq. pinball in paper).
    
    Parameters
    ----------
    q : float
        Target quantile in (0, 1).
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss


def build_quantile_model(dim):
    """
    Build quantile regression neural network.
    
    Architecture: 3 hidden layers (256→128→64) with ReLU, BN, Dropout.
    Outputs: q10, q50, q90 quantile predictions.
    
    Parameters
    ----------
    dim : int
        Input feature dimension.
    
    Returns
    -------
    keras.Model
        Compiled multi-output quantile regression model.
    """
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001))(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    q10 = keras.layers.Dense(1, activation='relu', name='q10')(x)
    q50 = keras.layers.Dense(1, activation='relu', name='q50')(x)
    q90 = keras.layers.Dense(1, activation='relu', name='q90')(x)

    model = keras.Model(inp, [q10, q50, q90])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=[quantile_loss(0.1), quantile_loss(0.5), quantile_loss(0.9)]
    )
    return model


def build_lstm(dim):
    """
    Build Bidirectional LSTM model.
    
    Architecture: 2-layer BiLSTM (64, 32 units) with dropout and L2
    regularization. Uses Huber loss for robustness to outliers.
    
    Parameters
    ----------
    dim : int
        Input feature dimension.
    
    Returns
    -------
    keras.Model
        Compiled BiLSTM model.
    """
    inp = keras.Input(shape=(1, dim))
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True,
                          kernel_regularizer=keras.regularizers.l2(0.001))
    )(inp)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(1, activation='relu')(x)

    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='huber')
    return model


def build_transformer(dim):
    """
    Build Transformer model with self-attention.
    
    Architecture: 128-dim embedding, self-attention layer, position-wise
    feed-forward network with residual connection and LayerNorm.
    
    Parameters
    ----------
    dim : int
        Input feature dimension.
    
    Returns
    -------
    keras.Model
        Compiled Transformer model.
    """
    inp = keras.Input(shape=(1, dim))
    
    # Embedding layer
    x = keras.layers.Dense(128)(inp)
    
    # Self-attention (Eq. 17)
    attention = keras.layers.Dense(128, activation='softmax')(x)
    x_att = keras.layers.Multiply()([x, attention])
    
    # Feed-forward network (Eq. 18)
    ff = keras.layers.Dense(128, activation='relu')(x_att)
    ff = keras.layers.Dropout(0.2)(ff)
    ff = keras.layers.Dense(128)(ff)
    
    # Residual connection + LayerNorm
    x = keras.layers.Add()([x, ff])
    x = keras.layers.LayerNormalization()(x)
    
    # Output
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    out = keras.layers.Dense(1, activation='relu')(x)

    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='huber')
    return model
