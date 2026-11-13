- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import os
import warnings

from model_utils import (
    normalize_interval,
    reset_keras_seeds,
    temporal_train_val_split,
    default_keras_callbacks,
    forecast_nn_direct,
)

#: Network sizes (README section 105). ``standard``: the shipped network, 64
#: filters per block and a dense head of 64 and 32. ``small``: 16 filters per
#: block and a linear output from the last step, no dense head; Bai, Kolter and
#: Koltun (arXiv 1803.01271) used 20-27 channels on their small tasks, and
#: Borovykh, Bohte and Oosterlee (arXiv 1703.04691) one filter per layer on
#: daily S&P 500 returns.
SIZES = {"standard": dict(filters=64, mlp_units=None),
         "small": dict(filters=16, mlp_units=[])}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================
# 1. TCN Residual Block
# ==========================================
def tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout):
    """
    Single Temporal Convolutional Network residual block with skip output.

    Architecture: two stacked causal Conv1D layers (each followed by
    LayerNorm and ReLU, with one SpatialDropout after the second
    activation), plus a residual connection.  If the channel dimension
    of the input differs from ``filters``, a 1x1 convolution projects
    the input before adding.  Returns both the residual output (fed
    to the next block) and a skip output (for cross-block aggregation).

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, timesteps, channels).
    filters : int
        Number of convolutional filters (output channels).
    kernel_size : int
        Width of the causal convolution kernel.
    dilation_rate : int
        Dilation factor controlling the receptive-field growth.
    dropout : float
        Dropout rate applied after the second activation.

    Returns
    -------
    residual : tf.Tensor
        Output tensor of shape (batch, timesteps, filters) — fed to
        the next block.
    skip : tf.Tensor
        Skip connection tensor of shape (batch, timesteps, filters) —
        collected and summed across all blocks.
    """
    conv = layers.Conv1D(
        filters, kernel_size, padding='causal',
        dilation_rate=dilation_rate,
        kernel_initializer='he_normal',
    )(x)
    conv = layers.LayerNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv1D(
        filters, kernel_size, padding='causal',
        dilation_rate=dilation_rate,
        kernel_initializer='he_normal',
    )(conv)
    conv = layers.LayerNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.SpatialDropout1D(dropout)(conv)

    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, kernel_initializer='he_normal')(x)

    residual = layers.Add()([x, conv])
    skip = conv

    return residual, skip


def _compute_dilations(look_back, kernel_size=3):
    """
    Compute dilation rates so the receptive field fits the look-back.

    Grows dilations exponentially (1, 2, 4, ...) until the cumulative
    receptive field reaches or exceeds ``look_back``.  Avoids wasted
    layers whose receptive fields overshoot the input length.

    Parameters
    ----------
    look_back : int
        Input sequence length.
    kernel_size : int, optional
        Convolution kernel width. Default 3.

    Returns
    -------
    list of int
        Dilation rates for the TCN blocks.
    """
    dilations = []
    receptive = 1
    d = 1
    while receptive < look_back:
        dilations.append(d)
        receptive += (kernel_size - 1) * d * 2
        d *= 2
    return dilations


# ==========================================
# 2. Build TCN Model
# ==========================================
def build_tcn_model(input_shape, filters=64, kernel_size=3,
                    dilations=None, dropout=0.15, mlp_units=None,
                    output_dim=1):
    """
    Build a Temporal Convolutional Network for time-series regression.

    Architecture: stack of TCN residual blocks with exponentially
    increasing dilation rates (auto-sized to the look-back window)
    and cross-block skip connections -> last-timestep extraction
    from summed skips -> MLP head -> single output.

    Skip connections aggregate information at multiple temporal
    scales and the last-timestep extraction focuses the model on the
    most recent state, analogous to the final hidden state in an RNN.

    Parameters
    ----------
    input_shape : tuple of int
        (sequence_length, n_features).
    filters : int, optional
        Number of convolutional filters per block. Default 64.
    kernel_size : int, optional
        Causal convolution kernel width. Default 3.
    dilations : list of int, optional
        Dilation rates for successive blocks.  If None, auto-computed
        from ``input_shape[0]`` to just cover the look-back window.
    dropout : float, optional
        Dropout rate (single layer per block). Default 0.15.
    mlp_units : list of int, optional
        MLP head hidden sizes. Default [64, 32].

    Returns
    -------
    tf.keras.Model
        Compiled TCN model.
    """
    if dilations is None:
        dilations = _compute_dilations(input_shape[0], kernel_size)
    if mlp_units is None:
        mlp_units = [64, 32]

    inputs = Input(shape=input_shape)
    x = inputs
    skip_connections = []

    for d in dilations:
        x, skip = tcn_residual_block(x, filters, kernel_size, d, dropout)
        skip_connections.append(skip)

    if len(skip_connections) > 1:
        x = layers.Add()(skip_connections)
    else:
        x = skip_connections[0]

    x = layers.Lambda(lambda t: t[:, -1, :])(x)

    for dim in mlp_units:
        x = layers.Dense(dim, activation='relu',
                         kernel_initializer='he_normal')(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_dim, kernel_initializer='he_normal')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='huber',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    )
    return model


# ==========================================
# 3. Training Wrapper
# ==========================================
def train_tcn(X, y, output_dim=1, size="standard"):
    """
    Build and train the TCN model with temporal validation.

    Clears TF session, sets random seeds, splits data temporally
    (last 10 % for validation), and trains with early stopping
    and learning-rate scheduling.

    Parameters
    ----------
    X : np.ndarray
        Input sequences (n_samples, look_back, n_features).
    y : np.ndarray
        Targets of shape (n_samples,) or (n_samples, output_dim) — the
        per-horizon cumulative log returns for direct multi-horizon output.
    output_dim : int, optional
        Number of outputs (= forecast horizon for direct multi-horizon).
        Default 1.
    size : str, optional
        A key of :data:`SIZES` (default ``'standard'``).

    Returns
    -------
    tf.keras.Model
        Trained TCN model.
    """
    reset_keras_seeds()

    input_shape = X.shape[1:]

    X_train, X_val, y_train, y_val = temporal_train_val_split(X, y)

    print(
        f"Building TCN model "
        f"({input_shape[0]} steps x {input_shape[1]} features)..."
    )
    model = build_tcn_model(input_shape, output_dim=output_dim, **SIZES[size])

    print("Training TCN...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=default_keras_callbacks(),
        shuffle=True,
        verbose=0,
    )
    return model


def train_tcn_window_norm(X, Y, output_dim=1, size="standard"):
    """:func:`train_tcn` on per-window-normalised inputs and targets
    (``CNN_LSTM_model.WindowNormModel``)."""
    from CNN_LSTM_model import WindowNormModel
    Xt, sd = WindowNormModel.transform(X)
    return WindowNormModel(train_tcn(Xt, np.asarray(Y) / sd, output_dim=output_dim, size=size))


#: The configuration before README section 105.
LEGACY = dict(inputs="features", window_norm=False, size="standard", look_back=None)


# ==========================================
# 4. Main Orchestrator
# ==========================================
def run_tcn_prediction(df, interval, horizon, inputs="features", window_norm=False,
                       size="standard", look_back=None):
    """
    Main function to run TCN prediction.

    Orchestrates data preparation (9-feature engineering), TCN model
    building (dilated causal convolutions with cross-block skip
    connections), training with temporal validation, and direct
    multi-horizon forecasting (one output per step, no recursive feedback)
    with distribution-free conformal prediction intervals.  Dilation rates
    are auto-sized to the look-back window.  Training data is limited to
    recent observations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    inputs : str, optional
        ``'features'`` (default): the nine price features plus market context.
        ``'returns'``: the stock's own log returns only (README section 105).
    window_norm : bool, optional
        True: each window centred and divided by its own standard deviation,
        the forecast scaled back (needs ``inputs='returns'``).
    size : str, optional
        A key of :data:`SIZES` (default ``'standard'``).
    look_back : int, optional
        Window length; default 90 daily, 48 hourly; at least 10.

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'predicted_price', 'lower_bound',
        'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    if inputs not in ("features", "returns"):
        raise ValueError(f"inputs must be 'features' or 'returns', not {inputs!r}")
    if window_norm and inputs != "returns":
        raise ValueError("window_norm needs inputs='returns': a window of mixed "
                         "features has no single scale")
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}, not {size!r}")
    if look_back is not None and int(look_back) < 10:
        raise ValueError("look_back must be at least 10")
    print(f"--- Starting TCN Prediction ({interval}) ---")

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    max_obs = 1000 if interval == 'daily' else 750
    if len(df) > max_obs:
        df = df.tail(max_obs).reset_index(drop=True)

    if look_back is None:
        look_back = 90 if interval == 'daily' else 48
    look_back = int(look_back)
    trainer = train_tcn_window_norm if window_norm else train_tcn
    feature_fn = None
    if inputs == "returns":
        from NBeats_model import returns_only as feature_fn

    results_df = forecast_nn_direct(
        df, interval, horizon, look_back,
        build_train_fn=lambda X, Y, od: trainer(X, Y, output_dim=od, size=size),
        desc="TCN forecasts",
        model_name='TCN',
        feature_fn=feature_fn,
        use_context=inputs == "features",
    )

    print("TCN Prediction Complete.")
    return results_df
```
