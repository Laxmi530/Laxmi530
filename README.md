- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
CNN-LSTM: an LSTM stacked over a 1-D convolutional feature extractor.

Motivated by *Hands-On Deep Learning for Finance* (Troiano et al.), Chapter 7,
"Asset Allocation by LSTM over a CNN" — the hybrid the book uses for tactical
allocation, where convolutions first compress each window into local patterns
and a recurrent layer then models how those patterns evolve.

Why add it to this zoo
----------------------
The project already has four sequence architectures, but they all read the
look-back window in one particular way:

* **LSTM / GRU** consume the window step-by-step, so every bar gets equal
  attention and short local shapes must be re-learned at each offset.
* **Transformer** attends globally, which is powerful but data-hungry — the
  weakest fit for ~1000 rows.
* **TCN** convolves with dilations, so it is *all* convolution and no recurrence.
* **N-BEATS** flattens the window entirely, discarding temporal ordering inside
  the block.

CNN-LSTM occupies the remaining, genuinely distinct corner: **local translation-
invariant pattern extraction followed by temporal state**. The Conv1D stack acts
as a learned, multi-scale technical-indicator bank — a filter can key on "three
rising bars then a gap" wherever it occurs in the window — and max-pooling
shortens the sequence so the LSTM models a handful of *pattern activations* over
time rather than 90 raw bars. The shorter recurrent path is the practical
benefit: fewer timesteps means less vanishing-gradient trouble and far fewer
recurrent parameters, which is exactly what a small-sample financial series
wants.

It reuses the shared ``forecast_nn_direct`` engine, so it inherits the whole
robustness stack for free: **direct** multi-horizon targets (one output per step,
no recursive error accumulation), a training-block-only feature scaler, an
embargoed train/validation split, and per-horizon **split-conformal** prediction
intervals. Nothing about it is exempt from the project's honest read — the
first-moment evidence says a random walk is hard to beat — but it is a
cheap, architecturally distinct vote in the model comparison.
"""

import os
import warnings

import numpy as np
import pandas as pd

from model_utils import (
    normalize_interval,
    reset_keras_seeds,
    temporal_train_val_split,
    default_keras_callbacks,
    forecast_nn_direct,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================
# 1. Architecture
# ==========================================
def build_cnn_lstm_model(input_shape, output_dim=1, filters=None,
                         kernel_size=None, pool_size=None, lstm_units=None,
                         dropout=None):
    """
    Build the CNN -> LSTM -> Dense regression network.

    Layer plan (the book's "LSTM over a CNN" arrangement):

    ``Conv1D(32, causal) -> BN -> ReLU -> Conv1D(64, causal) -> BN -> ReLU ->
    MaxPool -> LSTM(48, seq) -> Dropout -> LSTM(24) -> Dropout -> Dense(16) ->
    Dense(output_dim)``

    Three deliberate choices:

    * **Causal padding.** ``padding='causal'`` means a filter at position *t*
      only sees positions <= *t*. With ``'same'`` padding a convolution reaches
      forward, which would leak future bars of the window into earlier
      timesteps — harmless for a plain classifier but wrong for a time series,
      and the kind of leak that quietly inflates validation scores.
    * **BatchNorm between conv and activation.** Financial feature channels have
      wildly different scales even after min-max scaling (returns vs RSI); the
      normalisation keeps the conv stack trainable at a single learning rate.
    * **Pool once, not twice.** A single ``MaxPool1D`` halves the sequence for
      the LSTM (90 -> 45 steps) which is the point of the hybrid; pooling harder
      would start destroying the recent-bar detail that dominates a
      short-horizon forecast.

    Parameters
    ----------
    input_shape : tuple of int
        (look_back, n_features).
    output_dim : int, optional
        Number of outputs (= forecast horizon for direct multi-horizon).
    filters : tuple of int, optional
        Channel widths of the two convolution layers. Default (32, 64).
    kernel_size : int, optional
        Convolution kernel width. Default 3.
    pool_size : int, optional
        Max-pooling factor. Default 2.
    lstm_units : tuple of int, optional
        Hidden sizes of the two LSTM layers. Default (48, 24).
    dropout : float, optional
        Dropout rate after each recurrent layer. Default 0.2.

    Returns
    -------
    tf.keras.Model
        Compiled model (Huber loss, Adam with gradient clipping).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input

    # Defaults resolve from architectures.SPECS rather than the signature, so the
    # shipped network is a declared object the run manifest can record and a JSON
    # file can override. Explicit arguments still win, which keeps this usable as
    # a plain builder in tests.
    import architectures as arch
    _sp = arch.spec('CNN-LSTM')
    filters = tuple(filters if filters is not None else _sp.get('filters', (32, 64)))
    kernel_size = int(kernel_size if kernel_size is not None
                      else _sp.get('kernel_size', 3))
    pool_size = int(pool_size if pool_size is not None
                    else _sp.get('pool_size', 2))
    lstm_units = tuple(lstm_units if lstm_units is not None
                       else _sp.get('lstm_units', (48, 24)))
    dropout = float(dropout if dropout is not None else _sp.get('dropout', 0.2))

    inputs = Input(shape=input_shape)

    x = inputs
    for f in filters:
        x = layers.Conv1D(f, kernel_size, padding='causal',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Only pool when the window is long enough to spare the resolution.
    if input_shape[0] >= pool_size * 8:
        x = layers.MaxPooling1D(pool_size=pool_size)(x)

    x = layers.LSTM(lstm_units[0], return_sequences=True,
                    recurrent_dropout=0.1)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units[1], return_sequences=False,
                    recurrent_dropout=0.1)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(output_dim)(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='huber',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    )
    return model


# ==========================================
# 2. Training wrapper
# ==========================================
def train_cnn_lstm(X, Y, output_dim=1):
    """
    Build and train the CNN-LSTM with an embargoed temporal validation split.

    Matches the other sequence models' trainer contract
    (``fn(X, Y, output_dim) -> model``) so it drops straight into
    ``forecast_nn_direct``. The embargo in ``temporal_train_val_split`` purges a
    gap equal to the horizon between train and validation, so overlapping
    multi-horizon label windows cannot leak into the early-stopping signal.

    Parameters
    ----------
    X : np.ndarray
        (n_samples, look_back, n_features) input windows.
    Y : np.ndarray
        (n_samples, output_dim) per-horizon cumulative log-return targets.
    output_dim : int, optional
        Number of outputs (= horizon). Default 1.

    Returns
    -------
    tf.keras.Model
        Trained model.
    """
    reset_keras_seeds()

    X_train, X_val, Y_train, Y_val = temporal_train_val_split(X, Y)

    print(f"Building CNN-LSTM model "
          f"({X.shape[1]} steps x {X.shape[2]} features -> {output_dim})...")
    model = build_cnn_lstm_model(X.shape[1:], output_dim=output_dim)

    print("Training CNN-LSTM...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=32,
        callbacks=default_keras_callbacks(),
        shuffle=True,
        verbose=0,
    )
    return model


class WindowNormModel:
    """
    A trained CNN-LSTM that reads each window centred and divided by its own
    standard deviation, with the targets divided by the same number and the
    forecast multiplied back: reversible instance normalization (Kim et al.,
    ICLR 2022), without its mean on the output, since a return window's mean
    is not a forecast of it. Univariate windows only. Attribute reads other
    than ``predict`` go to the network.
    """

    def __init__(self, model=None):
        self.model = model

    @staticmethod
    def transform(X):
        """The normalised windows and each window's scale, shape (n, 1)."""
        X = np.asarray(X, dtype=np.float32)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mu) / sd, sd[:, 0, :1]

    def predict(self, X, verbose=0, **kwargs):
        """Forecast in the targets' units."""
        Xt, sd = self.transform(X)
        return np.asarray(self.model.predict(Xt, verbose=0)) * sd

    def __getattr__(self, name):
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)


def train_cnn_lstm_window_norm(X, Y, output_dim=1):
    """:func:`train_cnn_lstm` on per-window-normalised inputs and targets."""
    Xt, sd = WindowNormModel.transform(X)
    return WindowNormModel(train_cnn_lstm(Xt, np.asarray(Y) / sd, output_dim=output_dim))


#: The configuration before README section 102.
LEGACY = dict(inputs="features", window_norm=False, look_back=None)

#: The inputs on daily bars up to gating.MAX_DIRECTIONAL_SESSIONS (13): the
#: stock's own log returns, each window scaled by its own standard deviation.
#: Passed rules B1-B3 (error first) on a draw not used before (five-session
#: error 1.06 against 1.31 times no change's, lower at 5 of 6 origins; score
#: 19.22% against 19.96%; run_returns_followup.py, README section 108). Hourly
#: bars and daily horizons beyond 13 sessions keep LEGACY.
SHORT_DEFAULT = dict(inputs="returns", window_norm=True)


def resolve_switches(interval, horizon, inputs=None, window_norm=None):
    """
    The ``(inputs, window_norm)`` a run uses. A switch left as None takes the
    default: :data:`SHORT_DEFAULT` on daily bars up to
    ``gating.MAX_DIRECTIONAL_SESSIONS``, otherwise :data:`LEGACY`. An explicit
    value is kept; an explicit ``inputs`` that differs from the default's keeps
    window scaling off unless it is asked for.
    """
    from gating import MAX_DIRECTIONAL_SESSIONS
    short = normalize_interval(interval) == "daily" and int(horizon) <= MAX_DIRECTIONAL_SESSIONS
    default = SHORT_DEFAULT if short else {k: LEGACY[k] for k in ("inputs", "window_norm")}
    if inputs is None:
        inputs = default["inputs"]
    if window_norm is None:
        window_norm = default["window_norm"] if inputs == default["inputs"] else False
    return inputs, bool(window_norm)


# ==========================================
# 3. Main orchestrator
# ==========================================
def run_cnn_lstm_prediction(df: pd.DataFrame, interval: str, horizon: int,
                            inputs=None, window_norm=None, look_back=None):
    """
    Main function to run CNN-LSTM prediction.

    Delegates to the shared direct multi-horizon engine, so the forecast is
    produced with one network output per step (no recursive feedback), a
    train-block-only scaler, and per-horizon distribution-free conformal
    prediction intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    inputs : str, optional
        ``'features'``: the nine shared price features plus any market
        context. ``'returns'``: the stock's own log returns only, the input
        that helped N-BEATS (README section 94) and that Mehtab and Sen found
        best for a CNN-LSTM on NIFTY (arXiv 2010.13891). None (default):
        :func:`resolve_switches`, own returns on daily bars up to 13 sessions
        (README section 108), the nine features otherwise.
    window_norm : bool, optional
        True: :class:`WindowNormModel` (needs ``inputs='returns'``). None
        (default): :func:`resolve_switches`.
    look_back : int, optional
        Window length; None: 90 sessions daily, 48 bars hourly.

    Returns
    -------
    pd.DataFrame
        Columns ['datetime', 'predicted_price', 'lower_bound', 'upper_bound',
        'volatility'].
    """
    interval = normalize_interval(interval)
    print(f"--- Starting CNN-LSTM Prediction ({interval}) ---")
    inputs, window_norm = resolve_switches(interval, horizon, inputs, window_norm)
    if inputs not in ("features", "returns"):
        raise ValueError(f"inputs must be 'features' or 'returns', not {inputs!r}")
    if window_norm and inputs != "returns":
        raise ValueError("window_norm needs inputs='returns': a window of mixed "
                         "features has no single scale")
    if look_back is not None and int(look_back) < 10:
        raise ValueError("look_back must be at least 10")

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    max_obs = 1000 if interval == 'daily' else 750
    if len(df) > max_obs:
        df = df.tail(max_obs).reset_index(drop=True)

    if look_back is None:
        look_back = 90 if interval == 'daily' else 48
    look_back = int(look_back)
    trainer = train_cnn_lstm_window_norm if window_norm else train_cnn_lstm
    feature_fn = None
    if inputs == "returns":
        from NBeats_model import returns_only as feature_fn

    results_df = forecast_nn_direct(
        df, interval, horizon, look_back,
        build_train_fn=lambda X, Y, od: trainer(X, Y, output_dim=od),
        desc="CNN-LSTM forecasts",
        model_name='CNN-LSTM',
        feature_fn=feature_fn,
        use_context=inputs == "features",
    )

    print("CNN-LSTM Prediction Complete.")
    return results_df
```
