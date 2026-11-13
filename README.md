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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import os
import warnings

from model_utils import (
    normalize_interval,
    reset_keras_seeds,
    temporal_train_val_split,
    default_keras_callbacks,
    forecast_nn_direct,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================
# 1. Build & Train Model
# ==========================================
#: Network sizes other than the declared stack (README section 104).
#: ``small``: Fischer and Krauss (European Journal of Operational Research 2018,
#: "Deep learning with LSTM networks for financial market predictions"): one
#: recurrent layer of 25 units, dropout 0.1 on the inputs and on the recurrent
#: connections, straight to the output layer. ``single``: one layer of the
#: declared stack's first width with its dropout and dense head (depth only).
SIZES = ("standard", "single", "small")


def _layers(size, sp):
    """(units, output dropouts, input dropout, recurrent dropout, dense units)."""
    if size == "small":
        return [25], [], 0.1, 0.1, None
    units = list(sp.get('recurrent_units', [64, 64, 32]))
    drops = list(sp.get('dropout', [0.3, 0.3, 0.2]))
    rec_drop = float(sp.get('recurrent_dropout', 0.1))
    dense = int(sp.get('dense_units', 16))
    if size == "single":
        return units[:1], drops[:1], 0.0, rec_drop, dense
    return units, drops, 0.0, rec_drop, dense


def train_dl_model(X, y, model_type='LSTM', output_dim=1, size="standard"):
    """
    Build and train a 3-layer stacked LSTM or GRU model.

    Architecture: three recurrent layers (64, 64, 32 units) with
    output dropout (0.3, 0.3, 0.2) and recurrent dropout (0.1) for
    hidden-to-hidden regularisation, followed by a Dense bottleneck
    (16 units, ReLU) and an ``output_dim``-wide linear output. Trained
    with Huber loss, Adam optimiser with gradient clipping, learning-rate
    scheduling, and early stopping on a temporal validation set (last 10%).

    Parameters
    ----------
    X : np.ndarray
        Input sequences of shape (n_samples, look_back, n_features).
    y : np.ndarray
        Targets of shape (n_samples,) or (n_samples, output_dim) — the
        per-horizon cumulative log returns for direct multi-horizon output.
    model_type : str, optional
        'LSTM' or 'GRU' (default: 'LSTM').
    output_dim : int, optional
        Number of outputs (= forecast horizon for direct multi-horizon).
        Default 1.
    size : str, optional
        One of :data:`SIZES`: ``'standard'`` (default, the declared stack),
        ``'single'`` or ``'small'`` (README section 104).

    Returns
    -------
    tf.keras.Model
        Trained Keras model.
    """
    reset_keras_seeds()

    n_features = X.shape[2]
    look_back = X.shape[1]

    X_train, X_val, y_train, y_val = temporal_train_val_split(X, y)

    model = Sequential()

    # Layer sizes come from architectures.SPECS rather than literals here, so the
    # network is a declared, JSON-overridable object that the run manifest can
    # record. The defaults reproduce the previous hardcoded stack exactly
    # (64/64/32 recurrent, 0.3/0.3/0.2 dropout, Dense(16)) — see
    # tests/test_architectures.py, which pins that equivalence.
    import architectures as arch
    sp = arch.spec('DL')
    units, drops, in_drop, rec_drop, dense_units = _layers(size, sp)

    print(f"Building {model_type} model ({look_back} steps x {n_features} "
          f"features, units={units})...")

    layer = LSTM if model_type == 'LSTM' else GRU

    for i, n_units in enumerate(units):
        is_last = (i == len(units) - 1)
        kwargs = dict(return_sequences=not is_last, recurrent_dropout=rec_drop)
        if in_drop:
            kwargs['dropout'] = in_drop
        if i == 0:
            kwargs['input_shape'] = (look_back, n_features)
        model.add(layer(n_units, **kwargs))
        # A dropout list shorter than the unit list simply stops applying
        # dropout, rather than raising: an over-ridden spec should degrade
        # predictably, not crash a forecast.
        if i < len(drops):
            model.add(Dropout(drops[i]))

    if dense_units:
        model.add(Dense(dense_units, activation=sp.get('dense_activation', 'relu')))
    model.add(Dense(output_dim))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss='huber',
    )

    print("Training Model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=default_keras_callbacks(),
        shuffle=True,
        verbose=0,
    )

    return model


def train_dl_window_norm(X, Y, model_type='LSTM', output_dim=1, size="standard"):
    """:func:`train_dl_model` on per-window-normalised inputs and targets
    (``CNN_LSTM_model.WindowNormModel``)."""
    from CNN_LSTM_model import WindowNormModel
    Xt, sd = WindowNormModel.transform(X)
    return WindowNormModel(train_dl_model(Xt, np.asarray(Y) / sd, model_type,
                                          output_dim=output_dim, size=size))


#: The configuration before README section 104.
LEGACY = dict(inputs="features", window_norm=False, size="standard")

#: The GRU's inputs on daily bars beyond gating.MAX_DIRECTIONAL_SESSIONS (13):
#: the stock's own log returns, each window scaled by its own standard
#: deviation. Passed rules A1-A3 on a draw not used before (20-90 session score
#: 63.46% against 79.86%, lower at 5 of 6 origins; error 1.09 against 1.43
#: times no change's; run_returns_followup.py, README section 108). The LSTM
#: took the same default after passing the same rules on a further unused draw
#: (20-90 session score 119.30% against 130.40%, lower at 6 of 6 origins; error
#: 1.03 against 1.17; run_long_followup.py, README section 109). Hourly bars and
#: both models up to 13 sessions keep LEGACY.
GRU_LONG = dict(inputs="returns", window_norm=True)
#: The model types that take GRU_LONG beyond 13 daily sessions.
LONG_MODELS = ("GRU", "LSTM")


def resolve_switches(model_type, interval, horizon, inputs=None, window_norm=None):
    """
    The ``(inputs, window_norm)`` a run uses. A switch left as None takes the
    default: :data:`GRU_LONG` for the GRU and the LSTM (:data:`LONG_MODELS`) on
    daily bars beyond ``gating.MAX_DIRECTIONAL_SESSIONS``, otherwise
    :data:`LEGACY`. An explicit value is kept; an explicit ``inputs`` that
    differs from the default's keeps window scaling off unless it is asked for.
    """
    from gating import MAX_DIRECTIONAL_SESSIONS
    long = (str(model_type).upper() in LONG_MODELS and normalize_interval(interval) == "daily"
            and int(horizon) > MAX_DIRECTIONAL_SESSIONS)
    default = GRU_LONG if long else {k: LEGACY[k] for k in ("inputs", "window_norm")}
    if inputs is None:
        inputs = default["inputs"]
    if window_norm is None:
        window_norm = default["window_norm"] if inputs == default["inputs"] else False
    return inputs, bool(window_norm)


# ==========================================
# 2. Main Orchestrator
# ==========================================
def run_DL_prediction(df: pd.DataFrame, interval: str, horizon: int,
                      model_type='LSTM', inputs=None, window_norm=None,
                      size="standard"):
    """
    Main function to run LSTM or GRU prediction.

    Orchestrates data preparation (9-feature engineering with RSI,
    momentum, dual-window volatility, z-score, and acceleration),
    model training with recurrent dropout and temporal validation,
    and direct multi-horizon forecasting (one output per step, no
    recursive feedback) with distribution-free conformal prediction
    intervals. Training data is limited to recent observations to
    focus on the current market regime.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    model_type : str, optional
        'LSTM' (default) or 'GRU'.
    inputs : str, optional
        ``'features'``: the nine shared price features plus any market
        context. ``'returns'``: the stock's own log returns only (README
        sections 94, 103 and 104). None (default): :func:`resolve_switches`,
        own returns for the GRU and the LSTM on daily bars beyond 13 sessions
        (README sections 108 and 109), the nine features otherwise.
    window_norm : bool, optional
        True: each window centred and scaled by its own standard deviation,
        the forecast scaled back (needs ``inputs='returns'``). None (default):
        :func:`resolve_switches`.
    size : str, optional
        ``'standard'`` (default), ``'single'`` or ``'small'``; see :data:`SIZES`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    print(f"--- Starting {model_type} Prediction ({interval}) ---")
    inputs, window_norm = resolve_switches(model_type, interval, horizon, inputs, window_norm)
    if inputs not in ("features", "returns"):
        raise ValueError(f"inputs must be 'features' or 'returns', not {inputs!r}")
    if window_norm and inputs != "returns":
        raise ValueError("window_norm needs inputs='returns': a window of mixed "
                         "features has no single scale")
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, not {size!r}")

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    max_obs = 1000 if interval == 'daily' else 750
    if len(df) > max_obs:
        df = df.tail(max_obs).reset_index(drop=True)

    look_back = 90 if interval == 'daily' else 48

    trainer = train_dl_window_norm if window_norm else train_dl_model
    feature_fn = None
    if inputs == "returns":
        from NBeats_model import returns_only as feature_fn

    results_df = forecast_nn_direct(
        df, interval, horizon, look_back,
        build_train_fn=lambda X, Y, od: trainer(X, Y, model_type, output_dim=od, size=size),
        model_name='DL',
        feature_fn=feature_fn,
        use_context=inputs == "features",
    )

    print(f"{model_type} Prediction Complete.")
    return results_df
```
