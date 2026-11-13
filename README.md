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
from reproducibility import set_all_seeds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================
# 1. N-BEATS Block (Generic)
# ==========================================
class NBeatsBlock(layers.Layer):
    """
    Generic N-BEATS block with learned basis expansion.

    Applies a stack of fully connected layers to produce two outputs:
    a backcast (reconstruction of input) and a forecast (partial
    prediction). The doubly residual topology subtracts the backcast
    from the block's input, allowing subsequent blocks to model the
    residual signal.

    Parameters
    ----------
    units : int
        Width of each fully connected hidden layer.
    backcast_dim : int
        Dimensionality of the backcast output (flattened input length).
    forecast_dim : int
        Dimensionality of the forecast output (1 for single-step).
    n_layers : int, optional
        Number of FC hidden layers. Default 4.
    """

    def __init__(self, units, backcast_dim, forecast_dim, n_layers=4,
                 **kwargs):
        """
        Initialise block with FC layers, backcast, and forecast heads.

        Parameters
        ----------
        units : int
            Hidden layer width.
        backcast_dim : int
            Output dimension for backcast branch.
        forecast_dim : int
            Output dimension for forecast branch.
        n_layers : int, optional
            Number of hidden layers. Default 4.
        **kwargs
            Forwarded to ``layers.Layer``.
        """
        super().__init__(**kwargs)
        self.hidden = [
            layers.Dense(units, activation='relu') for _ in range(n_layers)
        ]
        self.theta_b = layers.Dense(backcast_dim, activation='linear')
        self.theta_f = layers.Dense(forecast_dim, activation='linear')

    def call(self, x):
        """
        Forward pass through hidden layers then backcast/forecast split.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, backcast_dim).

        Returns
        -------
        tuple of tf.Tensor
            backcast : (batch, backcast_dim) — input reconstruction.
            forecast : (batch, forecast_dim) — partial prediction.
        """
        h = x
        for layer in self.hidden:
            h = layer(h)
        return self.theta_b(h), self.theta_f(h)


# ==========================================
# 2. Full N-BEATS Model
# ==========================================
def build_nbeats_model(input_dim, n_stacks=2, n_blocks=2, units=64,
                       dropout=0.2, forecast_dim=1):
    """
    Build a Generic N-BEATS model for single-step regression.

    Architecture: flattened multivariate input -> N stacks of M generic
    blocks each -> sum of all block forecasts. Each block receives the
    residual from previous blocks (input minus accumulated backcasts).
    Dropout is applied between stacks for regularisation.

    Capacity is deliberately small (64 units x 2 blocks x 2 stacks,
    dropout 0.2). The original 256-unit / 3-block configuration badly
    overfit the ~700-row training windows and produced wildly inaccurate,
    direction-flipping forecasts in the walk-forward backtest; shrinking
    the network is the first-line remedy for that overfitting.

    Parameters
    ----------
    input_dim : int
        Flattened input dimensionality (look_back * n_features).
    n_stacks : int, optional
        Number of stacks. Default 2.
    n_blocks : int, optional
        Number of blocks per stack. Default 2.
    units : int, optional
        FC hidden layer width in each block. Default 64.
    dropout : float, optional
        Dropout rate between stacks. Default 0.2.

    Returns
    -------
    tf.keras.Model
        Compiled N-BEATS model.
    """
    inputs = Input(shape=(input_dim,))
    residuals = inputs
    forecast = layers.Lambda(
        lambda t: tf.zeros((tf.shape(t)[0], forecast_dim)),
        output_shape=(forecast_dim,),
    )(inputs)

    for s in range(n_stacks):
        for b in range(n_blocks):
            backcast, block_fcast = NBeatsBlock(
                units, input_dim, forecast_dim,
                name=f'nbeats_s{s}_b{b}',
            )(residuals)
            residuals = layers.Subtract()([residuals, backcast])
            forecast = layers.Add()([forecast, block_fcast])
        if s < n_stacks - 1:
            residuals = layers.Dropout(dropout)(residuals)

    model = Model(inputs, forecast)
    model.compile(
        loss='huber',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    )
    return model


# ==========================================
# 3. Training Wrapper
# ==========================================
def train_nbeats(X, y, look_back, n_features, output_dim=1):
    """
    Build and train the N-BEATS model with temporal validation.

    Clears the TF session, sets random seeds, splits data temporally
    (last 10 % for validation), and trains with early stopping and
    learning-rate scheduling.

    Parameters
    ----------
    X : np.ndarray
        Flattened input of shape (n_samples, look_back * n_features).
    y : np.ndarray
        Targets of shape (n_samples,) or (n_samples, output_dim) — the
        per-horizon cumulative log returns for direct multi-horizon output.
    look_back : int
        Sequence length before flattening.
    n_features : int
        Number of per-step features.
    output_dim : int, optional
        Number of forecast outputs (= horizon for direct multi-horizon).
        Default 1.

    Returns
    -------
    tf.keras.Model
        Trained N-BEATS model.
    """
    reset_keras_seeds()

    input_dim = look_back * n_features

    X_train, X_val, y_train, y_val = temporal_train_val_split(X, y)

    print(f"Building N-BEATS model (input_dim={input_dim})...")
    model = build_nbeats_model(input_dim, forecast_dim=output_dim)

    print("Training N-BEATS...")
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


# ==========================================
# 3b. Switches of README section 94
# ==========================================
#: Input per interval. Daily: the stock's own returns, chosen and then confirmed
#: on a fresh pre-2025 draw (README section 94). Hourly: the same, chosen on
#: 32 hourly forecasts and confirmed on 66 fresh ones before 2025
#: (run_hourly_neural.py, README section 119).
DEFAULT_INPUTS = {"daily": "returns", "hourly": "returns"}

#: The configuration before README section 94, for comparisons.
LEGACY = dict(inputs="features")

def returns_only(prices):
    """
    The series' own one-step log returns as the only input: N-BEATS as
    published is univariate (Oreshkin et al., ICLR 2020, section 2), and the
    nine shared features flatten to 540 inputs against about 600 training rows.
    """
    p = np.asarray(prices, dtype=np.float64)
    r = np.zeros(len(p))
    r[1:] = np.log(p[1:] / p[:-1])
    return r[:, None].astype(np.float32)


class NBeatsMember:
    """
    One trained network and how it reads a flattened window: the last
    ``keep`` steps of it and, with ``window_norm``, each window centred and
    divided by its own standard deviation, the targets divided by the same
    number and the forecast multiplied back (reversible instance
    normalization, Kim et al., ICLR 2022, without its mean on the output,
    since a return's window mean is not a forecast of it).
    """

    def __init__(self, keep, n_features, window_norm=False, model=None):
        self.keep, self.n_features = int(keep), int(n_features)
        self.window_norm, self.model = bool(window_norm), model

    def transform(self, X):
        """The member's input and, with window_norm, each row's scale."""
        X = np.asarray(X, dtype=np.float32)[:, -self.keep * self.n_features:]
        if not self.window_norm:
            return X, None
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mu) / sd, sd

    def predict(self, X, verbose=0):
        """Forecast in the targets' units."""
        Xt, sd = self.transform(X)
        out = np.asarray(self.model.predict(Xt, verbose=0))
        return out if sd is None else out * sd


class NBeatsEnsemble:
    """
    The median of several members' forecasts, as N-BEATS was published: its
    authors found ensembling "a much more powerful regularization technique"
    than dropout or an L2 penalty and used the median over seeds and look-backs
    (Oreshkin et al., ICLR 2020, section 3.4). Attribute reads other than
    ``predict`` go to the first member's network, so the architecture manifest
    still records a real network.
    """

    def __init__(self, members):
        self.members = list(members)

    def predict(self, X, verbose=0):
        """Element-wise median of the members' forecasts."""
        return np.median([m.predict(X) for m in self.members], axis=0)

    def __getattr__(self, name):
        if name == "members":
            raise AttributeError(name)
        return getattr(self.members[0].model, name)


def train_nbeats_members(X, y, look_back, n_features, output_dim=1,
                         lookbacks=None, n_seeds=1, window_norm=False):
    """
    Train one network per (look-back, seed) and return the plain Keras model
    when that is a single shipped-configuration network, otherwise an
    :class:`NBeatsEnsemble`. The first member is trained exactly as
    :func:`train_nbeats` (seed 42 after a cleared session); later members are
    reseeded without clearing, so the networks already trained stay usable.
    """
    lookbacks = tuple(lookbacks or (look_back,))
    if len(lookbacks) == 1 and n_seeds == 1 and not window_norm and lookbacks[0] == look_back:
        return train_nbeats(X, y, look_back, n_features, output_dim=output_dim)
    members = []
    for k, (lb, seed) in enumerate((lb, 42 + j) for j in range(n_seeds) for lb in lookbacks):
        member = NBeatsMember(lb, n_features, window_norm)
        Xt, sd = member.transform(X)
        yt = np.asarray(y) if sd is None else np.asarray(y) / sd
        if k == 0:
            reset_keras_seeds(seed)
        else:
            set_all_seeds(seed)
        X_train, X_val, y_train, y_val = temporal_train_val_split(Xt, yt)
        model = build_nbeats_model(Xt.shape[1], forecast_dim=output_dim)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,
                  batch_size=32, callbacks=default_keras_callbacks(), shuffle=True,
                  verbose=0)
        member.model = model
        members.append(member)
    print(f"Trained {len(members)} N-BEATS member(s): look-backs {lookbacks}, "
          f"{n_seeds} seed(s), window_norm={window_norm}")
    return NBeatsEnsemble(members)


# ==========================================
# 4. Main Orchestrator
# ==========================================
def run_nbeats_prediction(df, interval, horizon, inputs=None,
                          window_norm=False, n_seeds=1, lookbacks=None):
    """
    Main function to run N-BEATS prediction.

    Orchestrates data preparation (feature engineering, flattened
    sequences), model building (generic N-BEATS with doubly residual
    topology and `horizon` forecast outputs), training with temporal
    validation, and direct multi-horizon forecasting (no recursive
    feedback) with distribution-free conformal prediction intervals.
    Training data is limited to recent observations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    inputs : str, optional
        ``'returns'``: the stock's own log returns only (:func:`returns_only`),
        the default for daily bars since README section 94, where it lowered
        the five-session interval score and price error on a fresh draw.
        ``'features'``: the nine shared price features and any market context,
        flattened, as before (the default on hourly bars until section 119).
        None picks by interval: own returns on both.
    window_norm : bool, optional
        Scale each window by its own standard deviation
        (:class:`NBeatsMember`); needs ``inputs='returns'``.
    n_seeds : int, optional
        Networks per look-back, combined by the median
        (:class:`NBeatsEnsemble`). Default 1.
    lookbacks : tuple of int, optional
        Look-backs of the ensemble's members; None: the single default
        (60 daily, 30 hourly).

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'predicted_price', 'lower_bound',
        'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    print(f"--- Starting N-BEATS Prediction ({interval}) ---")
    if inputs is None:
        inputs = DEFAULT_INPUTS[interval]
    if inputs not in ("features", "returns"):
        raise ValueError(f"inputs must be 'features' or 'returns', not {inputs!r}")
    if window_norm and inputs != "returns":
        raise ValueError("window_norm needs inputs='returns': a window of mixed "
                         "features has no single scale")

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    max_obs = 1000 if interval == 'daily' else 750
    if len(df) > max_obs:
        df = df.tail(max_obs).reset_index(drop=True)

    look_back = 60 if interval == 'daily' else 30
    lbs = tuple(int(x) for x in (lookbacks or (look_back,)))
    window = max(lbs)

    results_df = forecast_nn_direct(
        df, interval, horizon, window,
        build_train_fn=lambda X, Y, od: train_nbeats_members(
            X, Y, window, X.shape[1] // window, output_dim=od,
            lookbacks=lbs, n_seeds=int(n_seeds), window_norm=window_norm),
        flatten=True, desc="N-BEATS forecasts",
        model_name='N-BEATS',
        feature_fn=returns_only if inputs == "returns" else None,
        use_context=inputs == "features",
    )

    print("N-BEATS Prediction Complete.")
    return results_df
```
