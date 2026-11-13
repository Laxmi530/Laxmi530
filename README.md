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
from tensorflow.keras import layers, models, Input
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
# 1. Transformer Architecture
# ==========================================
class InputEmbedding(layers.Layer):
    """
    Input projection with learnable positional embedding.

    Projects raw features to ``d_model`` dimensions via a Dense
    layer, then adds a learnable positional embedding so the
    Transformer knows the temporal position of each timestep
    in the sequence.
    """

    def __init__(self, max_len, d_model, **kwargs):
        """
        Initialise projection and positional embedding layers.

        Parameters
        ----------
        max_len : int
            Maximum sequence length (size of the positional embedding
            lookup table).
        d_model : int
            Dimensionality of the model's internal representation.
        **kwargs
            Additional keyword arguments forwarded to the base Layer.
        """
        super().__init__(**kwargs)
        self.projection = layers.Dense(d_model)
        self.pos_embedding = layers.Embedding(
            input_dim=max_len, output_dim=d_model,
        )

    def call(self, x):
        """
        Forward pass: project features to d_model and add positional embeddings.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, seq_len, n_features).

        Returns
        -------
        tf.Tensor
            Embedded tensor of shape (batch, seq_len, d_model).
        """
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x = self.projection(x)
        x = x + self.pos_embedding(positions)
        return x


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, causal=True):
    """
    Pre-norm Transformer encoder block with causal attention.

    Architecture per block:
    LayerNorm -> CausalMultiHeadAttention -> Residual ->
    LayerNorm -> FeedForward (Dense-ReLU-Dropout-Dense) -> Residual.

    Causal masking prevents each position from attending to future
    positions within the look-back window, which is more principled
    for autoregressive time-series prediction.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor of shape (batch, seq_len, d_model).
    head_size : int
        Dimensionality of each attention head (d_model // num_heads).
    num_heads : int
        Number of parallel attention heads.
    ff_dim : int
        Hidden dimensionality of the position-wise feed-forward network.
    dropout : float, optional
        Dropout rate applied in attention and the feed-forward network.
        Default is 0.
    causal : bool, optional
        Mask attention to earlier positions (default). The patch model of
        README section 103 attends both ways, as PatchTST does: every token
        lies inside the look-back window, so nothing leaks either way.

    Returns
    -------
    tf.Tensor
        Output tensor of same shape as *inputs*.
    """
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout,
    )(x, x, use_causal_mask=causal)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    return x + res


def build_transformer_model(input_shape,
                            d_model=64,
                            num_heads=4,
                            ff_dim=128,
                            num_blocks=3,
                            dropout=0.2,
                            mlp_units=None,
                            output_dim=1):
    """
    Build a Transformer encoder model for time-series regression.

    Architecture: InputEmbedding (projection + learnable positional
    encoding) -> N causal transformer encoder blocks ->
    GlobalAveragePooling1D -> MLP head -> single output. Compiled
    with Huber loss and Adam optimiser with gradient clipping.

    Parameters
    ----------
    input_shape : tuple of int
        (sequence_length, n_features) shape of each input sample.
    d_model : int, optional
        Internal representation dimension. Default 64.
    num_heads : int, optional
        Number of attention heads. Default 4.
    ff_dim : int, optional
        Hidden dimension of the feed-forward network. Default 128.
    num_blocks : int, optional
        Number of stacked transformer encoder blocks. Default 3.
    dropout : float, optional
        Dropout rate applied inside encoder blocks and MLP head.
        Default 0.2.
    mlp_units : list of int, optional
        Hidden layer sizes for the MLP head. Default [32].

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    if mlp_units is None:
        mlp_units = [32]

    inputs = Input(shape=input_shape)

    x = InputEmbedding(max_len=input_shape[0], d_model=d_model)(inputs)

    head_size = d_model // num_heads
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)

    for dim in mlp_units:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_dim)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        loss='huber',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    )
    return model


class Patcher(layers.Layer):
    """
    Cut each window into overlapping patches, PatchTST style (Nie et al.,
    ICLR 2023, arXiv 2211.14730): the window is padded at the end by repeating
    its last step ``stride`` times, then framed into patches of ``patch_len``
    steps every ``stride`` steps, giving ``(L - patch_len) // stride + 2``
    tokens of ``patch_len * n_features`` values each.
    """

    def __init__(self, patch_len, stride, **kwargs):
        super().__init__(**kwargs)
        self.patch_len, self.stride = int(patch_len), int(stride)

    def call(self, x):
        pad = tf.repeat(x[:, -1:, :], self.stride, axis=1)
        x = tf.concat([x, pad], axis=1)
        f = tf.signal.frame(x, self.patch_len, self.stride, axis=1)
        return tf.reshape(f, (tf.shape(f)[0], tf.shape(f)[1], self.patch_len * x.shape[-1]))


#: PatchTST's reduced model for its small datasets, "to mitigate the possible
#: overfitting" (heads 4, latent 16, feed-forward 128).
SMALL = dict(d_model=16, num_heads=4, ff_dim=128)
PATCH_LEN, PATCH_STRIDE = 16, 8


def n_patches(look_back, patch_len=PATCH_LEN, stride=PATCH_STRIDE):
    """Tokens the patch model makes from a window of ``look_back`` steps."""
    return (int(look_back) - min(patch_len, int(look_back))) // stride + 2


def build_patch_transformer_model(input_shape, patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                                  num_blocks=3, dropout=0.2, output_dim=1, **size):
    """
    The patch Transformer of README section 103: :class:`Patcher` tokens, a
    learnable position embedding, ``num_blocks`` encoder blocks attending both
    ways, then PatchTST's flatten-and-linear head. Size from :data:`SMALL`
    unless given.
    """
    size = {**SMALL, **size}
    look_back = input_shape[0]
    plen = min(patch_len, look_back)
    inputs = Input(shape=input_shape)
    x = Patcher(plen, stride)(inputs)
    x = InputEmbedding(max_len=n_patches(look_back, plen, stride), d_model=size["d_model"])(x)
    for _ in range(num_blocks):
        x = transformer_encoder(x, size["d_model"] // size["num_heads"], size["num_heads"],
                                size["ff_dim"], dropout, causal=False)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(output_dim)(x)
    model = models.Model(inputs, outputs)
    model.compile(
        loss='huber',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    )
    return model


ARCHES = ("standard", "small", "patch")


def _build(arch, input_shape, output_dim):
    if arch == "patch":
        return build_patch_transformer_model(input_shape, output_dim=output_dim)
    if arch == "small":
        return build_transformer_model(input_shape, output_dim=output_dim, **SMALL)
    return build_transformer_model(input_shape, output_dim=output_dim)


# ==========================================
# 2. Training Wrapper
# ==========================================
def train_transformer(X, y, output_dim=1, arch="standard"):
    """
    Build and train the Transformer model with temporal validation.

    Clears the TF session, sets random seeds for reproducibility,
    splits data temporally (last 10% for validation), and trains
    with early stopping and learning-rate scheduling.

    Parameters
    ----------
    X : np.ndarray
        Input sequences of shape (n_samples, look_back, n_features).
    y : np.ndarray
        Targets of shape (n_samples,) or (n_samples, output_dim) — the
        per-horizon cumulative log returns for direct multi-horizon output.
    output_dim : int, optional
        Number of outputs (= forecast horizon for direct multi-horizon).
        Default 1.
    arch : str, optional
        One of :data:`ARCHES`: ``'standard'`` (default), ``'small'`` or
        ``'patch'`` (README section 103).

    Returns
    -------
    tf.keras.Model
        Trained Keras Transformer model.
    """
    reset_keras_seeds()

    input_shape = X.shape[1:]

    X_train, X_val, y_train, y_val = temporal_train_val_split(X, y)

    print(
        f"Building Transformer Model "
        f"({input_shape[0]} steps x {input_shape[1]} features)..."
    )
    model = _build(arch, input_shape, output_dim)

    print("Training Transformer (this may take time)...")
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


def train_transformer_window_norm(X, Y, output_dim=1, arch="standard"):
    """:func:`train_transformer` on per-window-normalised inputs and targets
    (``CNN_LSTM_model.WindowNormModel``)."""
    from CNN_LSTM_model import WindowNormModel
    Xt, sd = WindowNormModel.transform(X)
    return WindowNormModel(train_transformer(Xt, np.asarray(Y) / sd, output_dim=output_dim, arch=arch))


#: The configuration before README section 103.
LEGACY = dict(inputs="features", window_norm=False, arch="standard")

#: The default by bar size. Daily: the stock's own log returns, each window
#: scaled by its own standard deviation, chosen on 32 forecasts and confirmed on
#: 72 fresh ones before the reserved period (run_transformer_improve.py, README
#: section 103). Hourly bars: the same, chosen on 32 hourly forecasts and
#: confirmed on 66 fresh ones before 2025 (run_hourly_neural.py, README
#: section 119), by a small margin (score 22.61% to 22.35%, 5 of 6 dates).
DEFAULTS = {"daily": dict(inputs="returns", window_norm=True),
            "hourly": dict(inputs="returns", window_norm=True)}


# ==========================================
# 3. Main Orchestrator
# ==========================================
def run_transformer_prediction(df: pd.DataFrame, interval: str, horizon: int,
                               inputs=None, window_norm=None, arch="standard"):
    """
    Main function to run Transformer prediction.

    Orchestrates data preparation (9-feature engineering with RSI,
    momentum, dual-window volatility, z-score, and acceleration),
    Transformer model training with causal attention and temporal
    validation, and direct multi-horizon forecasting (one output per
    step, no recursive feedback) with distribution-free conformal
    prediction intervals.  Training data is limited to recent
    observations to focus on the current regime.

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
        context. ``'returns'``: the stock's own log returns only (README
        sections 94 and 103). None (default): :data:`DEFAULTS` for the bar
        size: own returns on daily and hourly bars.
    window_norm : bool, optional
        True: each window centred and scaled by its own standard deviation,
        the forecast scaled back (needs ``inputs='returns'``). None (default):
        :data:`DEFAULTS`'s value when ``inputs`` is the default's, else False.
    arch : str, optional
        ``'standard'`` (default), ``'small'`` (PatchTST's reduced size) or
        ``'patch'`` (PatchTST-style patches and head); README section 103.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    print(f"--- Starting Transformer Prediction ({interval}) ---")
    default = DEFAULTS[interval]
    if inputs is None:
        inputs = default["inputs"]
    if window_norm is None:
        window_norm = default["window_norm"] if inputs == default["inputs"] else False
    if inputs not in ("features", "returns"):
        raise ValueError(f"inputs must be 'features' or 'returns', not {inputs!r}")
    if window_norm and inputs != "returns":
        raise ValueError("window_norm needs inputs='returns': a window of mixed "
                         "features has no single scale")
    if arch not in ARCHES:
        raise ValueError(f"arch must be one of {ARCHES}, not {arch!r}")

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    max_obs = 1000 if interval == 'daily' else 750
    if len(df) > max_obs:
        df = df.tail(max_obs).reset_index(drop=True)

    look_back = 90 if interval == 'daily' else 48

    trainer = train_transformer_window_norm if window_norm else train_transformer
    feature_fn = None
    if inputs == "returns":
        from NBeats_model import returns_only as feature_fn

    results_df = forecast_nn_direct(
        df, interval, horizon, look_back,
        build_train_fn=lambda X, Y, od: trainer(X, Y, output_dim=od, arch=arch),
        model_name='Transformer',
        feature_fn=feature_fn,
        use_context=inputs == "features",
    )

    print("Transformer Prediction Complete.")
    return results_df
```
