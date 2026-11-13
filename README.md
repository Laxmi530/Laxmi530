- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Denoising autoencoder over price-feature windows -> a **regime-novelty** gate.

Motivated by *Hands-On Deep Learning for Finance* (Troiano et al.), Chapter 4,
"Index Replication by Autoencoders" (and its Chapter 1 warm-up), which builds a
vanilla autoencoder and then explores the **denoising** and **sparse** variants.
The book's application is index replication; this module repurposes the same
machinery for the question this project actually needs answered.

Why an autoencoder here, and not as a feature transform
-------------------------------------------------------
The tempting use of an autoencoder in a forecasting pipeline is dimensionality
reduction: compress the feature window to a latent code and feed that to the
predictor. This project has already run that experiment in a different guise and
reported the result honestly — the market-context features and excess-return
reframing both **hurt or did nothing** (README "Empirical findings"), because at
a 5-day single-name horizon the first moment is a random walk and extra
representation capacity is mostly extra overfitting surface. Adding a learned
compression in front of a model that cannot beat a random walk would be more of
the same, so latent features ship here as an available function
(:func:`latent_codes`) rather than as a default in any model.

The genuinely useful application is the autoencoder's *other* property: an AE
trained on one regime **reconstructs that regime well and anything else badly**.
Reconstruction error is therefore a distribution-shift detector. That plugs
straight into the part of this project that already carries the value — the
serving gate. The app abstains when the data is degraded (``data_quality``) and
when the walk-forward test says the model has no edge (``gating``); what it
cannot currently see is *"today's market state is unlike anything this model was
fitted on"*. A backtest-derived confidence bucket is a statement about the past
regime; it says nothing about a novel present. That blind spot is exactly what
this module closes.

How it stays honest
-------------------
* **Leak-free by construction.** The AE is fitted on the earliest block of
  windows, its reference error distribution is measured on a *held-out* middle
  block it never saw (in-sample errors are optimistically low and would
  exaggerate every novelty score), and only then is the most recent window
  scored. A caller inside a walk-forward backtest passes an origin-truncated
  frame and gets a genuinely out-of-sample novelty read.
* **Scale-free inputs.** Windows are built from ``compute_price_features``
  (returns, MA ratios, volatilities, RSI, z-score, acceleration) and standardized
  on training statistics only, so the score reflects *regime shape*, not the
  price level — a ₹100 stock and a ₹4000 stock produce comparable numbers.
* **Never fails a run.** TensorFlow is imported lazily and every failure path
  returns a neutral report (``verdict='unknown'``, no gate action), so enabling
  this can only ever add information.
"""

import os
import warnings

import numpy as np
import pandas as pd

from model_utils import (
    normalize_interval,
    coerce_numeric_columns,
    compute_price_features,
    reset_keras_seeds,
    default_keras_callbacks,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Window of bars compressed by the autoencoder. 20 bars (~1 month daily) is long
# enough to carry a regime's shape and short enough that ~900 rows still yield
# ~880 training windows.
DEFAULT_WINDOW = 20
# Rows below which the AE is not attempted (a network on a few dozen windows
# would report noise as novelty).
_MIN_ROWS = 260
# Fractions of the window sequence used to fit / to measure the reference error.
_FIT_FRAC, _REF_FRAC = 0.70, 0.20
# Novelty percentile thresholds -> verdict.
_UNUSUAL_PCT, _UNPRECEDENTED_PCT = 0.95, 0.995
_EPS = 1e-12

# Columns of compute_price_features the windows are built from. Column 0 is the
# raw adjusted close: standardised on the earliest 70% of history, a stock that
# has since trended carries a price far outside the fit range, and the AE then
# scores the LEVEL as a novel regime. The other eight are scale-free.
ALL_COLUMNS = tuple(range(9))
SCALE_FREE_COLUMNS = tuple(range(1, 9))


# ==========================================
# 1. Architecture (vanilla / denoising / sparse)
# ==========================================
def build_autoencoder(input_dim, latent_dim=8, hidden=(64, 32), noise_std=0.1,
                      l1_activity=0.0, dropout=0.0):
    """
    Build a symmetric autoencoder, optionally **denoising** and/or **sparse**.

    The three variants the book walks through, exposed as arguments on one
    architecture rather than three near-duplicate builders:

    * **Vanilla** (``noise_std=0``, ``l1_activity=0``) — plain bottleneck
      reconstruction. On financial windows it will happily learn a near-identity
      map if the bottleneck is generous, which makes it a poor novelty detector.
    * **Denoising** (``noise_std>0``) — Gaussian noise is injected into the input
      *during training only* (Keras ``GaussianNoise`` is a no-op at inference),
      and the target is the clean input. The network must therefore learn the
      data manifold rather than a copy operation, which is precisely what makes
      the reconstruction error of an *off-manifold* (novel-regime) window large.
      This is the default and the reason this module works at all.
    * **Sparse** (``l1_activity>0``) — an L1 activity penalty on the latent layer
      forces most codes toward zero, so each unit specialises. Useful when the
      latent code is going to be *read* (interpretability) rather than only
      scored.

    Encoder: ``input -> [GaussianNoise] -> Dense(64) -> Dense(32) ->
    Dense(latent, relu)``. Decoder mirrors it and ends in a linear layer, because
    the standardized inputs are unbounded (a sigmoid output would clip the tails,
    destroying exactly the extreme windows we want to flag).

    Parameters
    ----------
    input_dim : int
        Flattened window size (window * n_features).
    latent_dim : int, optional
        Bottleneck width. Default 8 — deliberately tight relative to a ~180-dim
        input so the code cannot memorise.
    hidden : tuple of int, optional
        Encoder hidden widths (mirrored in the decoder). Default (64, 32).
    noise_std : float, optional
        Std of the training-time input corruption. Default 0.1.
    l1_activity : float, optional
        L1 penalty on the latent activations (sparse variant). Default 0.
    dropout : float, optional
        Dropout between encoder layers. Default 0.

    Returns
    -------
    (tf.keras.Model, tf.keras.Model)
        ``(autoencoder, encoder)`` — the encoder shares weights, for
        :func:`latent_codes`.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input, regularizers

    inputs = Input(shape=(input_dim,))
    x = inputs
    if noise_std and noise_std > 0:
        x = layers.GaussianNoise(noise_std)(x)
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
        if dropout:
            x = layers.Dropout(dropout)(x)

    reg = regularizers.l1(l1_activity) if l1_activity else None
    latent = layers.Dense(latent_dim, activation='relu',
                          activity_regularizer=reg, name='latent')(x)

    y = latent
    for h in reversed(hidden):
        y = layers.Dense(h, activation='relu')(y)
    outputs = layers.Dense(input_dim, activation='linear')(y)

    autoencoder = Model(inputs, outputs, name='denoising_autoencoder')
    autoencoder.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    )
    encoder = Model(inputs, latent, name='encoder')
    return autoencoder, encoder


# ==========================================
# 2. Window construction
# ==========================================
def feature_windows(df, window=DEFAULT_WINDOW, columns=SCALE_FREE_COLUMNS):
    """
    Build flattened, scale-free feature windows from a raw stock frame.

    Reuses ``compute_price_features`` — the matrix the sequence models train
    on — so the autoencoder's notion of "market state" is the one the rest of
    the project already uses, minus its raw price column (see
    :data:`SCALE_FREE_COLUMNS`). The first 30 rows are dropped because
    the long rolling windows are incomplete there (matching
    ``prepare_feature_sequences``), and each window is flattened so a dense
    autoencoder can consume it.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close'.
    window : int, optional
        Bars per window. Default 20.
    columns : tuple of int, optional
        Columns of ``compute_price_features`` to use. Default the eight
        scale-free ones; :data:`ALL_COLUMNS` restores the raw price.

    Returns
    -------
    (np.ndarray, np.ndarray, int)
        ``(X, timestamps, n_features)`` where ``X`` has shape
        (n_windows, window * n_features) and ``timestamps[i]`` is the last bar of
        window ``i``. Returns empty arrays when there is too little history.
    """
    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    d = d.set_index('datetime').sort_index()
    coerce_numeric_columns(d)

    prices = d['adj_close'].dropna()
    if len(prices) < window + 40:
        return np.array([]), np.array([]), 0

    idx = prices.index
    feats = compute_price_features(prices.values.astype(np.float64))
    feats = feats[:, list(columns)]
    # Drop the rolling-window warmup rows (30 = the long SMA/vol window).
    feats, idx = feats[30:], idx[30:]
    n_features = feats.shape[1]

    n_win = len(feats) - window + 1
    if n_win < 40:
        return np.array([]), np.array([]), n_features

    X = np.stack([feats[i:i + window].reshape(-1) for i in range(n_win)])
    stamps = np.asarray(idx[window - 1:])
    return X.astype(np.float64), stamps, n_features


# ==========================================
# 3. Fit + score
# ==========================================
def fit_regime_autoencoder(df, interval='1d', window=DEFAULT_WINDOW,
                           latent_dim=8, noise_std=0.1, l1_activity=0.0,
                           epochs=150, batch_size=32, verbose=False,
                           columns=SCALE_FREE_COLUMNS):
    """
    Fit the denoising AE and measure its held-out reconstruction-error baseline.

    Three temporal blocks, in order, with no shuffling across them:

    1. **Fit block** (earliest 70% of windows) — trains the AE. The scaler's
       mean/std come from this block only.
    2. **Reference block** (next 20%) — never trained on. Its reconstruction
       errors form the baseline distribution that every later score is compared
       against. Using in-sample errors here would be the classic mistake: they
       are systematically too small, so *everything* subsequent would look novel.
    3. **Score block** (final 10%, plus whatever the caller scores later) — the
       recent windows whose novelty we actually care about.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close'.
    interval : str, optional
        '1d'/'daily' or '1h'/'hourly' (recorded in the result; the window is in
        bars either way).
    window : int, optional
        Bars per window. Default 20.
    latent_dim : int, optional
        Bottleneck width. Default 8.
    noise_std : float, optional
        Denoising corruption level. Default 0.1 (0 = vanilla AE).
    l1_activity : float, optional
        Latent L1 penalty (sparse AE). Default 0.
    epochs, batch_size : int, optional
        Training budget. Defaults 150 / 32 (early stopping usually fires first).
    verbose : bool, optional
        Print the Keras training log. Default False.

    Returns
    -------
    dict or None
        A bundle with keys ``autoencoder``, ``encoder``, ``mu``, ``sd``,
        ``window``, ``n_features``, ``ref_errors`` (the held-out baseline),
        ``score_errors`` / ``score_stamps`` (the recent block), and
        ``ref_quantiles``. Returns None when there is too little history or
        TensorFlow is unavailable.
    """
    interval = normalize_interval(interval)
    X, stamps, n_features = feature_windows(df, window=window, columns=columns)
    if len(X) == 0 or len(X) < _MIN_ROWS // 4:
        print(f"autoencoder_regime: only {len(X)} windows — skipping the AE.")
        return None

    n = len(X)
    n_fit = int(n * _FIT_FRAC)
    n_ref = max(20, int(n * _REF_FRAC))
    n_fit = min(n_fit, n - n_ref - 5)
    if n_fit < 60:
        print("autoencoder_regime: too few fit windows — skipping the AE.")
        return None

    fit_block = X[:n_fit]
    ref_block = X[n_fit:n_fit + n_ref]
    score_block = X[n_fit + n_ref:]
    score_stamps = stamps[n_fit + n_ref:]

    mu = fit_block.mean(axis=0)
    sd = fit_block.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)

    def _scale(a):
        return (a - mu) / sd

    try:
        reset_keras_seeds()
        ae, enc = build_autoencoder(
            X.shape[1], latent_dim=latent_dim, noise_std=noise_std,
            l1_activity=l1_activity,
        )
        Z = _scale(fit_block)
        # Last 10% of the fit block is the Keras validation split for early
        # stopping — temporally the most recent part of the fit block, so the
        # stopping signal is not measured on windows the AE has already memorised
        # from an earlier era.
        val = max(1, int(len(Z) * 0.1))
        ae.fit(Z[:-val], Z[:-val],
               validation_data=(Z[-val:], Z[-val:]),
               epochs=epochs, batch_size=batch_size, shuffle=True,
               callbacks=default_keras_callbacks(patience=12, lr_patience=6),
               verbose=1 if verbose else 0)
    except Exception as exc:
        print(f"autoencoder_regime: AE fit failed ({exc}); novelty unavailable.")
        return None

    ref_errors = reconstruction_error(ae, _scale(ref_block))
    score_errors = (reconstruction_error(ae, _scale(score_block))
                    if len(score_block) else np.array([]))

    if len(ref_errors):
        q = np.quantile(ref_errors, [0.5, 0.9, 0.95, 0.99])
        ref_q = dict(zip(['p50', 'p90', 'p95', 'p99'], [float(v) for v in q]))
        print(f"autoencoder_regime: AE fitted on {len(fit_block)} windows; "
              f"held-out error median {ref_q['p50']:.4f} (p95 {ref_q['p95']:.4f})")
    else:
        ref_q = {}
        print("autoencoder_regime: AE fitted (no reference block available)")

    return {
        'autoencoder': ae,
        'encoder': enc,
        'mu': mu,
        'sd': sd,
        'window': int(window),
        'columns': tuple(columns),
        'interval': interval,
        'n_features': int(n_features),
        'n_fit': int(len(fit_block)),
        'ref_errors': ref_errors,
        'ref_quantiles': ref_q,
        'score_errors': score_errors,
        'score_stamps': score_stamps,
    }


def reconstruction_error(autoencoder, X_scaled):
    """
    Per-window mean squared reconstruction error.

    The novelty score itself. Mean (not sum) so it is comparable across window
    sizes, and per-window (not pooled) so a caller can plot it as a time series
    and see *when* the regime drifted.
    """
    if len(X_scaled) == 0:
        return np.array([])
    recon = np.asarray(autoencoder.predict(X_scaled, verbose=0))
    return np.mean((np.asarray(X_scaled) - recon) ** 2, axis=1)


def latent_codes(bundle, df=None, X=None):
    """
    Encode windows into the AE's latent space.

    Provided for experimentation — compressed regime codes are a legitimate
    feature set for a *cross-sectional* or regime-conditioned model. It is
    deliberately **not** wired into any forecaster here: this project's own A/B
    evidence is that extra representation capacity does not improve single-name
    absolute-price forecasts at these horizons, and shipping an unvalidated
    feature transform would contradict that finding.

    Parameters
    ----------
    bundle : dict
        Output of :func:`fit_regime_autoencoder`.
    df : pd.DataFrame, optional
        Frame to encode (windows are rebuilt with the bundle's window size).
    X : np.ndarray, optional
        Pre-built unscaled windows, used instead of ``df`` when given.

    Returns
    -------
    np.ndarray
        (n_windows, latent_dim) codes; empty if nothing could be encoded.
    """
    if bundle is None:
        return np.array([])
    if X is None:
        if df is None:
            return np.array([])
        X, _, _ = feature_windows(df, window=bundle['window'],
                                  columns=bundle.get('columns', SCALE_FREE_COLUMNS))
    if len(X) == 0:
        return np.array([])
    Z = (X - bundle['mu']) / bundle['sd']
    return np.asarray(bundle['encoder'].predict(Z, verbose=0))


# ==========================================
# 4. The gate-facing report
# ==========================================
def regime_novelty(df, interval='1d', window=DEFAULT_WINDOW, bundle=None,
                   **fit_kwargs):
    """
    Score how unlike its training history the **latest** market window is.

    This is the function the serving layer calls. It fits (or reuses) the AE,
    reconstructs the most recent window, and expresses the error as a percentile
    of the held-out reference distribution:

    * ``< 95th`` -> ``'familiar'`` — the current regime looks like what the
      models were fitted on; the usual confidence machinery applies.
    * ``95th-99.5th`` -> ``'unusual'`` — a caller should widen the cone and treat
      the directional call with extra suspicion.
    * ``> 99.5th`` -> ``'unprecedented'`` — the state is off-manifold. Every
      model's backtest-derived confidence bucket is an extrapolation here, so the
      honest action is to withhold the directional call and show the risk cone
      only (see ``gating.apply_novelty_gate``).

    ``cone_multiplier`` is a modest, bounded suggestion (1.0-1.5) for scaling an
    uncertainty band, derived from the error ratio. It is intentionally *not*
    applied automatically anywhere: the HAR-RV/Vol-LSTM cones are conformally
    calibrated, and silently multiplying a calibrated interval by an
    uncalibrated factor would break the very guarantee that makes it useful.
    Treat it as a research knob to A/B, in the same spirit as the India-VIX tilt.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close'.
    interval : str, optional
        '1d'/'daily' or '1h'/'hourly'.
    window : int, optional
        Bars per window. Default 20.
    bundle : dict, optional
        A previously fitted bundle from :func:`fit_regime_autoencoder`, to avoid
        refitting (e.g. across several models in one request).
    **fit_kwargs
        Forwarded to :func:`fit_regime_autoencoder`.

    Returns
    -------
    dict
        {'available', 'verdict', 'error', 'percentile', 'ratio_to_p50',
         'cone_multiplier', 'reference', 'asof', 'note'}. On any failure:
        ``available=False``, ``verdict='unknown'`` and a populated ``note`` —
        never an exception.
    """
    neutral = {'available': False, 'verdict': 'unknown', 'error': float('nan'),
               'percentile': float('nan'), 'ratio_to_p50': float('nan'),
               'cone_multiplier': 1.0, 'reference': {}, 'asof': None, 'note': ''}

    try:
        if bundle is None:
            bundle = fit_regime_autoencoder(df, interval=interval,
                                            window=window, **fit_kwargs)
        if bundle is None:
            return {**neutral, 'note': 'insufficient history for the autoencoder'}

        X, stamps, _ = feature_windows(
            df, window=bundle['window'],
            columns=bundle.get('columns', SCALE_FREE_COLUMNS))
        if len(X) == 0:
            return {**neutral, 'note': 'no usable feature windows'}

        Z = (X[-1:] - bundle['mu']) / bundle['sd']
        err = float(reconstruction_error(bundle['autoencoder'], Z)[0])

        ref = np.asarray(bundle['ref_errors'], dtype=np.float64)
        if len(ref) < 10:
            return {**neutral, 'error': err,
                    'note': 'reference error distribution too small'}

        pct = float(np.mean(ref <= err))
        p50 = float(np.median(ref))
        ratio = float(err / p50) if p50 > _EPS else float('nan')

        if pct > _UNPRECEDENTED_PCT:
            verdict = 'unprecedented'
        elif pct > _UNUSUAL_PCT:
            verdict = 'unusual'
        else:
            verdict = 'familiar'

        # Bounded widening suggestion: sqrt of the error ratio (errors are in
        # squared units, bands are in sigma units), clipped to [1.0, 1.5] so a
        # single freak window can never blow the cone up.
        mult = 1.0
        if np.isfinite(ratio) and ratio > 1.0:
            mult = float(np.clip(np.sqrt(ratio), 1.0, 1.5))

        return {
            'available': True,
            'verdict': verdict,
            'error': err,
            'percentile': round(pct, 4),
            'ratio_to_p50': round(ratio, 3) if np.isfinite(ratio) else float('nan'),
            'cone_multiplier': round(mult, 3),
            'reference': bundle.get('ref_quantiles', {}),
            'asof': (pd.Timestamp(stamps[-1]).isoformat()
                     if len(stamps) else None),
            'note': '',
        }
    except Exception as exc:                     # novelty is advisory, never fatal
        return {**neutral, 'note': f'novelty check failed: {str(exc)[:120]}'}


def novelty_series(bundle):
    """
    The recent-block novelty scores as a tidy time series, for plotting.

    Returns the held-out *score block* errors alongside their timestamps and the
    reference percentile of each, so a caller can show "when did this stock's
    regime drift?" rather than only the latest scalar.

    Returns
    -------
    pd.DataFrame
        Columns ['datetime', 'error', 'percentile'] (empty if unavailable).
    """
    if not bundle or len(bundle.get('score_errors', [])) == 0:
        return pd.DataFrame(columns=['datetime', 'error', 'percentile'])
    ref = np.asarray(bundle['ref_errors'], dtype=np.float64)
    errs = np.asarray(bundle['score_errors'], dtype=np.float64)
    pct = ([float(np.mean(ref <= e)) for e in errs] if len(ref) >= 10
           else [float('nan')] * len(errs))
    return pd.DataFrame({'datetime': pd.to_datetime(bundle['score_stamps']),
                         'error': errs, 'percentile': pct})
```
