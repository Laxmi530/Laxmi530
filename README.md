- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Vol-LSTM: deep-learning **volatility** forecasting with a conformalized cone.

Motivated by *Hands-On Deep Learning for Finance* (Troiano et al.), Chapter 5,
"Volatility Forecasting by LSTM" — the chapter that builds an LSTM on realized
volatility, improves it with stacked layers and online learning, and then
benchmarks it against a plain RNN and a **GARCH** model using the cumulative
squared error.

Why this model belongs here
---------------------------
The project's own walk-forward evidence (README "Empirical findings") is blunt:
at single-name daily/hourly horizons the **first moment is a random walk** — no
feature set, tree, or network beat it in a data-snooping-robust test (White's
Reality Check p = 0.90). The **second moment** is different: volatility clusters
and is genuinely forecastable, which is why HAR-RV and the GARCH layer earn
their place. So the useful place to spend a neural network here is on
*variance*, not on direction.

This model is therefore the deep-learning **counterpart to HAR-RV**, and the
comparison the book runs (LSTM vs GARCH) is exactly the comparison this codebase
was missing:

* **HAR-RV** explains today's log-variance with three *linear* lags (day, week,
  month) — four OLS parameters, extremely hard to beat, but strictly linear and
  strictly additive in those three horizons.
* **Vol-LSTM** reads a whole look-back window of log-RV (plus the HAR averages
  and the signed-return channel) through stacked LSTM layers, so it can learn
  *non-linear* and *asymmetric* variance dynamics — notably the leverage effect
  (down-moves raise tomorrow's variance more than equal up-moves), which the
  symmetric HAR regression cannot express and which GJR/EGARCH only capture with
  a hand-specified term.

Contract and honesty
--------------------
Like HAR-RV, this model makes **no directional claim**: the point path is the
random walk (last price held flat), and the deliverable is the **calibrated
uncertainty cone**. That keeps it directly A/B-able against HAR-RV on the metric
that matters — interval coverage and sharpness — without smuggling in a
first-moment claim the evidence does not support.

The cone is built the same way as HAR-RV's, so the two are like-for-like:

1. Forecast per-step log realized variance for h = 1..horizon **directly**
   (one output per step, no recursive feedback — the same fix applied to the
   price models in ``forecast_nn_direct``).
2. Exponentiate with a **data-estimated level offset** so the variance forecast
   is unbiased (``log(r^2)`` sits ~1.27 nats below ``log(sigma^2)`` for Gaussian
   returns — see the note in ``_forecast_cone``), then cumulate (variance is
   additive across steps) to get sigma to each h.
3. Calibrate a **per-horizon conformal multiplier** from the standardized
   cumulative residuals on a held-out recent block, optionally refined by
   **Adaptive Conformal Inference** so the cone responds to a regime shift.
4. Wrap the flat path in ``last_price * exp(±k_h * sigma_h)``.

Because steps 3-4 reuse ``HAR_RV_model``'s ACI machinery, any improvement there
lands in both models, and the signed residuals are exported as
``attrs['risk_log_returns']`` so the app's risk panel reads fat-tailed empirical
probabilities off this model too.
"""

import os
import warnings

import numpy as np
import pandas as pd

from model_utils import (
    normalize_interval,
    coerce_numeric_columns,
    generate_future_dates,
    reset_keras_seeds,
    temporal_train_val_split,
    default_keras_callbacks,
)
from HAR_RV_model import (
    _har_lags,
    _aci_multipliers,
    _asymmetric_multipliers,
    _fallback_variance,
    _realized_semivariance,
    _Z95,
    _EPS,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Look-back window of RV bars fed to the LSTM. 60 daily bars (~3 months) covers
# the monthly HAR lag with room for the network to find longer structure; hourly
# uses 70 (= 10 NSE sessions of 7 bars).
_LOOK_BACK = {'daily': 60, 'hourly': 70}
# Recent fraction held out to conformalize the cone (never trained on).
_CAL_FRAC = 0.25
_MIN_CAL = 20

#: Guard on the served cone (README section 83). On some histories the network's
#: per-step variance swings far from anything realised, the conformal layer then
#: calibrates against that spread, and the band explodes (572% at five sessions
#: for HDFCBANK from 2021-01-01; 91 million percent at 13). When the forecast
#: leaves this range the cone falls back to the EWMA variance with the normal
#: multiplier, the same path as a short history: any per-step sd above
#: _GUARD_RATIO times, or below 1/_GUARD_RATIO of, the sd of the last
#: _GUARD_WINDOW returns; any conformal multiplier above _GUARD_K (normal 1.96);
#: or anything non-finite. Set before the effect was measured.
_GUARD_RATIO = 4.0
_GUARD_K = 6.0
_GUARD_WINDOW = 60
# Rows needed before the LSTM path is viable at all; below this we fall back to
# the EWMA cone rather than fitting a network on a handful of samples.
_MIN_ROWS = 260


# ==========================================
# 1. Realized-variance feature channels
# ==========================================
def build_rv_channels(returns, lags):
    """
    Build the per-bar input channels for the volatility network.

    Five channels, all strictly causal (each value at index ``k`` uses only bars
    up to and including ``k``):

    0. ``log_rv``        — log realized variance of the bar (``log(r^2)``), the
       target's own history and the single most informative input.
    1. ``log_rv_med``    — trailing mean of ``log_rv`` over the medium (weekly)
       window. Handing the network the HAR averages explicitly means it starts
       from HAR's information set instead of having to rediscover it from raw
       lags — a small, cheap inductive bias that matters a lot on ~1000 rows.
    2. ``log_rv_long``   — trailing mean over the long (monthly) window.
    3. ``signed``        — ``sign(r) * sqrt(RV)`` = the signed return itself. This
       is the channel HAR structurally cannot use: it lets the network learn the
       **leverage effect** (a −3% day implies more future variance than a +3%
       day), which is otherwise only available via a hand-coded GJR/EGARCH term.
    4. ``down_share``    — the bar's downside share of variance
       (``1`` if the return was negative, else ``0``, smoothed over the medium
       window) — Patton & Sheppard's "bad volatility", the empirically
       higher-signal half of realized variance.

    Parameters
    ----------
    returns : array-like
        Per-bar log returns.
    lags : tuple of int
        (short, medium, long) HAR memory windows in bars.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(channels, log_rv)`` where ``channels`` has shape (n_bars, 5) and
        ``log_rv`` is the 1-D target series (both aligned to ``returns``).
    """
    r = np.asarray(returns, dtype=np.float64)
    _, med, long = lags

    rv = r ** 2
    log_rv = np.log(rv + _EPS)
    # Clip the log-RV tails: a single near-zero return produces a hugely negative
    # log value that would dominate a squared-error fit.
    lo, hi = np.percentile(log_rv, [1, 99])
    log_rv = np.clip(log_rv, lo, hi)

    s = pd.Series(log_rv)
    ch_med = s.rolling(med, min_periods=1).mean().values
    ch_long = s.rolling(long, min_periods=1).mean().values

    rs_down, _ = _realized_semivariance(r)
    down_flag = (rs_down > 0).astype(np.float64)
    ch_down = pd.Series(down_flag).rolling(med, min_periods=1).mean().values

    signed = np.sign(r) * np.sqrt(np.maximum(rv, 0.0))

    channels = np.column_stack([log_rv, ch_med, ch_long, signed, ch_down])
    return channels.astype(np.float64), log_rv


# ==========================================
# 2. Network
# ==========================================
def build_vol_lstm(input_shape, output_dim, units=None, dropout=None):
    """
    Stacked LSTM regressor for direct multi-horizon log-variance forecasting.

    Two recurrent layers (the book's "stacking layers" improvement) with output
    and recurrent dropout, a small dense bottleneck, and ``output_dim`` linear
    outputs — one per horizon step, so no prediction is ever fed back and the
    h-step error is not an accumulation of 1-step errors.

    Capacity is intentionally modest (48/24 units). Volatility series are short
    (~1000 bars here) and smooth; the project already learned from N-BEATS that
    an over-parameterised network on ~700 rows produces spectacular
    out-of-sample failures. Huber loss keeps the fit from chasing the residual
    log-RV outliers that survive clipping.

    Parameters
    ----------
    input_shape : tuple of int
        (look_back, n_channels).
    output_dim : int
        Number of horizon steps to predict.
    units : tuple of int, optional
        Hidden sizes of the two LSTM layers. Default (48, 24).
    dropout : float, optional
        Output dropout rate. Default 0.2.

    Returns
    -------
    tf.keras.Model
        Compiled model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input

    # Defaults from architectures.SPECS; explicit arguments still win.
    import architectures as arch
    _sp = arch.spec('Vol-LSTM')
    units = tuple(units if units is not None else _sp.get('units', (48, 24)))
    dropout = float(dropout if dropout is not None else _sp.get('dropout', 0.2))

    inputs = Input(shape=input_shape)
    x = layers.LSTM(units[0], return_sequences=True, recurrent_dropout=0.1)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units[1], return_sequences=False, recurrent_dropout=0.1)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(output_dim)(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='huber',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    )
    return model


def train_vol_lstm(X, Y, epochs=120, batch_size=32, online_epochs=0):
    """
    Fit the volatility LSTM with an embargoed temporal validation split.

    Uses ``temporal_train_val_split``, which purges a gap equal to the number of
    target columns between train and validation — necessary here because each
    label spans ``horizon`` future bars, so without the embargo the last training
    labels and the first validation labels would share future bars and make
    early stopping look better than it is.

    Parameters
    ----------
    X : np.ndarray
        (n_samples, look_back, n_channels) input windows.
    Y : np.ndarray
        (n_samples, horizon) per-step log-variance targets.
    epochs : int, optional
        Maximum epochs (early stopping usually fires well before). Default 120.
    batch_size : int, optional
        Mini-batch size. Default 32.
    online_epochs : int, optional
        The book's **online learning** refinement: after the main fit, run this
        many extra low-learning-rate epochs on only the most recent 20% of
        samples, so the network is nudged toward the current volatility regime.
        Default 0 (off) — it is a recency bias that helps in a regime shift and
        hurts in a calm stretch, so it ships as an explicit opt-in rather than a
        silent default.

    Returns
    -------
    tf.keras.Model
        Trained model.
    """
    import tensorflow as tf

    reset_keras_seeds()
    X_train, X_val, Y_train, Y_val = temporal_train_val_split(X, Y)

    print(f"Building Vol-LSTM ({X.shape[1]} steps x {X.shape[2]} channels "
          f"-> {Y.shape[1]} horizons)...")
    model = build_vol_lstm(X.shape[1:], Y.shape[1])

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=default_keras_callbacks(),
        shuffle=True,
        verbose=0,
    )

    if online_epochs > 0 and len(X_train) > 40:
        tail = max(20, int(len(X_train) * 0.2))
        model.optimizer.learning_rate.assign(1e-4)
        print(f"  online fine-tune: {online_epochs} epochs on the last {tail} samples")
        model.fit(X_train[-tail:], Y_train[-tail:], epochs=online_epochs,
                  batch_size=16, shuffle=False, verbose=0)

    return model


# ==========================================
# 3. Dataset assembly
# ==========================================
def _make_samples(channels, log_rv, look_back, horizon, origins):
    """
    Assemble (window, per-step-log-RV-target) pairs for the given RV origins.

    An origin ``o`` means "the last observed RV bar is index ``o``": the input is
    ``channels[o - look_back + 1 : o + 1]`` and the target is
    ``log_rv[o + 1 : o + horizon + 1]``. Keeping this in one helper guarantees
    the training, calibration and serving paths index identically — the class of
    off-by-one that silently corrupts multi-horizon models.
    """
    X = np.stack([channels[o - look_back + 1:o + 1] for o in origins])
    Y = np.stack([log_rv[o + 1:o + horizon + 1] for o in origins])
    return X, Y


def _standardize(train_block):
    """Per-channel mean/std from the training block only (no lookahead)."""
    mu = train_block.reshape(-1, train_block.shape[-1]).mean(axis=0)
    sd = train_block.reshape(-1, train_block.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


# ==========================================
# 4. Main orchestrator
# ==========================================
def run_vol_lstm_prediction(df, interval, horizon, adaptive=True, aci_gamma=0.05,
                            asymmetric=False, online_epochs=0):
    """
    Forecast a volatility cone with a stacked LSTM; point path is the random walk.

    Mirrors ``HAR_RV_model.run_har_rv_prediction``'s contract exactly (same
    output columns, same flat point forecast, same conformal + ACI cone
    machinery) so the two can be A/B-ed on coverage and sharpness as a
    like-for-like linear-vs-deep comparison — the book's LSTM-vs-GARCH experiment
    transplanted onto this project's realized-variance setup.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close'.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to forecast.
    adaptive : bool, optional
        Refine the per-horizon conformal multiplier with Adaptive Conformal
        Inference so the cone tracks a recent volatility regime shift rather
        than the long-run calibration average. Default True (as in HAR-RV).
    aci_gamma : float, optional
        ACI learning rate. Default 0.05.
    asymmetric : bool, optional
        Build separate lower/upper cone edges from the signed residuals, so a
        fatter loss tail widens the downside. Default False — the same A/B that
        turned this off for HAR-RV applies (per-side calibration is noisier on
        near-symmetric residuals).
    online_epochs : int, optional
        Book Ch. 5 online-learning fine-tune epochs on the most recent samples.
        Default 0 (off).

    Returns
    -------
    pd.DataFrame
        Columns ['datetime', 'predicted_price', 'lower_bound', 'upper_bound',
        'volatility']. ``attrs`` carries ``risk_log_returns`` (the empirical,
        fat-tailed horizon-end return sample the risk panel reads),
        ``downside_share``, and ``vol_model`` diagnostics.
    """
    interval = normalize_interval(interval)
    print(f"--- Starting Vol-LSTM (volatility) Prediction ({interval}) ---")

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    coerce_numeric_columns(df)

    max_obs = 1500 if interval == 'daily' else 1000
    prices = df['adj_close'].dropna().values.astype(np.float64)
    prices = prices[prices > 0]
    if len(prices) > max_obs:
        prices = prices[-max_obs:]

    last_price = float(prices[-1])
    last_date = df.index[-1]
    future_dates = generate_future_dates(last_date, horizon, interval)

    returns = np.diff(np.log(prices))
    lags = _har_lags(interval)
    look_back = _LOOK_BACK[interval]

    step_var, mult, signed_resid, diag = _forecast_cone(
        prices, returns, lags, look_back, horizon, adaptive, aci_gamma,
        online_epochs,
    )

    # Asymmetric edges (optional): per-side multipliers from the signed residuals.
    k_lo = k_up = None
    if asymmetric and signed_resid:
        k_lo, k_up = _asymmetric_multipliers(
            signed_resid, horizon, adaptive=adaptive, gamma=aci_gamma)

    # Variance is additive across (assumed-uncorrelated) steps.
    cum_sigma = np.sqrt(np.cumsum(step_var))

    rows = []
    for i in range(horizon):
        sig = float(cum_sigma[i])
        kl, ku = ((float(k_lo[i]), float(k_up[i])) if k_lo is not None
                  else (float(mult[i]), float(mult[i])))
        rows.append({
            'datetime': future_dates[i],
            'predicted_price': last_price,                       # random walk
            'lower_bound': float(last_price * np.exp(-kl * sig)),
            'upper_bound': float(last_price * np.exp(ku * sig)),
            'volatility': float(np.sqrt(step_var[i])),           # per-step vol
        })

    out = pd.DataFrame(rows)

    # Empirical horizon-end return sample for the risk panel (fat tails kept).
    horizon_resid = signed_resid.get(horizon) if signed_resid else None
    if horizon_resid is not None and len(horizon_resid) >= _MIN_CAL:
        out.attrs['risk_log_returns'] = (
            np.asarray(horizon_resid, dtype=np.float64) * float(cum_sigma[-1]))
        out.attrs['risk_source'] = 'Vol-LSTM conformal residuals (empirical)'

    rs_down, rs_up = _realized_semivariance(returns[-min(len(returns), 60):])
    denom = float(rs_down.sum() + rs_up.sum())
    if denom > 0:
        out.attrs['downside_share'] = round(float(rs_down.sum()) / denom, 3)
    if k_lo is not None:
        out.attrs['cone'] = 'asymmetric'
        out.attrs['cone_k'] = {'lower': float(k_lo[-1]), 'upper': float(k_up[-1])}
    out.attrs['vol_model'] = diag
    if isinstance(diag, dict) and diag.get('architecture'):
        out.attrs['architecture'] = diag['architecture']

    print("Vol-LSTM Prediction Complete.")
    return out


def _guard_reason(step_var, mult, returns):
    """Why a fitted cone is out of range (a short string), or None when it is sane."""
    sd = np.sqrt(np.asarray(step_var, dtype=np.float64))
    k = np.asarray(mult, dtype=np.float64)
    if not (np.all(np.isfinite(sd)) and np.all(np.isfinite(k))):
        return "non-finite forecast"
    recent = np.asarray(returns, dtype=np.float64)[-_GUARD_WINDOW:]
    ref = float(np.std(recent, ddof=1)) if len(recent) > 2 else 0.0
    if ref > 0:
        hi, lo = float(sd.max()) / ref, float(sd.min()) / ref
        if hi > _GUARD_RATIO:
            return f"forecast volatility {hi:.1f}x the last {_GUARD_WINDOW} sessions'"
        if lo < 1.0 / _GUARD_RATIO:
            return f"forecast volatility {lo:.2f}x the last {_GUARD_WINDOW} sessions'"
    if float(k.max()) > _GUARD_K:
        return f"conformal multiplier {float(k.max()):.1f} (normal 1.96)"
    return None


def _forecast_cone(prices, returns, lags, look_back, horizon, adaptive,
                   aci_gamma, online_epochs):
    """
    Fit the network, forecast per-step variance, and conformalize the cone.

    Split out from the orchestrator so the (long) statistical bookkeeping is
    readable and so ``predict_step_variance`` below can reuse the same fit for
    the model-comparison harness.

    Returns
    -------
    (np.ndarray, np.ndarray, dict, dict)
        ``(step_var, multipliers, signed_residuals, diagnostics)``.
    """
    n_rv = len(returns)
    min_needed = look_back + horizon + _MIN_CAL + 40
    if n_rv < max(_MIN_ROWS, min_needed):
        print(f"Vol-LSTM: only {n_rv} return bars (< {max(_MIN_ROWS, min_needed)}) "
              "— using the EWMA volatility fallback.")
        return (_fallback_variance(returns, horizon),
                np.full(horizon, _Z95), {},
                {'fitted': False, 'reason': 'insufficient history',
                 'n_bars': int(n_rv)})

    channels, log_rv = build_rv_channels(returns, lags)

    # ── Debias the log -> level transform ───────────────────────────────────
    # The network predicts the mean of log(r^2). Exponentiating that mean does
    # NOT give the conditional variance: with r = sigma * z, we have
    # log(r^2) = log(sigma^2) + log(z^2), and E[log z^2] = -1.27 for a Gaussian z
    # (the mean of a log-chi-squared(1)). So exp(prediction) understates sigma^2
    # by a factor of ~e^-1.27 ~ 0.28, and the reported volatility would be badly
    # biased low. The classic Jensen fix `exp(mu + resid_var/2)` is *also* wrong
    # here, because the residual is log-chi-squared, not Gaussian — with a
    # residual variance near pi^2/2 ~ 4.93 it over-corrects by an order of
    # magnitude in variance terms.
    #
    # Instead the offset is estimated from the data: the constant `c` that makes
    # the level forecast match the realised mean variance,
    # ``c = log(mean(RV)) - mean(log RV)``. That reproduces the theoretical 1.27
    # for Gaussian returns and self-adjusts for the percentile clipping applied
    # above and for genuinely fat-tailed returns. Fitted on the training block
    # only, so it carries no lookahead.
    #
    # This is not a hypothetical: ``run_vol_comparison.py`` measured HAR-RV --
    # which uses the Gaussian-Jensen form -- over-stating volatility by ~1.9x on
    # RELIANCE (bias_ratio 1.92 vs 1.25 here), with 2.5x the cumulative squared
    # error. The conformal layer hides it (an inflated sigma is cancelled by a
    # deflated multiplier, so coverage stays fine) which is exactly why the
    # reported per-step volatility needs its own check. See the README section
    # "Volatility model comparison -- and a latent bias it exposed in HAR-RV".
    rv_all = np.asarray(returns, dtype=np.float64) ** 2

    # Every usable RV origin: needs a full look-back behind and `horizon` ahead.
    all_origins = list(range(look_back - 1, n_rv - horizon))
    # Calibration origins additionally need `horizon` future PRICES beyond their
    # own price index (o + 1), for the standardized-residual computation.
    cal_size = max(_MIN_CAL, int(round(len(all_origins) * _CAL_FRAC)))
    cal_size = min(cal_size, len(all_origins) - 40)
    # Embargo `horizon` origins between train and calibration so no training
    # label window overlaps a calibration label window (a purged split).
    embargo = max(0, min(horizon, len(all_origins) - cal_size - 40))
    train_origins = all_origins[:len(all_origins) - cal_size - embargo]
    cal_origins = all_origins[-cal_size:]

    X_tr, Y_tr = _make_samples(channels, log_rv, look_back, horizon, train_origins)
    mu, sd = _standardize(X_tr)
    X_tr = (X_tr - mu) / sd

    model = train_vol_lstm(X_tr, Y_tr, online_epochs=online_epochs)

    # ── Level offset, calibrated against the MODEL's own predictions ────────
    # Two candidate offsets, and the difference matters:
    #
    #   (a) unconditional: log(mean RV) - mean(log RV) on the training block.
    #       This corrects the log-chi-squared transform but NOT the network's own
    #       conditional bias, so a model that systematically over-forecasts stays
    #       over-forecasting. Measured bias ratio with (a): ~1.25.
    #   (b) model-calibrated: log(mean RV_cal) - mean(predicted log RV_cal) on
    #       the held-out calibration block. This absorbs the transform AND the
    #       network's own bias in one scalar, because it asks 'what constant makes
    #       THIS model's forecasts land on the realised level?'.
    #
    # (b) is used whenever a calibration block exists, with (a) as the fallback.
    # Fitting one scalar on the calibration block is the same thing the conformal
    # layer already does with its per-horizon multiplier, and the block is held
    # out of training either way, so this adds no lookahead — but it is a second
    # use of that block, which is why it is one scalar and not a fitted curve.
    train_end = train_origins[-1] + 1
    offset = float(np.log(max(float(np.mean(rv_all[:train_end])), _EPS))
                   - float(np.mean(log_rv[:train_end])))
    offset_mode = 'unconditional'

    usable = [o for o in cal_origins if o + 1 + horizon <= len(prices) - 1]
    if usable:
        Xc = (np.stack([channels[o - look_back + 1:o + 1] for o in usable])
              - mu) / sd
        Yc = np.stack([log_rv[o + 1:o + horizon + 1] for o in usable])
        Pc = np.asarray(model.predict(Xc, verbose=0))
        cal_offset = float(np.log(max(float(np.mean(np.exp(Yc))), _EPS))
                           - float(np.mean(Pc)))
        if np.isfinite(cal_offset):
            offset, offset_mode = cal_offset, 'model-calibrated'

    def _step_var_at(origins):
        """Per-step variance forecasts for a batch of origins (debiased level)."""
        Xb = (np.stack([channels[o - look_back + 1:o + 1] for o in origins])
              - mu) / sd
        logv = np.asarray(model.predict(Xb, verbose=0))
        return np.exp(logv + offset)

    # ── Conformal calibration on the held-out recent block ──────────────────
    resid = {h: [] for h in range(1, horizon + 1)}
    signed = {h: [] for h in range(1, horizon + 1)}
    if usable:
        sv = _step_var_at(usable)
        cum_sigma = np.sqrt(np.cumsum(sv, axis=1))
        for row, o in enumerate(usable):
            p_idx = o + 1                       # price index of the RV origin
            for h in range(1, horizon + 1):
                s = float(cum_sigma[row, h - 1])
                if s > 0:
                    cr = float(np.log(prices[p_idx + h] / prices[p_idx]))
                    resid[h].append(abs(cr) / s)
                    signed[h].append(cr / s)

    mult = np.full(horizon, _Z95)
    for h in range(1, horizon + 1):
        z = resid[h]
        m = len(z)
        if m >= _MIN_CAL:
            q_level = min(1.0, np.ceil((m + 1) * (1 - 0.05)) / m)
            mult[h - 1] = float(np.quantile(z, q_level))
    signed = {h: np.asarray(v, dtype=np.float64) for h, v in signed.items()}

    mode = "static"
    if adaptive and any(len(v) >= _MIN_CAL for v in signed.values()):
        mult = _aci_multipliers(signed, horizon, gamma=aci_gamma)
        mode = "ACI-adaptive"

    # ── Serving forecast: the most recent origin ────────────────────────────
    step_var = _step_var_at([n_rv - 1])[0]

    print(f"Vol-LSTM fit ({len(train_origins)} train / {len(usable)} cal windows); "
          f"next-step vol {np.sqrt(step_var[0]):.4%}; {mode} conformal k(h) "
          f"{np.round(mult, 2)} (normal=1.96)")

    diag = {
        'fitted': True,
        'mode': mode,
        'look_back': int(look_back),
        'train_windows': int(len(train_origins)),
        'cal_windows': int(len(usable)),
        'embargo': int(embargo),
        'level_offset': round(offset, 4),
        'level_offset_mode': offset_mode,
        'next_step_vol': float(np.sqrt(step_var[0])),
        'online_epochs': int(online_epochs),
    }
    # The trained Keras object exists only inside this function, so the
    # architecture is captured here and travels out on ``diag``. Recording it
    # where the model is NOT in scope would have produced a silent no-op.
    try:
        import architectures as _arch
        diag['architecture'] = _arch.describe(model, 'Vol-LSTM')
    except Exception:
        pass
    reason = _guard_reason(step_var, mult, returns)
    if reason:
        print(f"Vol-LSTM: {reason}; using the EWMA volatility fallback.")
        diag.update(mode='EWMA fallback (guard)', guard=reason)
        return _fallback_variance(returns, horizon), np.full(horizon, _Z95), {}, diag
    return step_var, mult, signed, diag


# ==========================================
# 5. Model comparison (book Ch. 5: cumulative squared error)
# ==========================================
def cumulative_squared_error(actual_rv, predicted_rv):
    """
    Running sum of squared variance-forecast errors, as the book plots it.

    A single aggregate score hides *when* a volatility model earns or loses its
    edge; the cumulative curve makes it visible — a flat stretch means the models
    tie, and a step means one of them mispriced a specific shock. Comparing two
    curves is how the book decides between LSTM, RNN and GARCH.

    Parameters
    ----------
    actual_rv, predicted_rv : array-like
        Realised and forecast variance, aligned element-wise.

    Returns
    -------
    np.ndarray
        Cumulative squared error, same length as the inputs.
    """
    a = np.asarray(actual_rv, dtype=np.float64)
    p = np.asarray(predicted_rv, dtype=np.float64)
    n = min(len(a), len(p))
    return np.cumsum((a[:n] - p[:n]) ** 2)


def qlike_loss(actual_rv, predicted_rv):
    """
    QLIKE loss ``sum(RV/sigma^2 + log sigma^2)`` — the standard variance score.

    Squared error on *variance* is dominated by the largest observations and is
    not robust when the variance proxy is noisy (``r^2`` is an extremely noisy
    proxy for the day's true variance). QLIKE is the loss function shown by
    Patton (2011) to be robust to that proxy noise, so it is the fairer of the
    two scores for ranking volatility models — reported alongside the book's
    cumulative squared error rather than instead of it.

    Returns
    -------
    float
        Mean QLIKE loss (lower is better); NaN if no valid pairs.
    """
    a = np.asarray(actual_rv, dtype=np.float64)
    p = np.asarray(predicted_rv, dtype=np.float64)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
    mask = np.isfinite(a) & np.isfinite(p) & (p > 0)
    if not mask.any():
        return float('nan')
    return float(np.mean(a[mask] / p[mask] + np.log(p[mask])))
```
