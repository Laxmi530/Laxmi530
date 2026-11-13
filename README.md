- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Direct multi-horizon forecasting engine for the tree / ML models.

The walk-forward backtest showed the gradient-boosted models (XGBoost,
LightGBM, Quantile-LGBM, stacking Ensemble) were the *worst* performers —
well below the naive random walk — because they forecast **recursively**:
they predict one step, feed the prediction back in, recompute features, and
repeat. Two problems compound across the horizon:

1. **Error accumulation / drift** — each step's error feeds the next, and a
   constant-volume assumption biases the volume features.
2. **A one-step origin offset** — ``create_features`` drops the most recent
   row (its next-step target is unknown), so the recursive path started its
   forecast one observation early.

This module replaces recursion with the **direct strategy**: for each horizon
step ``h`` it trains a model on the *cumulative* log return ``log(P[t+h]/P[t])``
and forecasts every step straight from the **observed** features at the true
last timestamp (the origin). No prediction is ever fed back, so errors do not
compound, and the origin is the real last bar.

The per-horizon training reuses each model family's existing trainer (they are
already parameterised by ``target_col``); this module only supplies the
multi-horizon targets, the clean feature-column list, and the origin feature
row, then compounds each predicted cumulative return onto the last price.
"""

import numpy as np
import pandas as pd

# Columns that must never be used as model inputs.
# 'mkt_index' is the market level used to build excess-return targets; it is a
# target ingredient, not a feature, so it is excluded here too.
_RAW_COLS = {'open', 'high', 'low', 'close', 'adj_close', 'volume', 'mkt_index'}
_TARGET_PREFIX = 'target_h'


def horizon_feature_cols(features_df):
    """
    Feature columns for direct models: everything except raw OHLCV and targets.

    Excludes the raw price columns, the recursive ``target_log_return``, and any
    ``target_h*`` multi-horizon target columns, so no future information leaks
    in as a feature.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of ``create_features`` (optionally augmented with target_h*).

    Returns
    -------
    list of str
        Ordered feature column names.
    """
    return [
        c for c in features_df.columns
        if c not in _RAW_COLS
        and c != 'target_log_return'
        and not c.startswith(_TARGET_PREFIX)
    ]


def make_horizon_targets(features_df, horizon):
    """
    Build the cumulative-log-return target for each horizon step.

    ``target_h{h}`` at time t is ``log(adj_close[t+h] / adj_close[t])`` — the
    total log return from the origin to step h. Forecasting these directly (one
    model per h) avoids recursive error accumulation; the trailing ``h`` rows
    are NaN (no realised future) and are dropped per-horizon at train time.

    **Excess-return reframing.** If the frame carries an ``mkt_index`` column
    (attached by :func:`market_context.add_market_index`), the target becomes
    the *excess* cumulative return ``log(P[t+h]/P[t]) - log(M[t+h]/M[t])`` — the
    stock-specific component with the broad market move removed. Reconstruction
    is unchanged because the expected future market return is ~0, so the caller
    still maps the prediction back with ``last_price * exp(prediction)``.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature frame containing 'adj_close' (and optionally 'mkt_index').
    horizon : int
        Number of steps to forecast.

    Returns
    -------
    dict[int, pd.Series]
        Mapping h -> cumulative log-return target series (aligned to features_df).
        Excess returns when 'mkt_index' is present, absolute returns otherwise.
    """
    price = features_df['adj_close']
    targets = {h: np.log(price.shift(-h) / price) for h in range(1, horizon + 1)}
    if 'mkt_index' in features_df.columns:
        m = features_df['mkt_index']
        mkt_ret = {h: np.log(m.shift(-h) / m) for h in range(1, horizon + 1)}
        targets = {h: targets[h] - mkt_ret[h] for h in range(1, horizon + 1)}
    return targets


def training_frame(features_df, feature_cols, target_series, target_name):
    """
    Assemble a clean training frame of features + a single target.

    Contains only ``feature_cols`` plus the named target, with rows whose target
    is NaN (the trailing horizon rows) removed. Because no raw or other-target
    columns are present, the model trainers' internal column selection yields
    exactly ``feature_cols``.

    Parameters
    ----------
    features_df : pd.DataFrame
        Engineered feature frame.
    feature_cols : list of str
        Columns to use as inputs.
    target_series : pd.Series
        Target aligned to ``features_df`` (positionally).
    target_name : str
        Name to give the target column in the returned frame.

    Returns
    -------
    pd.DataFrame
        Frame with ``feature_cols + [target_name]`` and no NaN target rows.
    """
    frame = features_df[feature_cols].copy()
    frame[target_name] = np.asarray(target_series)
    return frame[frame[target_name].notna()]


def conformal_calibration(train_df, feature_cols, target_name, fit_predict_fn,
                          cal_frac=0.2, min_cal=20, embargo=0):
    """
    The calibration block of :func:`conformal_halfwidth`: its realised targets
    and the calibration model's predictions for them, or None when there is too
    little data or the fit fails. Same split, purge and fit as the half-width.
    """
    n = len(train_df)
    cal_size = max(min_cal, int(round(n * cal_frac)))
    if cal_size < 1 or n - cal_size < min_cal:
        return None
    cal = train_df.iloc[-cal_size:]
    embargo = max(0, min(embargo, n - cal_size - min_cal))
    proper = train_df.iloc[:n - cal_size - embargo]
    try:
        preds = np.asarray(fit_predict_fn(proper, cal[feature_cols]), dtype=float)
    except Exception:
        return None
    y = np.asarray(cal[target_name].values, dtype=float)
    if len(y) == 0:
        return None
    return y, preds


def conformal_quantile(resid, alpha=0.05):
    """Finite-sample split-conformal (1 - alpha) quantile of absolute residuals."""
    resid = np.abs(np.asarray(resid, dtype=float))
    m = len(resid)
    q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
    return float(np.quantile(resid, q_level))


def conformal_halfwidth(train_df, feature_cols, target_name, fit_predict_fn,
                        alpha=0.05, cal_frac=0.2, min_cal=20, embargo=0):
    """
    Horizon-wise split-conformal half-width for a direct cumulative-return model.

    Split-conformal prediction gives distribution-free interval guarantees by
    measuring a model's errors on a held-out calibration block. Here, for one
    horizon, the calibration model is fit on the earlier part of the training
    rows and scored on the most recent ``cal_frac`` block; the (1 - alpha)
    empirical quantile of the absolute calibration residuals (with the standard
    finite-sample correction) is the interval half-width, expressed in
    cumulative-log-return space.

    Because the data is a time series (exchangeability is violated), the split
    is temporal — calibration uses the most recent block — and the half-width
    is computed **per horizon**, so intervals widen naturally with horizon. The
    coverage guarantee is therefore approximate but empirically checkable (the
    backtest reports realised coverage).

    The target ``log(P[t+h] / P[t])`` spans ``h`` future bars, so the last
    ``h`` proper-train rows have label windows that overlap the calibration
    block. Passing ``embargo=h`` drops those rows (a purged split), removing the
    overlap so the calibration residuals — and hence the interval width — are
    not optimistically correlated with the fitted labels.

    Parameters
    ----------
    train_df : pd.DataFrame
        Per-horizon training frame (feature_cols + target_name), no NaN target.
    feature_cols : list of str
        Feature column names.
    target_name : str
        Cumulative-log-return target column name.
    fit_predict_fn : callable
        ``fit_predict_fn(proper_df, X_cal_df) -> np.ndarray`` — trains on the
        proper-train rows and predicts the calibration features.
    alpha : float, optional
        Miscoverage level; 0.05 -> 95% intervals. Default 0.05.
    cal_frac : float, optional
        Fraction of the most recent rows used for calibration. Default 0.2.
    min_cal : int, optional
        Minimum calibration rows; below this the calibration is skipped.
    embargo : int, optional
        Number of rows to drop between the proper-train and calibration blocks
        to purge overlapping label windows. Set to the horizon ``h`` of
        ``target_name``. Default 0. Capped so the proper block keeps >=min_cal.

    Returns
    -------
    float or None
        Half-width in log-return space, or None if there is too little data to
        calibrate (the caller should then fall back to a simple estimate).
    """
    n = len(train_df)
    cal_size = max(min_cal, int(round(n * cal_frac)))
    if cal_size < 1 or n - cal_size < min_cal:
        return None

    cal = train_df.iloc[-cal_size:]
    # Purge `embargo` rows between proper-train and calibration (capped so the
    # proper block still has at least min_cal rows to fit on).
    embargo = max(0, min(embargo, n - cal_size - min_cal))
    proper = train_df.iloc[:n - cal_size - embargo]

    try:
        preds = np.asarray(fit_predict_fn(proper, cal[feature_cols]))
    except Exception:
        return None

    resid = np.abs(np.asarray(cal[target_name].values, dtype=float) - preds)
    m = len(resid)
    if m == 0:
        return None
    # Finite-sample conformal quantile level.
    q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
    return float(np.quantile(resid, q_level))


def compute_origin_row(df, feature_cols, interval):
    """
    Compute the forecast-origin feature row at the true last observed bar.

    Uses the buffer-based ``_features_from_buffers`` (the same definitions as
    ``create_features``) on the full observed history, so the origin features
    correspond to the real last timestamp rather than the row ``create_features``
    drops. Returns the feature row, the last observed price, and the last date.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (already max-obs-capped) DataFrame with datetime + OHLCV.
    feature_cols : list of str
        Feature columns the models expect, in order.
    interval : str
        'daily' or 'hourly'.

    Returns
    -------
    X : pd.DataFrame
        Single-row feature frame reindexed to ``feature_cols``.
    last_price : float
        Last observed adjusted close.
    last_date : pd.Timestamp
        Last observed datetime (the forecast origin).
    """
    # Lazy import avoids a circular dependency (ML_model imports this module).
    from ML_model import _features_from_buffers, _BUFFER_SIZE

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    prices = pd.to_numeric(df['adj_close'], errors='coerce').values.astype(float)
    volumes = pd.to_numeric(df['volume'], errors='coerce').values.astype(float)

    last_date = df['datetime'].iloc[-1]
    last_price = float(prices[-1])

    # Pass the raw bars when training had them, so ATR and range_pct at the
    # origin are true range like every training row, not close-to-close
    # proxies (a ~46% skew in atr_norm before this; README §55).
    ohlc = {}
    if all(c in df.columns for c in ('high', 'low', 'close')):
        ohlc = {k: pd.to_numeric(df[c], errors='coerce').values.astype(float)[-_BUFFER_SIZE:]
                for k, c in (('highs', 'high'), ('lows', 'low'), ('closes', 'close'))}
    feat = _features_from_buffers(
        prices[-_BUFFER_SIZE:], volumes[-_BUFFER_SIZE:], last_date, interval,
        **ohlc,
    )
    # Float throughout: an int fill makes the filled columns int64, and pandas 3
    # refuses to write the context overlay's floats into them (with market
    # context on, every ML model raised "Invalid value ... for dtype 'int64'";
    # README section 68).
    X = pd.DataFrame([feat]).reindex(columns=feature_cols, fill_value=0.0).astype(float)

    # Overlay any context features (e.g. market/VIX columns from
    # market_context) that are present on the df but not derivable from the
    # price/volume buffers. Without this the origin row would carry 0 for those
    # features while training rows carry their real values — a train/serve skew.
    for col in feature_cols:
        if col not in feat and col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").iloc[-1]
            X.at[0, col] = 0.0 if pd.isna(val) else float(val)

    return X, last_price, last_date
```
