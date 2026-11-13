- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Shared utilities for the stock-prediction models.

This module collects the logic that was previously duplicated across the
individual ``*_model.py`` files so each model can import a single, canonical
implementation instead of carrying its own copy. It covers:

- interval normalisation and OHLCV cleaning,
- scale-invariant feature engineering (Wilder RSI, the 9-feature matrix),
- NSE-exchange-calendar-aware future date generation (holiday + session-aware),
- statistical helpers (winsorisation, GARCH selection, Monte Carlo
  price-path simulation), and
- Keras training helpers (seed reset, temporal split, standard callbacks,
  sequence preparation, recursive forecasting).

Heavy / optional dependencies (``arch``, ``scikit-learn``, ``tensorflow``)
are imported lazily inside the functions that need them, so importing this
module from a purely statistical model does not drag TensorFlow into memory.
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

# ==========================================
# Shared constants
# ==========================================
OHLCV_COLS = ['open', 'high', 'low', 'close', 'adj_close', 'volume']

_SMA_SHORT = 10
_SMA_LONG = 30
_RSI_PERIOD = 14

N_SIMS = 5000

#: Thread settings of the tree models (LightGBM and XGBoost in ML_model,
#: Quantile_model and Ensemble_model; README section 112). One thread each:
#: their training sets are about 1,000 rows, where many threads cost more than
#: they save (LightGBM's parameter guide: do not use many threads on a small
#: dataset), and a fixed thread count makes a forecast the same on any machine.
#: LightGBM's ``deterministic`` and ``force_col_wise`` make it stable as well.
#: Before section 112 every tree model used ``n_jobs=-1`` (all processors).
LGB_THREADS = {'n_jobs': 1, 'deterministic': True, 'force_col_wise': True}
XGB_THREADS = {'n_jobs': 1}
LEGACY_THREADS = {'n_jobs': -1}


def _threads(new):
    # STOCKAPP_TREE_THREADS=legacy restores n_jobs=-1, for section 112's check.
    import os
    return dict(LEGACY_THREADS if os.environ.get('STOCKAPP_TREE_THREADS') == 'legacy' else new)


def lgb_threads():
    """Thread keyword arguments for a LightGBM model (see LGB_THREADS)."""
    return _threads(LGB_THREADS)


def xgb_threads():
    """Thread keyword arguments for an XGBoost model (see XGB_THREADS)."""
    return _threads(XGB_THREADS)


# ==========================================
# Interval / preprocessing helpers
# ==========================================
def normalize_interval(interval):
    """
    Normalise a free-form interval string to 'daily' or 'hourly'.

    Parameters
    ----------
    interval : str
        Any of '1d' / '1D' / 'daily' / '1 Day' (treated as daily); anything
        else is treated as hourly.

    Returns
    -------
    str
        'daily' or 'hourly'.
    """
    return 'daily' if interval in ('1d', '1D', 'daily', '1 Day') else 'hourly'


def coerce_numeric_columns(df, cols=OHLCV_COLS):
    """
    Coerce the given OHLCV columns to numeric in place (errors -> NaN).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clean (modified in place).
    cols : iterable of str, optional
        Column names to coerce. Defaults to the standard OHLCV columns.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, for chaining.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def prepare_log_returns(df, max_obs=None):
    """
    Build a datetime-indexed log-return series from raw stock data.

    Converts the datetime column to a sorted index, coerces OHLCV to numeric,
    computes log returns from ``adj_close``, drops NaN/Inf returns, and
    optionally keeps only the most recent ``max_obs`` observations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    max_obs : int, optional
        Maximum number of recent observations to keep. None keeps all.

    Returns
    -------
    pd.DataFrame
        DateTime-indexed DataFrame with adj_close and a 'log_return' column.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    coerce_numeric_columns(df)

    df['log_return'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df = df.dropna(subset=['log_return'])
    df = df[np.isfinite(df['log_return'])]

    if max_obs and len(df) > max_obs:
        df = df.iloc[-max_obs:]

    return df


# ==========================================
# Feature engineering
# ==========================================
def winsorize_bounds(arr, limits=(0.01, 0.01)):
    """
    Compute the (lower, upper) winsorization bounds for an array.

    Exposed separately so the bounds can be **fit on training data only** and
    then reused to clip a validation/test/calibration block, keeping the
    preprocessing free of look-ahead bias (see ``winsorize``'s ``bounds`` arg).

    Parameters
    ----------
    arr : array-like
        Values to derive the percentile bounds from (the training block).
    limits : tuple of float, optional
        (lower, upper) percentile limits as fractions in (0, 1). Default
        (0.01, 0.01) -> 1st and 99th percentiles.

    Returns
    -------
    tuple of float
        (lower, upper) clip bounds.
    """
    lower = np.percentile(arr, limits[0] * 100)
    upper = np.percentile(arr, 100 - limits[1] * 100)
    return float(lower), float(upper)


def winsorize(arr, limits=(0.01, 0.01), bounds=None):
    """
    Clip array values at the given percentiles to reduce outlier impact.

    By default the clip bounds are computed from ``arr`` itself. When the array
    being clipped is purely training data (as in every current caller — the
    backtest harness slices the training window before the model sees it) that
    is leak-free. If you ever winsorize a series that spans a train/test split,
    fit the bounds on the training block with ``winsorize_bounds`` and pass them
    via ``bounds`` so future values never influence the clipping.

    Parameters
    ----------
    arr : array-like
        Input array of values to winsorize.
    limits : tuple of float, optional
        Tuple of (lower, upper) percentile limits as fractions in (0, 1).
        Default (0.01, 0.01) clips at the 1st and 99th percentiles. Ignored
        when ``bounds`` is supplied.
    bounds : tuple of float, optional
        Pre-computed (lower, upper) clip bounds (e.g. from ``winsorize_bounds``
        fit on the training block). When given, ``limits`` is not used.

    Returns
    -------
    np.ndarray
        Array with values clipped to the specified percentile bounds.
    """
    if bounds is None:
        lower, upper = winsorize_bounds(arr, limits)
    else:
        lower, upper = bounds
    return np.clip(arr, lower, upper)


def wilder_rsi(prices, period=_RSI_PERIOD, fillna=None):
    """
    Compute Relative Strength Index using Wilder's exponential smoothing.

    Uses EMA with alpha = 1/period for the gain and loss averages, matching
    the RSI definition used in professional technical-analysis platforms.

    Parameters
    ----------
    prices : array-like or pd.Series
        Price series (typically adj_close).
    period : int, optional
        RSI lookback period. Default is 14.
    fillna : float, optional
        If given, NaN values (during the warmup window) are replaced with
        this value (e.g. 50.0 for a neutral RSI). If None, NaNs are kept.

    Returns
    -------
    np.ndarray
        RSI values (0-100) for each observation.
    """
    s = pd.Series(prices, dtype=np.float64)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if fillna is not None:
        rsi = rsi.fillna(fillna)
    return rsi.values


def compute_price_features(prices):
    """
    Derive a multivariate feature matrix from a 1-D price array.

    Every feature is computable from prices alone, so the function can be
    called during recursive prediction using predicted prices.

    Columns (9 total):
    0. adj_close — raw price level
    1. log_return — single-period log return
    2. sma_ratio — SMA(10) / SMA(30), trend direction
    3. volatility — rolling 10-period std of log returns
    4. vol_long — rolling 30-period std of log returns (regime)
    5. rsi — Relative Strength Index (Wilder, period 14)
    6. momentum_5 — 5-period log return
    7. zscore — (price - SMA30) / rolling_std(price, 30), mean-reversion
    8. acceleration — diff of log returns, trend-change signal

    Parameters
    ----------
    prices : array-like
        1-D array of price values (e.g., adj_close).

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, 9) with dtype float32.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)

    log_ret = np.zeros(n)
    log_ret[1:] = np.log(prices[1:] / prices[:-1])

    sma_s = pd.Series(prices).rolling(_SMA_SHORT, min_periods=1).mean().values
    sma_l = pd.Series(prices).rolling(_SMA_LONG, min_periods=1).mean().values
    sma_ratio = sma_s / np.where(sma_l == 0, 1.0, sma_l)

    vol = pd.Series(log_ret).rolling(
        _SMA_SHORT, min_periods=1,
    ).std().fillna(0).values

    vol_long = pd.Series(log_ret).rolling(
        _SMA_LONG, min_periods=1,
    ).std().fillna(0).values

    rsi = wilder_rsi(prices, fillna=50.0)

    momentum_5 = np.zeros(n)
    momentum_5[5:] = np.log(prices[5:] / prices[:-5])

    price_std = pd.Series(prices).rolling(
        _SMA_LONG, min_periods=1,
    ).std().fillna(1e-10).values
    price_std = np.where(price_std == 0, 1e-10, price_std)
    zscore = (prices - sma_l) / price_std

    acceleration = np.zeros(n)
    acceleration[2:] = log_ret[2:] - log_ret[1:-1]

    return np.column_stack([
        prices, log_ret, sma_ratio, vol, vol_long, rsi, momentum_5,
        zscore, acceleration,
    ]).astype(np.float32)


def inverse_scaled_target(scaler, scaled_value, n_features):
    """
    Inverse-transform a single scaled adj_close value (column 0).

    Constructs a dummy row with zeros for non-target columns and applies the
    fitted scaler's inverse transform to recover the price in its original
    scale.

    Parameters
    ----------
    scaler : MinMaxScaler
        Fitted scaler used to transform the features.
    scaled_value : float
        Scaled adj_close value to inverse-transform.
    n_features : int
        Number of features in the original feature matrix.

    Returns
    -------
    float
        Price in original scale.
    """
    row = np.zeros((1, n_features))
    row[0, 0] = scaled_value
    return scaler.inverse_transform(row)[0, 0]


# ==========================================
# Future date generation (trading calendar)
# ==========================================
# NSE hourly bar start times (Asia/Kolkata). yfinance returns exactly these 7
# bars per session for the 09:15-15:30 trading window (the last is a 15-min
# stub). Hard-coded because it matches the data convention the models train on.
_NSE_HOURLY_BAR_TIMES = [
    (9, 15), (10, 15), (11, 15), (12, 15), (13, 15), (14, 15), (15, 15),
]
NSE_BARS_PER_SESSION = len(_NSE_HOURLY_BAR_TIMES)   # 7
_NSE_CAL = None   # cached pandas_market_calendars calendar


def _nse_calendar():
    """Return a cached NSE trading calendar (pandas_market_calendars)."""
    global _NSE_CAL
    if _NSE_CAL is None:
        import pandas_market_calendars as mcal
        _NSE_CAL = mcal.get_calendar('NSE')
    return _NSE_CAL


def _future_sessions(last_date, n_sessions):
    """
    The next ``n_sessions`` NSE trading dates strictly after ``last_date``.

    Uses the exchange calendar so weekends AND holidays (Diwali, Independence
    Day, etc.) are skipped. Returns tz-naive normalised dates.
    """
    last_day = pd.Timestamp(last_date).normalize()
    # Generous window so even a holiday-dense stretch yields enough sessions;
    # widen and retry if needed.
    span = n_sessions * 3 + 45
    for _ in range(4):
        sched = _nse_calendar().schedule(
            start_date=last_day,
            end_date=last_day + pd.Timedelta(days=span),
        )
        sessions = [pd.Timestamp(d).normalize() for d in sched.index
                    if pd.Timestamp(d).normalize() > last_day]
        if len(sessions) >= n_sessions:
            return sessions[:n_sessions]
        span *= 2
    return sessions[:n_sessions]


def widen_to_monotone(pred_df, price_col="predicted_price", lower_col="lower_bound",
                      upper_col="upper_bound"):
    """
    Make a forecast's 95% band widen (or hold) with the horizon, never narrow.

    Each side's distance from the forecast, ``log(upper / price)`` and
    ``log(price / lower)``, is replaced by its running maximum over the steps.
    A band calibrated separately at each step can dip from sampling noise
    (HAR-RV's per-step multipliers read 1.64, 1.69, 1.13, 1.50, 1.25 in one
    run), and a 3-day range narrower than a 2-day one has no meaning for a
    cumulative price. Only widens, so coverage cannot fall (README section 70).
    Changes ``pred_df`` in place (keeping its ``attrs``) and returns it; a
    frame without both bounds is returned unchanged.
    """
    if pred_df is None or not {price_col, lower_col, upper_col} <= set(pred_df.columns):
        return pred_df
    p = pd.to_numeric(pred_df[price_col], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(pred_df[lower_col], errors="coerce").to_numpy(float)
    up = pd.to_numeric(pred_df[upper_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(p) & np.isfinite(lo) & np.isfinite(up) & (p > 0) & (lo > 0) & (up > 0)
    if not ok.all():
        return pred_df
    raw_up, raw_lo = np.log(up / p), np.log(p / lo)
    d_up = np.maximum.accumulate(np.maximum(raw_up, 0.0))
    d_lo = np.maximum.accumulate(np.maximum(raw_lo, 0.0))
    # Only the steps that narrowed change; every other value is kept exactly.
    pred_df[upper_col] = np.where(d_up > raw_up + 1e-12, p * np.exp(d_up), up)
    pred_df[lower_col] = np.where(d_lo > raw_lo + 1e-12, p * np.exp(-d_lo), lo)
    return pred_df


def _row_quantile(sorted_rows, q):
    """``np.quantile(row, q[i])`` for each row of a row-sorted 2-D array (linear method)."""
    m = sorted_rows.shape[1]
    pos = np.clip(q, 0.0, 1.0) * (m - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, m - 1)
    frac = pos - lo
    r = np.arange(sorted_rows.shape[0])
    a, b = sorted_rows[r, lo], sorted_rows[r, hi]
    return a + frac * (b - a)


def aci_quantile(scores, alpha=0.05, gamma=0.05, alpha_lo=0.005, alpha_hi=0.30):
    """
    Adaptive Conformal Inference (Gibbs & Candes, 2021) quantile of a score stream.

    The static split-conformal half-width is a fixed ``(1 - alpha)`` quantile of
    the calibration scores, blind to whether the recent ones have been missing.
    Here the scores are walked in time order and the miscoverage level is
    updated as ``alpha_t+1 = alpha_t + gamma * (alpha - err_t)``, ``err_t = 1``
    when score ``t`` fell outside the current quantile. A run of misses lowers
    ``alpha_t`` (a wider band), a run of covers raises it. The quantile is
    always taken over the whole pool and only the level adapts, clipped to
    ``[alpha_lo, alpha_hi]``; the result is the quantile at the level reached
    at the end of the stream. The same procedure as HAR-RV's cone
    (``HAR_RV_model._aci_multipliers``), batched.

    Parameters
    ----------
    scores : array-like
        1-D stream, or 2-D with one stream per row; time runs along the last
        axis. Non-finite values are not allowed in 2-D input and are dropped
        from 1-D input.
    alpha, gamma, alpha_lo, alpha_hi : float, optional
        Target miscoverage, learning rate, and the clipping bounds.

    Returns
    -------
    float or np.ndarray
        The adapted quantile (one per row for 2-D input); NaN for an empty stream.
    """
    s = np.asarray(scores, dtype=np.float64)
    one = s.ndim == 1
    if one:
        s = s[np.isfinite(s)][None, :]
    if s.shape[1] == 0:
        return float("nan") if one else np.full(s.shape[0], np.nan)
    srt = np.sort(s, axis=1)
    a = np.full(s.shape[0], float(alpha))
    for i in range(s.shape[1]):
        k = _row_quantile(srt, 1.0 - np.clip(a, alpha_lo, alpha_hi))
        err = (s[:, i] > k).astype(np.float64)
        a = np.clip(a + gamma * (alpha - err), alpha_lo, alpha_hi)
    out = _row_quantile(srt, 1.0 - a)
    return float(out[0]) if one else out


def generate_future_dates(last_date, horizon, interval):
    """
    Generate future **NSE trading** datetimes for the forecast horizon.

    Uses the NSE exchange calendar (``pandas_market_calendars``) so generated
    timestamps fall only on real trading sessions — skipping weekends *and*
    exchange holidays, which the previous business-day/9-16h logic did not.

    - **daily**: the next ``horizon`` NSE session dates (holiday-aware),
      preserving ``last_date``'s time-of-day.
    - **hourly**: the next ``horizon`` NSE **hourly bars** at the real session
      times (09:15-15:15, 7 bars/session for the 09:15-15:30 window), rolling to
      the next trading session after the close. So a horizon expressed in hours
      maps onto genuine market bars (~7 per trading day), and a "5 trading days"
      view corresponds to ~35 bars.

    Falls back to the previous weekend-only heuristic if the calendar library is
    unavailable, so the function never hard-fails.

    Parameters
    ----------
    last_date : datetime-like
        Last observed datetime.
    horizon : int
        Number of future steps (session days for daily; hourly bars for hourly).
    interval : str
        'daily' or 'hourly'.

    Returns
    -------
    list of pd.Timestamp
        Future trading datetimes, one per forecast step.
    """
    last_date = pd.Timestamp(last_date)
    try:
        if interval == 'daily':
            tod = last_date - last_date.normalize()   # preserve time-of-day
            return [d + tod for d in _future_sessions(last_date, horizon)]

        # hourly: walk session bars, keeping only those strictly after last_date
        out = []
        # Enough sessions to cover `horizon` bars at 7 bars/session, plus margin.
        n_sessions = horizon // NSE_BARS_PER_SESSION + 3
        # Include last_date's own session too (it may have remaining bars today).
        last_day = last_date.normalize()
        sessions = [last_day] + _future_sessions(last_date, n_sessions)
        for sd in sessions:
            for hh, mm in _NSE_HOURLY_BAR_TIMES:
                ts = sd + pd.Timedelta(hours=hh, minutes=mm)
                if ts > last_date:
                    out.append(ts)
                    if len(out) >= horizon:
                        return out
        return out
    except Exception as exc:
        print(f"NSE calendar unavailable ({exc}); using weekend-only fallback.")
        return _generate_future_dates_naive(last_date, horizon, interval)


def _generate_future_dates_naive(last_date, horizon, interval):
    """
    Fallback date generation: weekends only, no holiday calendar (legacy logic).

    Kept as a safety net for when ``pandas_market_calendars`` is unavailable.
    """
    dates = []
    current_date = pd.Timestamp(last_date)
    date_offset = (
        BusinessDay(n=1) if interval == 'daily'
        else pd.DateOffset(hours=1)
    )
    for _ in range(horizon):
        current_date += date_offset
        if interval == 'hourly':
            if current_date.dayofweek >= 5:
                days_add = 7 - current_date.dayofweek
                current_date += pd.DateOffset(days=days_add)
                current_date = current_date.replace(hour=9, minute=15)
            if current_date.hour >= 16 or current_date.hour < 9:
                current_date += pd.DateOffset(days=1)
                current_date = current_date.replace(hour=9, minute=15)
                if current_date.dayofweek >= 5:
                    days_add = 7 - current_date.dayofweek
                    current_date += pd.DateOffset(days=days_add)
                    current_date = current_date.replace(hour=9, minute=15)
        dates.append(current_date)
    return dates


# ==========================================
# Statistical helpers (GARCH + Monte Carlo)
# ==========================================
def select_best_garch(residuals):
    """
    Select the best GARCH variant by BIC among multiple specifications.

    Tries symmetric GARCH, asymmetric GJR-GARCH, and EGARCH with Normal,
    Student-t, Skewed Student-t, and GED innovations. Uses mean='Zero'
    because ARIMA/ARIMAX residuals have zero mean by construction;
    estimating a redundant constant wastes a degree of freedom and
    double-counts the mean during simulation.

    Parameters
    ----------
    residuals : array-like
        Residual series from ARIMA/ARIMAX (e.g., model.resid()).

    Returns
    -------
    arch.univariate.base.ARCHModelResult
        Fitted volatility model with the lowest BIC.
    """
    from arch import arch_model

    candidates = [
        {'vol': 'Garch',  'p': 1, 'q': 1, 'o': 0, 'dist': 'normal'},
        {'vol': 'Garch',  'p': 1, 'q': 1, 'o': 0, 'dist': 't'},
        {'vol': 'Garch',  'p': 1, 'q': 1, 'o': 0, 'dist': 'skewt'},
        {'vol': 'Garch',  'p': 1, 'q': 1, 'o': 0, 'dist': 'ged'},
        {'vol': 'Garch',  'p': 2, 'q': 1, 'o': 0, 'dist': 't'},
        {'vol': 'Garch',  'p': 2, 'q': 2, 'o': 0, 'dist': 't'},
        {'vol': 'Garch',  'p': 1, 'q': 1, 'o': 1, 'dist': 't'},      # GJR-GARCH
        {'vol': 'Garch',  'p': 1, 'q': 1, 'o': 1, 'dist': 'skewt'},  # GJR-GARCH
        {'vol': 'EGARCH', 'p': 1, 'q': 1, 'o': 1, 'dist': 'normal'},
        {'vol': 'EGARCH', 'p': 1, 'q': 1, 'o': 1, 'dist': 't'},
        {'vol': 'EGARCH', 'p': 1, 'q': 1, 'o': 1, 'dist': 'skewt'},
    ]

    best_model = None
    best_bic = np.inf
    best_name = ""

    for spec in candidates:
        try:
            mdl = arch_model(
                residuals,
                mean='Zero',
                vol=spec['vol'], p=spec['p'], q=spec['q'],
                o=spec.get('o', 0), dist=spec['dist'],
                rescale=False,
            )
            result = mdl.fit(disp='off', show_warning=False)
            if result.bic < best_bic:
                best_bic = result.bic
                best_model = result
                best_name = (
                    f"{spec['vol']}({spec['p']},{spec.get('o', 0)},{spec['q']}) "
                    f"dist={spec['dist']}"
                )
        except Exception:
            continue

    if best_model is None:
        mdl = arch_model(
            residuals, mean='Zero', vol='Garch', p=1, q=1, rescale=False,
        )
        best_model = mdl.fit(disp='off')
        best_name = 'GARCH(1,1) [fallback]'

    print(f"Best volatility model: {best_name} (BIC={best_bic:.2f})")
    return best_model


def simulate_price_paths(mean_forecast, garch_model, last_price, horizon,
                         method='simulation'):
    """
    Monte Carlo simulation of future price paths for an ARIMA/ARIMAX+GARCH model.

    Combines a deterministic mean-return forecast with GARCH-simulated
    residual paths to produce realistic price distributions that respect the
    fitted innovation distribution (Normal, Student-t, Skewed-t, etc.), fat
    tails, and volatility clustering.

    Parameters
    ----------
    mean_forecast : array-like
        Deterministic mean log-return forecast per step (length ``horizon``).
        For ARIMAX this is conditioned on the estimated future exogenous
        variables; for plain ARIMA it is the unconditional mean forecast.
    garch_model : arch.univariate.base.ARCHModelResult
        Fitted GARCH volatility model (mean='Zero').
    last_price : float
        Last observed adjusted close price.
    horizon : int
        Number of steps to simulate.
    method : str, optional
        'simulation' (default) draws the shocks from the fitted distribution.
        'bootstrap' draws them, with replacement, from the model's own
        standardized residuals: filtered historical simulation (Barone-Adesi,
        Giannopoulos and Vosper, Journal of Futures Markets 1999; README
        section 114).

    Returns
    -------
    dict
        'predicted_price' : np.ndarray (horizon,) -- median price per step.
        'lower_bound'     : np.ndarray (horizon,) -- 2.5th percentile (95% CI).
        'upper_bound'     : np.ndarray (horizon,) -- 97.5th percentile (95% CI).
        'volatility'      : np.ndarray (horizon,) -- std of simulated returns.
    """
    np.random.seed(42)
    # arch draws its simulation shocks from the fitted distribution's OWN
    # generator, which np.random.seed does not touch: two identical runs gave
    # different ARIMA/ARIMAX bands. Reseed that generator so the cone is
    # reproducible like every other model's.
    try:
        garch_model.model.distribution._generator = np.random.default_rng(42)
    except Exception:
        pass

    mean_forecast = np.asarray(mean_forecast, dtype=np.float64)

    try:
        if method == 'bootstrap':
            garch_sim = garch_model.forecast(
                horizon=horizon, method='bootstrap', simulations=N_SIMS,
                random_state=np.random.RandomState(42),
            )
        else:
            garch_sim = garch_model.forecast(
                horizon=horizon, method='simulation', simulations=N_SIMS,
            )
        simulated_residuals = garch_sim.simulations.values[-1]
        if simulated_residuals.shape[0] != horizon:
            simulated_residuals = simulated_residuals.T
    except Exception:
        try:
            garch_fcast = garch_model.forecast(horizon=horizon)
        except Exception:
            # Same path count as the main branch: at 1,000 paths the band's
            # 2.5% edge moves ~4-7% of its half-width between seeds, about
            # twice the noise at N_SIMS (run_mc_variance_reduction.py).
            garch_fcast = garch_model.forecast(
                horizon=horizon, method='simulation', simulations=N_SIMS,
            )
        pred_var = garch_fcast.variance.values[-1, :]
        pred_std = np.sqrt(np.maximum(pred_var, 0.0))
        simulated_residuals = (
            pred_std[:, None] * np.random.standard_normal((horizon, N_SIMS))
        )

    simulated_returns = mean_forecast[:, None] + simulated_residuals

    cum_returns = np.cumsum(simulated_returns, axis=0)
    price_paths = last_price * np.exp(cum_returns)

    valid = np.all(np.isfinite(price_paths) & (price_paths > 0), axis=0)
    if valid.sum() >= 100:
        price_paths = price_paths[:, valid]
        simulated_returns = simulated_returns[:, valid]

    return {
        'predicted_price': np.median(price_paths, axis=1),
        'lower_bound': np.percentile(price_paths, 2.5, axis=1),
        'upper_bound': np.percentile(price_paths, 97.5, axis=1),
        'volatility': np.std(simulated_returns, axis=1),
    }


# ==========================================
# Keras training helpers
# ==========================================
_GPU_CONFIGURED = False
_GPU_AVAILABLE = False


def configure_compute(verbose=True):
    """
    Detect GPU availability and configure TensorFlow to use it efficiently.

    Idempotent: the actual configuration — enabling per-GPU memory growth, which
    must happen before any GPU is initialised — runs only on the first call;
    later calls return the cached result. When a GPU is present, TensorFlow/Keras
    use it automatically for the deep-learning models (and Chronos uses it via
    torch); when none is found, everything runs on CPU. Memory growth is enabled
    so TensorFlow allocates VRAM lazily instead of grabbing all of it up front.

    The gradient-boosted tree models (XGBoost/LightGBM) are intentionally left on
    CPU: on the small windows used here (~hundreds of rows) GPU tree training is
    typically slower than CPU due to kernel-launch overhead.

    Parameters
    ----------
    verbose : bool, optional
        Print a one-line summary of the selected device. Default True.

    Returns
    -------
    bool
        True if at least one usable GPU was detected, else False.
    """
    global _GPU_CONFIGURED, _GPU_AVAILABLE
    if _GPU_CONFIGURED:
        return _GPU_AVAILABLE
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass  # GPU already initialised; growth can no longer be set
        _GPU_AVAILABLE = len(gpus) > 0
        if verbose:
            msg = (f"TensorFlow sees {len(gpus)} GPU(s) — the Keras models will use it"
                   if _GPU_AVAILABLE else
                   "TensorFlow sees no GPU — the Keras models run on CPU "
                   "(Chronos-2 checks PyTorch separately; see compute_status)")
            print(f"[compute] {msg}.")
    except Exception as exc:
        _GPU_AVAILABLE = False
        if verbose:
            print(f"[compute] GPU check failed ({exc}); running on CPU.")
    _GPU_CONFIGURED = True
    return _GPU_AVAILABLE


def gpu_available():
    """
    Return whether a usable GPU is present (runs :func:`configure_compute` once).

    Returns
    -------
    bool
        True if a GPU was detected, else False.
    """
    if _GPU_CONFIGURED:
        return _GPU_AVAILABLE
    return configure_compute(verbose=False)


_COMPUTE_STATUS = None


def compute_status():
    """
    Which framework can use a GPU here, and why not when one is present (cached).

    The two GPU frameworks are independent, and asking only TensorFlow was
    wrong: on native Windows TensorFlow >= 2.11 has no GPU support at all, so
    the app reported "CPU" on a machine with a GPU even when PyTorch (Chronos-2)
    could have used it, and never said why (README section 79). An NVIDIA GPU
    is taken to be present when ``nvidia-smi`` is on the PATH, as the driver
    installs it.

    Returns
    -------
    dict
        ``tf_gpus`` (int), ``torch_cuda`` (bool), ``torch_version`` (str or
        None), ``nvidia`` (bool), ``hints`` (list of str: what to change to
        use the GPU that is there).
    """
    global _COMPUTE_STATUS
    if _COMPUTE_STATUS is not None:
        return _COMPUTE_STATUS
    import os
    import shutil
    configure_compute(verbose=False)
    tf_gpus = 0
    if _GPU_AVAILABLE:
        try:
            import tensorflow as tf
            tf_gpus = len(tf.config.list_physical_devices('GPU'))
        except Exception:
            tf_gpus = 0
    torch_cuda, torch_version = False, None
    try:
        import torch
        torch_version = torch.__version__
        torch_cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    nvidia = shutil.which("nvidia-smi") is not None
    hints = []
    if nvidia and tf_gpus == 0:
        hints.append(
            "TensorFlow has no GPU support on native Windows since 2.11, so the "
            "Keras models (LSTM, GRU, Transformer, TCN, N-BEATS, CNN-LSTM, Hybrid, "
            "Vol-LSTM) run on CPU; WSL2 is the only GPU route for them."
            if os.name == "nt" else
            "TensorFlow sees no GPU; install its CUDA build (pip install "
            "\"tensorflow[and-cuda]\") for the Keras models to use it.")
    if nvidia and not torch_cuda:
        hints.append(
            f"PyTorch {torch_version or '(not installed)'} cannot use the GPU, so "
            "Chronos-2 runs on CPU. pip installs the CPU-only build on Windows; "
            "install the CUDA build for this driver from "
            "https://pytorch.org/get-started/locally/ "
            "(pip install torch --index-url https://download.pytorch.org/whl/cuXXX).")
    _COMPUTE_STATUS = {"tf_gpus": tf_gpus, "torch_cuda": torch_cuda,
                       "torch_version": torch_version, "nvidia": nvidia, "hints": hints}
    return _COMPUTE_STATUS


def compute_device_label(status=None):
    """
    Human-readable label of what runs where, for UI/logging.

    Returns
    -------
    str
        e.g. 'CPU', 'GPU (TensorFlow and PyTorch)', or
        'GPU for Chronos-2 (PyTorch); CPU for the TensorFlow models'.
    """
    s = compute_status() if status is None else status
    if s["tf_gpus"] and s["torch_cuda"]:
        return "GPU (TensorFlow and PyTorch)"
    if s["tf_gpus"]:
        n = s["tf_gpus"]
        return f"GPU for the TensorFlow models ({n} device{'s' if n != 1 else ''}); CPU for Chronos-2"
    if s["torch_cuda"]:
        return "GPU for Chronos-2 (PyTorch); CPU for the TensorFlow models"
    return "CPU (an NVIDIA GPU is present but unused)" if s["nvidia"] else "CPU"


def reset_keras_seeds(seed=42):
    """
    Clear the Keras session and set NumPy/TensorFlow seeds for reproducibility.

    Also ensures GPU detection / memory-growth configuration has run (once), so
    the deep-learning models transparently use a GPU when one is available and
    fall back to CPU otherwise.

    Parameters
    ----------
    seed : int, optional
        Random seed applied to both TensorFlow and NumPy. Default 42.
    """
    import tensorflow as tf
    configure_compute()
    tf.keras.backend.clear_session()
    # Centralised seeding (random + numpy + tf + torch) so the NN path is pinned
    # the same way every other run is; falls back to local seeding if the helper
    # is somehow unavailable.
    try:
        from reproducibility import set_all_seeds
        set_all_seeds(seed)
    except Exception:
        tf.random.set_seed(seed)
        np.random.seed(seed)


def temporal_train_val_split(X, y, val_frac=0.1, embargo=None):
    """
    Split sequences temporally, reserving the most recent fraction for validation.

    An **embargo** gap is dropped between the train and validation blocks. With
    direct multi-horizon targets each label spans several future bars
    (``log(P[t+h] / P[t])``), so the last training samples and the first
    validation samples would otherwise share overlapping future windows. That
    overlap leaks information into validation and makes early-stopping
    ``val_loss`` look better than it really is. Embargoing the gap (purged
    validation) removes the overlap so early stopping reflects genuine
    out-of-sample error.

    Parameters
    ----------
    X : np.ndarray
        Input sequences/samples.
    y : np.ndarray
        Targets aligned with X. May be 1-D (single-step) or 2-D (n, horizon)
        for direct multi-horizon models.
    val_frac : float, optional
        Fraction of the most recent samples used for validation. Default 0.1.
    embargo : int, optional
        Number of samples to drop between train and validation. When None it is
        auto-derived from the label span: the number of target columns for 2-D
        ``y`` (the max horizon), else 1 for single-step targets. Automatically
        capped so the training block keeps at least one sample.

    Returns
    -------
    tuple of np.ndarray
        (X_train, X_val, y_train, y_val).
    """
    y = np.asarray(y)
    if embargo is None:
        embargo = y.shape[1] if y.ndim > 1 else 1
    val_size = max(1, int(len(X) * val_frac))
    # Keep at least one training sample after removing val + embargo rows.
    embargo = max(0, min(embargo, len(X) - val_size - 1))
    train_end = len(X) - val_size - embargo
    return X[:train_end], X[-val_size:], y[:train_end], y[-val_size:]


def default_keras_callbacks(patience=10, lr_patience=5):
    """
    Build the standard EarlyStopping + ReduceLROnPlateau callback list.

    Parameters
    ----------
    patience : int, optional
        EarlyStopping patience (epochs). Default 10.
    lr_patience : int, optional
        ReduceLROnPlateau patience (epochs). Default 5.

    Returns
    -------
    list of tf.keras.callbacks.Callback
        Callbacks monitoring 'val_loss'.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    return [
        EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=lr_patience, min_lr=1e-6,
        ),
    ]


def prepare_feature_sequences(df, look_back, flatten=False, val_frac=0.1):
    """
    Build scaled multivariate look-back sequences for neural-network models.

    Computes the 9-feature matrix from adjusted close prices, discards warmup
    rows where rolling windows are incomplete, scales the features to [0, 1],
    and creates overlapping look-back sequences for supervised learning. When
    ``flatten`` is True each sequence is flattened to a 1-D vector (as required
    by the fully-connected N-BEATS blocks).

    Two robustness choices distinguish this from a naive setup:

    - **No scaler leakage.** The MinMaxScaler is fit on the training portion
      only (the first ``1 - val_frac`` of rows); the validation/most-recent
      rows are merely transformed. Fitting on the whole series would leak the
      future feature range into training and inflate validation scores.
    - **Return target, not price level.** The target is the next-step log
      return ``log(price[i] / price[i-1])`` rather than the (scaled) price.
      Predicting the non-stationary price level on near-random-walk data makes
      the network simply echo the last price (the persistence trap) and pushes
      future prices outside the scaler's fitted [0, 1] range. Log returns are
      stationary and scale-free; the caller reconstructs prices by compounding.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    look_back : int
        Number of past time steps per input sequence.
    flatten : bool, optional
        If True, flatten each (look_back, n_features) sequence to
        (look_back * n_features,). Default False.
    val_frac : float, optional
        Fraction of the most recent rows excluded from the scaler fit (kept
        consistent with the training validation split). Default 0.1.

    Returns
    -------
    X : np.ndarray
        Input sequences. Shape (n_samples, look_back, 9), or
        (n_samples, look_back * 9) when ``flatten`` is True.
    y : np.ndarray
        Target values — next-step log returns — of shape (n_samples,).
    scaler : MinMaxScaler
        Scaler fit on the training feature rows, for scaling prediction inputs.
    n_features : int
        Number of features (9).
    prices : np.ndarray
        Price array used for feature computation (after warmup removal).
    """
    from sklearn.preprocessing import MinMaxScaler

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    coerce_numeric_columns(df)

    prices = df['adj_close'].dropna().values.astype(np.float64)
    features = compute_price_features(prices)

    features = features[_SMA_LONG:]
    prices = prices[_SMA_LONG:]

    n_features = features.shape[1]

    # Fit the scaler on the training portion only to avoid lookahead bias.
    n_rows = len(features)
    val_size = max(1, int(n_rows * val_frac))
    train_n = max(1, n_rows - val_size)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(features[:train_n])
    scaled = scaler.transform(features)

    X, y = [], []
    for i in range(look_back, len(scaled)):
        seq = scaled[i - look_back:i]
        X.append(seq.flatten() if flatten else seq)
        # Target is the next-step log return, reconstructed to price by caller.
        y.append(np.log(prices[i] / prices[i - 1]))

    return np.array(X), np.array(y), scaler, n_features, prices


def predict_recursive_prices(model, scaler, price_history, n_features, horizon,
                             interval, last_date, look_back, flatten=False,
                             desc="future steps recursively"):
    """
    Predict future prices by feeding predictions back into a neural model.

    The model predicts the next-step log return (see ``prepare_feature_sequences``);
    each predicted return is compounded onto the running price. After each step
    the full 9-feature set is recomputed from the updated price buffer so the
    network always sees coherent, self-consistent inputs. Date generation
    respects business days (daily) or market hours 9:15-16:00 (hourly).

    Parameters
    ----------
    model : tf.keras.Model
        Trained model (LSTM, GRU, Transformer, TCN, or N-BEATS).
    scaler : MinMaxScaler
        Fitted scaler from training for feature scaling.
    price_history : array-like
        Recent price history used as the initial prediction buffer.
    n_features : int
        Number of feature columns expected by the model (9).
    horizon : int
        Number of future steps to predict.
    interval : str
        'daily' or 'hourly'.
    last_date : datetime-like
        Last observed date in the training data.
    look_back : int
        Number of past timesteps the model expects as input.
    flatten : bool, optional
        If True, flatten each input window before prediction (for N-BEATS).
        Default False.
    desc : str, optional
        Description fragment used in the progress message.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price'].
    """
    future_dates = generate_future_dates(last_date, horizon, interval)
    print(f"Generating {horizon} {desc}...")
    path = _recursive_price_path(
        model, scaler, price_history, n_features, horizon, look_back, flatten,
    )
    return pd.DataFrame({'datetime': future_dates, 'predicted_price': path})


def _recursive_price_path(model, scaler, price_history, n_features, horizon,
                          look_back, flatten=False):
    """
    Recursively forecast ``horizon`` prices from a price buffer (no dates).

    Each step recomputes the 9-feature matrix from the running buffer, scales
    it, predicts the next log return, and compounds it onto the last price.
    Shared by ``predict_recursive_prices`` and the conformal calibration loop.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    scaler : MinMaxScaler
        Fitted feature scaler.
    price_history : array-like
        Initial price buffer (the observed prices up to the forecast origin).
    n_features : int
        Number of feature columns.
    horizon : int
        Steps to forecast.
    look_back : int
        Input window length.
    flatten : bool, optional
        Flatten the window for N-BEATS. Default False.

    Returns
    -------
    list of float
        ``horizon`` predicted prices.
    """
    price_buffer = list(price_history)
    out = []
    for _ in range(horizon):
        raw_features = compute_price_features(price_buffer)
        recent = raw_features[-look_back:]
        scaled = scaler.transform(recent)

        if flatten:
            input_seq = scaled.flatten().reshape(1, -1)
        else:
            input_seq = scaled.reshape(1, look_back, n_features)

        pred_return = float(model.predict(input_seq, verbose=0)[0, 0])
        pred_price = float(price_buffer[-1] * np.exp(pred_return))
        price_buffer.append(pred_price)
        out.append(pred_price)
    return out


def forecast_nn_conformal(df, interval, horizon, look_back, train_fn,
                          flatten=False, desc="future steps recursively",
                          alpha=0.05, cal_frac=0.2, max_cal_origins=80):
    """
    Train a sequence NN and forecast with split-conformal prediction intervals.

    Adds distribution-free uncertainty bands to the recursive neural models
    (LSTM/GRU, Transformer, TCN, N-BEATS), which previously emitted point
    forecasts only. Uses **horizon-wise split conformal**:

    1. Split the feature rows temporally — the earlier ``1 - cal_frac`` train
       the network (scaler fit on this block only, so no lookahead), the most
       recent ``cal_frac`` form the calibration block.
    2. From each calibration origin, run the model's own recursive ``horizon``-
       step forecast and record the absolute cumulative-log-return error at each
       step h: ``|log(actual[t+h] / predicted[t+h])|``.
    3. The finite-sample (1 - alpha) quantile of those errors is the per-horizon
       half-width, so intervals widen with horizon and reflect the model's real
       multi-step recursive error (including drift).
    4. Forecast the future from the full observed history and wrap each step in
       ``price * exp(±half_width)``.

    The same proper-train model is used for calibration and the final point
    forecast (textbook split conformal), so this costs only one network fit —
    but the point forecast now trains on ~80% of the data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (max-obs-capped) DataFrame with 'datetime' and 'adj_close'.
    interval : str
        'daily' or 'hourly'.
    horizon : int
        Steps to forecast.
    look_back : int
        Input window length.
    train_fn : callable
        ``train_fn(X, y) -> model`` building and training the architecture.
    flatten : bool, optional
        Flatten windows (N-BEATS). Default False.
    desc : str, optional
        Progress-message fragment.
    alpha : float, optional
        Miscoverage; 0.05 -> 95% intervals. Default 0.05.
    cal_frac : float, optional
        Fraction of most-recent rows held out for calibration. Default 0.2.
    max_cal_origins : int, optional
        Cap on calibration origins (subsampled with a stride) to bound cost.

    Returns
    -------
    pd.DataFrame
        Columns ['datetime', 'predicted_price', 'lower_bound', 'upper_bound',
        'volatility']. If there is too little data to calibrate, only
        ['datetime', 'predicted_price'] are returned.
    """
    from sklearn.preprocessing import MinMaxScaler

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    coerce_numeric_columns(df)

    prices = df['adj_close'].dropna().values.astype(np.float64)
    features = compute_price_features(prices)
    features = features[_SMA_LONG:]
    prices = prices[_SMA_LONG:]

    n = len(features)
    n_features = features.shape[1]

    # Temporal split: proper-train rows vs the recent calibration block.
    cal_size = max(look_back + horizon + 5, int(round(n * cal_frac)))
    train_end = n - cal_size
    calibrate = train_end > look_back + 10
    if not calibrate:
        train_end = n   # too little data: train on all, skip conformal bands

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(features[:train_end])
    scaled = scaler.transform(features)

    X, y = [], []
    for i in range(look_back, train_end):
        seq = scaled[i - look_back:i]
        X.append(seq.flatten() if flatten else seq)
        y.append(np.log(prices[i] / prices[i - 1]))
    X, y = np.array(X), np.array(y)

    model = train_fn(X, y)

    half = {h: None for h in range(1, horizon + 1)}
    if calibrate:
        resid = {h: [] for h in range(1, horizon + 1)}
        origins = list(range(train_end, n - horizon + 1))
        if len(origins) > max_cal_origins:
            stride = int(np.ceil(len(origins) / max_cal_origins))
            origins = origins[::stride]
        for t in origins:
            path = _recursive_price_path(
                model, scaler, prices[:t], n_features, horizon,
                look_back, flatten,
            )
            for h in range(1, horizon + 1):
                actual = prices[t + h - 1]
                resid[h].append(abs(np.log(actual / path[h - 1])))
        for h in range(1, horizon + 1):
            r = np.asarray(resid[h], dtype=float)
            m = len(r)
            if m >= 1:
                q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
                half[h] = float(np.quantile(r, q_level))

    last_date = df.index[-1]
    future_dates = generate_future_dates(last_date, horizon, interval)
    print(f"Generating {horizon} {desc} (conformal)...")
    point = _recursive_price_path(
        model, scaler, prices, n_features, horizon, look_back, flatten,
    )

    rows = []
    for i, h in enumerate(range(1, horizon + 1)):
        p = float(point[i])
        q = half[h]
        if q is None:
            rows.append({'datetime': future_dates[i], 'predicted_price': p})
        else:
            lower = float(p * np.exp(-q))
            upper = float(p * np.exp(q))
            rows.append({
                'datetime': future_dates[i],
                'predicted_price': p,
                'lower_bound': lower,
                'upper_bound': upper,
                'volatility': float((upper - lower) / (2 * 1.96)),
            })

    return pd.DataFrame(rows)


def forecast_nn_direct(df, interval, horizon, look_back, build_train_fn,
                       flatten=False, desc="future steps",
                       alpha=0.05, cal_frac=0.2, model_name=None,
                       feature_fn=None, use_context=True):
    """
    Train a direct multi-horizon sequence NN and forecast with conformal bands.

    The recursive scheme (``forecast_nn_conformal``) predicts one step and feeds
    it back, so errors compound across the horizon — the failure mode that made
    the recursive N-BEATS/LSTM forecasts blow up by 3-10x on some tickers. This
    **direct** variant instead trains a network with ``horizon`` outputs that
    predicts the cumulative log return ``log(P[t+h] / P[t])`` for every step h
    from the origin window in one shot. No prediction is ever fed back, so the
    horizon-h error is bounded by the model's own h-step accuracy rather than an
    accumulation of single-step errors.

    Distribution-free split-conformal intervals are computed per horizon from a
    held-out recent calibration block (one forward pass per calibration window,
    not a recursive rollout), so this is also faster than the recursive variant.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (max-obs-capped) DataFrame with 'datetime' and 'adj_close'.
    interval : str
        'daily' or 'hourly'.
    horizon : int
        Steps to forecast (also the network's output dimension).
    look_back : int
        Input window length.
    build_train_fn : callable
        ``build_train_fn(X, Y, output_dim) -> model`` building and training a
        network with ``output_dim`` outputs on inputs X and (n, horizon) targets Y.
    flatten : bool, optional
        Flatten windows (N-BEATS). Default False.
    desc : str, optional
        Progress-message fragment.
    alpha : float, optional
        Miscoverage; 0.05 -> 95% intervals. Default 0.05.
    cal_frac : float, optional
        Fraction of most-recent samples held out for calibration. Default 0.2.
    feature_fn : callable, optional
        ``feature_fn(prices) -> (n,) or (n, k) array`` replacing
        :func:`compute_price_features` (README section 94: N-BEATS on its own
        returns only). None: the shared 9 price features.
    use_context : bool, optional
        Append the market/VIX context columns when the frame carries them.
        Default True.

    Returns
    -------
    pd.DataFrame
        Columns ['datetime', 'predicted_price', 'lower_bound', 'upper_bound',
        'volatility']. Bounds are omitted only if there is too little data to
        calibrate.
    """
    from sklearn.preprocessing import MinMaxScaler

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    coerce_numeric_columns(df)

    adj = df['adj_close'].dropna()
    prices = adj.values.astype(np.float64)
    if feature_fn is None:
        features = compute_price_features(prices)
    else:
        features = np.asarray(feature_fn(prices), dtype=np.float32)
        if features.ndim == 1:
            features = features[:, None]

    # Feature parity with the tree models: append the market/VIX context
    # columns (if the df was enriched by market_context) so the sequence models
    # also see the dominant market component. Aligned to the same rows as
    # `prices`; degrades to the price-only matrix when no context is present.
    from market_context import context_matrix
    ctx, ctx_cols = context_matrix(df.loc[adj.index]) if use_context else (None, [])
    if ctx is not None and len(ctx) == len(features):
        features = np.hstack([features, ctx])
        print(f"  + {len(ctx_cols)} market-context features for the NN: {ctx_cols}")

    # Excess-return target reframing: if the df carries an aligned market index
    # (market_context.add_market_index), subtract the market's cumulative return
    # from the target so the network learns the stock-specific (alpha) part.
    # Reconstruction is unchanged (expected future market return ~ 0).
    mkt = None
    if 'mkt_index' in df.columns:
        m_full = pd.to_numeric(
            df.loc[adj.index, 'mkt_index'], errors='coerce'
        ).ffill().bfill().values.astype(np.float64)
        if len(m_full) == len(prices) and np.all(m_full > 0):
            mkt = m_full
            print("  using excess-return targets (stock - market) for the NN")

    features = features[_SMA_LONG:]
    prices = prices[_SMA_LONG:]
    if mkt is not None:
        mkt = mkt[_SMA_LONG:]
    n = len(features)
    n_features = features.shape[1]

    # Training samples: window [i-look_back:i], origin o = i-1, with realised
    # cumulative returns out to o + horizon (so i must satisfy i <= n - horizon).
    all_i = list(range(look_back, n - horizon + 1))
    if len(all_i) < 40:
        train_is, cal_is = all_i, []
    else:
        cal_size = max(look_back + horizon, int(round(len(all_i) * cal_frac)))
        cal_size = min(cal_size, len(all_i) - 20)
        cal_is = all_i[-cal_size:]
        # Embargo: drop `horizon` samples between the train and calibration
        # blocks so no training label window log(P[o+h]/P[o]) overlaps a
        # calibration label window. Without this purge the calibration
        # residuals are correlated with the fitted labels and the conformal
        # intervals under-cover (optimistic). Capped to keep >=20 train rows.
        embargo = max(0, min(horizon, len(all_i) - cal_size - 20))
        train_is = all_i[:len(all_i) - cal_size - embargo]

    scaler_fit_end = cal_is[0] if cal_is else n
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(features[:scaler_fit_end])
    scaled = scaler.transform(features)

    def _window(i):
        w = scaled[i - look_back:i]
        return w.flatten() if flatten else w

    def _targets(i):
        o = i - 1
        base = [np.log(prices[o + h] / prices[o]) for h in range(1, horizon + 1)]
        if mkt is not None:
            base = [base[h - 1] - np.log(mkt[o + h] / mkt[o])
                    for h in range(1, horizon + 1)]
        return base

    X = np.array([_window(i) for i in train_is])
    Y = np.array([_targets(i) for i in train_is])

    model = build_train_fn(X, Y, horizon)

    half = {h: None for h in range(1, horizon + 1)}
    if cal_is:
        Xc = np.array([_window(i) for i in cal_is])
        Yc = np.array([_targets(i) for i in cal_is])
        Pc = np.asarray(model.predict(Xc, verbose=0))
        resid = np.abs(Yc - Pc)
        for hi, h in enumerate(range(1, horizon + 1)):
            col = resid[:, hi]
            m = len(col)
            q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
            half[h] = float(np.quantile(col, q_level))

    # Final forecast: the most recent window, origin = last observed price.
    last_window = scaled[n - look_back:n]
    xin = (last_window.flatten().reshape(1, -1) if flatten
           else last_window.reshape(1, look_back, n_features))
    point_cumrets = np.asarray(model.predict(xin, verbose=0))[0]

    last_price = float(prices[-1])
    last_date = df.index[-1]
    future_dates = generate_future_dates(last_date, horizon, interval)
    print(f"Generating {horizon} {desc} (direct, conformal)...")

    rows = []
    for hi, h in enumerate(range(1, horizon + 1)):
        cr = float(point_cumrets[hi])
        p = float(last_price * np.exp(cr))
        q = half[h]
        if q is None:
            rows.append({'datetime': future_dates[hi], 'predicted_price': p})
        else:
            lower = float(last_price * np.exp(cr - q))
            upper = float(last_price * np.exp(cr + q))
            rows.append({
                'datetime': future_dates[hi],
                'predicted_price': p,
                'lower_bound': lower,
                'upper_bound': upper,
                'volatility': float((upper - lower) / (2 * 1.96)),
            })

    out = pd.DataFrame(rows)

    # Epistemic (model) uncertainty by MC dropout - Klaas Ch. 4. Captured HERE
    # because this is the only place both the trained model and the exact input
    # window `xin` are in scope; doing it downstream would need either to be
    # passed around or rebuilt, and a rebuilt input is not the one that produced
    # the forecast. Off by default (config.USE_EPISTEMIC) and always reported
    # separately from the conformal band, never folded into it.
    try:
        import config as _cfg
        if _cfg.USE_EPISTEMIC:
            import epistemic as _epi
            # The conformal half-width is the model's ALEATORIC scale, in the
            # same log-return units as the epistemic spread. Passing it makes
            # the two comparable and gives the only cross-model figure that
            # means anything: what share of this model's uncertainty comes from
            # not knowing the parameters, rather than from irreducible noise.
            _hw = [v for v in half.values() if v is not None and np.isfinite(v)]
            _sigma = (float(np.mean(_hw)) / 1.96) if _hw else None
            _epi.attach(out, model, xin, n_passes=_cfg.EPISTEMIC_PASSES,
                        aleatoric_scale=_sigma)
    except Exception:
        pass

    # Record the architecture that was actually built, not the one that was
    # requested. Every deep model routes through here, so one hook covers DL,
    # N-BEATS, TCN, Transformer and CNN-LSTM; the UI lifts it into the run
    # manifest. Introspection rather than a declared spec because it cannot
    # drift from the fit it describes. Best-effort: an audit annotation must
    # never be able to fail a forecast.
    try:
        import architectures as _arch
        # The registry KEY, not the display string: passing desc ("CNN-LSTM
        # forecasts") silently produced spec_driven=false for a model that is in
        # fact spec-driven, i.e. the manifest under-reported the control it has.
        _arch.attach(out, model, model_name)
    except Exception:
        pass
    return out
```
