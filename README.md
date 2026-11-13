- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
import contextlib
import io
import pandas as pd
import numpy as np
from prophet import Prophet
import logging
import warnings

from model_utils import normalize_interval, generate_future_dates

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _patch_prophet_pandas3_compat():
    """
    Make Prophet's component-matrix build robust to pandas >= 3.0.

    ``Prophet.add_group_component`` concatenates component rows without
    ``ignore_index``, leaving the frame with a **duplicate-labelled index**.
    pandas 3.0 tightened ``pd.crosstab`` so it now raises *"cannot reindex on an
    axis with duplicate labels"* inside ``regressor_column_matrix`` during
    ``model.fit`` — breaking Prophet 1.2.2 entirely on pandas 3.x. We wrap the
    method to return a clean ``RangeIndex``. This is safe because the index is
    positional only — the real feature-column positions live in the ``'col'``
    column — and is harmless on older pandas, so it can always be applied.
    Idempotent.

    Prophet 1.3.0+ fixed this upstream, and requirements now pin 1.5.0, where
    the patch changes nothing (same forecasts with and without it; README
    section 76). It stays because ``run_app.py`` never replaces an installed
    package, so an environment still on 1.2.2 depends on it.
    """
    try:
        if getattr(Prophet.add_group_component, '_pandas3_patched', False):
            return
        _orig = Prophet.add_group_component

        def _patched(self, components, name, group):
            return _orig(self, components, name, group).reset_index(drop=True)

        _patched._pandas3_patched = True
        Prophet.add_group_component = _patched
    except Exception as exc:   # never let a patch failure break import
        print(f"Prophet pandas-3 compat patch skipped: {exc}")


_patch_prophet_pandas3_compat()


# ==========================================
# 1. Helper: Prepare Data for Prophet
# ==========================================
#: README section 89. Every configuration of Prophet tried on NSE data before
#: 2025 forecast worse than no change at every horizon. The least bad, used by
#: default: the forecast starts from the last price (``ANCHOR``) instead of
#: Prophet's smooth fitted curve, which sat 2-3% away from it on average; the
#: trend bends far less (``CHANGEPOINT_PRIOR``); and the monthly and quarterly
#: seasonalities, which do not repeat in prices, are dropped. On the diagnosis
#: forecasts this cut the five-session error from 2.07 to 1.18 times no change.
ANCHOR = True
CHANGEPOINT_PRIOR = 0.01
EXTRA_SEASONALITY = False
#: The configuration before section 89, for comparisons.
LEGACY = dict(anchor=False, changepoint_prior_scale=0.1, extra_seasonality=True)
#: The switches of section 116, both off by default: a predicted change shrunk
#: toward no change by the calibration origins' slope (``shrink``), and a flat
#: trend with weekly seasonality only (``flat``).
PROPHET_SWITCHES_LEGACY = dict(shrink=False, flat=False)
#: Daily bars, every horizon: both on. Chosen on 32 forecasts and confirmed on
#: 72 fresh ones before the reserved period, at five sessions and at 20-90
#: (run_prophet_improve.py, README section 116).
PROPHET_DAILY = dict(shrink=True, flat=True)
#: Hourly bars: the same two switches, tested separately on hourly bars before
#: 2025 (chosen on 32 forecasts, confirmed on 66 fresh ones; run_hourly_fast.py,
#: README section 118).
PROPHET_HOURLY = dict(shrink=True, flat=True)


def resolve_switches(interval, horizon, shrink=None, flat=None):
    """
    The switches a run uses. A switch left as None takes the default:
    PROPHET_DAILY on daily bars, PROPHET_HOURLY on hourly bars
    (PROPHET_SWITCHES_LEGACY restores the old behaviour). An explicit value
    is kept. ``horizon`` is taken so a horizon-dependent
    default can be added without changing callers.
    """
    base = PROPHET_DAILY if normalize_interval(interval) == 'daily' else PROPHET_HOURLY
    return {k: base[k] if v is None else bool(v) for k, v in dict(shrink=shrink, flat=flat).items()}


def prepare_prophet_data(df: pd.DataFrame, max_obs=None):
    """
    Prepare data for Prophet: rename columns, log-transform the target,
    and compute exogenous regressors.

    Prophet requires columns named 'ds' (date) and 'y' (target).
    The target is log-transformed so Prophet's additive decomposition
    operates in log-space, equivalent to a multiplicative decomposition
    in price-space. Two exogenous regressors are computed from OHLCV
    data to help Prophet separate signal from noise during training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with columns datetime, open, high, low,
        close, adj_close, volume.
    max_obs : int, optional
        Maximum number of recent observations to keep. If None, use all.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['ds', 'y', 'log_volume_ratio',
        'range_pct']. Regressor columns may be absent if the
        source columns are missing from *df*.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    for col in ['adj_close', 'volume', 'high', 'low']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if max_obs and len(df) > max_obs:
        df = df.iloc[-max_obs:]

    if 'volume' in df.columns:
        vol_ma = df['volume'].rolling(20, min_periods=1).mean()
        df['log_volume_ratio'] = np.log(
            (df['volume'] / vol_ma.replace(0, 1)).clip(lower=0.01)
        )

    if 'high' in df.columns and 'low' in df.columns:
        df['range_pct'] = (
            (df['high'] - df['low']) / df['adj_close'].replace(0, 1)
        )

    prophet_df = df.rename(columns={'datetime': 'ds', 'adj_close': 'y'})
    prophet_df = prophet_df.dropna(subset=['y'])
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    prophet_df = prophet_df[prophet_df['y'] > 0]

    prophet_df['y'] = np.log(prophet_df['y'])

    for col in ['log_volume_ratio', 'range_pct']:
        if col in prophet_df.columns:
            prophet_df[col] = (
                prophet_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            )

    keep = ['ds', 'y']
    for col in ['log_volume_ratio', 'range_pct']:
        if col in prophet_df.columns:
            keep.append(col)

    return prophet_df[keep].reset_index(drop=True)


# ==========================================
# 2. Helper: Train Advanced Model
# ==========================================
def train_prophet_model(df: pd.DataFrame, interval: str, uncertainty_samples=1000,
                        changepoint_prior_scale=CHANGEPOINT_PRIOR,
                        extra_seasonality=EXTRA_SEASONALITY, flat=False):
    """
    Configure and train a Prophet model for stock price forecasting.

    Uses log-space additive decomposition (equivalent to multiplicative
    in price-space), 95% uncertainty intervals, dynamically scaled
    changepoints, Indian market holidays, stock-specific monthly and
    quarterly seasonalities (daily only), and exogenous regressors
    (log volume ratio and high-low range) to absorb non-seasonal
    variance and improve trend/seasonality estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Prophet-formatted DataFrame with columns ['ds', 'y'] and
        optional regressor columns ['log_volume_ratio', 'range_pct'].
    interval : str
        'daily' or 'hourly'; controls which seasonalities are enabled.
    uncertainty_samples : int, optional
        Passed to Prophet. 0 skips the native interval, which the
        calibration refits do not need and which is most of a fit's cost.
    changepoint_prior_scale : float, optional
        How freely the trend bends (section 89: 0.01; it was 0.1).
    extra_seasonality : bool, optional
        Add the monthly and quarterly seasonalities (daily only); off since
        section 89.
    flat : bool, optional
        True: ``growth='flat'`` and no yearly seasonality, leaving the weekly
        (and, hourly, daily) seasonality, the holidays and the regressors
        (section 116; Prophet's documentation, 'Flat trend').

    Returns
    -------
    tuple of (Prophet, list of str)
        model : fitted Prophet model.
        regressor_cols : list of regressor column names registered
        with the model (needed for building the future dataframe).
    """
    print("Configuring Prophet Model...")

    n_changepoints = min(30, max(10, len(df) // 20))

    model = Prophet(
        growth='flat' if flat else 'linear',
        seasonality_mode='additive',
        interval_width=0.95,
        uncertainty_samples=uncertainty_samples,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=0.9,
        n_changepoints=n_changepoints,
        seasonality_prior_scale=5.0,
        holidays_prior_scale=5.0,
        daily_seasonality=(interval == 'hourly'),
        weekly_seasonality=True,
        yearly_seasonality=not flat,
    )

    if interval == 'daily' and extra_seasonality and not flat:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

    regressor_cols = [c for c in df.columns if c not in ('ds', 'y')]
    for col in regressor_cols:
        model.add_regressor(col, mode='additive')

    try:
        model.add_country_holidays(country_name='IN')
    except Exception as e:
        print(f"Warning: Could not add holidays. {e}")

    print(
        f"Training Prophet Model "
        f"(n_changepoints={n_changepoints}, regressors={regressor_cols})..."
    )
    model.fit(df)

    return model, regressor_cols


# ==========================================
# 3. Helper: Custom Future Logic
# ==========================================
def _future_frame(future_dates, regressor_cols, train_df):
    """Future ``ds`` rows with the regressors' neutral/estimated values."""
    future_df = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
    recent = train_df.tail(20)
    for col in regressor_cols:
        if col == 'range_pct' and col in recent.columns:
            future_df[col] = float(recent[col].median())
        else:
            future_df[col] = 0.0          # log_volume_ratio: volume at its average
    return future_df


def anchor_shift(model, train_df, regressor_cols):
    """
    Log distance from Prophet's fitted value at the last bar to the last price.

    Prophet forecasts from its smooth fitted curve, not from the last price,
    so the gap between them was error from the first step (README section 89).
    Adding this shift to every forecast step starts the forecast at the last
    price and keeps only Prophet's predicted change from there.
    """
    last = train_df.tail(1)
    fitted = model.predict(last[['ds'] + list(regressor_cols)])['yhat'].iloc[0]
    return float(last['y'].iloc[0] - fitted)


def generate_prophet_forecast(model, last_date, horizon, interval,
                              regressor_cols, train_df, shift=0.0):
    """
    Generate future dates, build regressor estimates, and predict.

    Creates a future dataframe with business dates (daily) or market
    hours (hourly), populates regressor columns with neutral/estimated
    future values, runs Prophet prediction, and converts results from
    log-space back to price-space. A volatility column is derived
    from the 95% confidence interval width for consistency with
    other models in the project.

    Regressor future-value strategy:
    - log_volume_ratio : 0.0 (neutral — volume at its 20-period average)
    - range_pct : recent 20-period median (mean-reverts)

    Parameters
    ----------
    model : Prophet
        Trained Prophet model.
    last_date : datetime-like
        Last date in the training data.
    horizon : int
        Number of future steps to predict.
    interval : str
        'daily' or 'hourly'.
    regressor_cols : list of str
        Regressor column names registered with the model.
    train_df : pd.DataFrame
        Training dataframe (used to compute recent regressor statistics
        for future estimation).
    shift : float, optional
        Added to Prophet's log forecast (:func:`anchor_shift`); 0 serves
        Prophet's own level.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'trend', 'volatility']
        in price-space.
    """
    print(f"Generating {horizon} future dates (Trading Logic)...")

    future_dates = generate_future_dates(last_date, horizon, interval)

    forecast = model.predict(_future_frame(future_dates, regressor_cols, train_df))

    predicted_price = np.exp(forecast['yhat'].values + shift)
    lower_bound = np.exp(forecast['yhat_lower'].values + shift)
    upper_bound = np.exp(forecast['yhat_upper'].values + shift)

    ci_width = upper_bound - lower_bound
    volatility = (ci_width / (2 * 1.96)).clip(min=0.0)

    results = pd.DataFrame({
        'datetime': forecast['ds'],
        'predicted_price': predicted_price,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'trend': np.exp(forecast['trend'].values + shift),
        'volatility': volatility,
    })

    return results


# ==========================================
# 4. Main Orchestrator
# ==========================================
#: Calibration origins for the conformal band. With 20, the 95% split-conformal
#: quantile is the largest of the 20 errors, which a new error exceeds with
#: probability 1/21 if the errors are exchangeable: 95.2% coverage.
N_CAL = 20
CAL_STRIDE = 5          # sessions (or bars) between calibration origins


def calibration_logs(raw, interval, horizon, origins, max_obs=750, anchor=ANCHOR,
                     changepoint_prior_scale=CHANGEPOINT_PRIOR,
                     extra_seasonality=EXTRA_SEASONALITY, flat=False):
    """
    At each calibration origin t, Prophet refitted on the data up to t: the
    log of the actual prices at t+1..t+horizon, Prophet's log forecasts for
    them (anchored when ``anchor``), and the log price at t. Each is an array
    with one row per origin. :func:`conformal_halfwidths` scores
    ``|actual - forecast|``; the ``shrink`` switch (section 116) regresses the
    realised change ``actual - base`` on the predicted change ``forecast - base``.
    """
    n_cal = len(origins)
    actual = np.empty((n_cal, horizon))
    pred = np.empty((n_cal, horizon))
    base = np.empty(n_cal)
    for i, t in enumerate(origins):
        train = prepare_prophet_data(raw.iloc[:t + 1], max_obs=max_obs)
        with contextlib.redirect_stdout(io.StringIO()):
            model, regressor_cols = train_prophet_model(
                train, interval, uncertainty_samples=0,
                changepoint_prior_scale=changepoint_prior_scale,
                extra_seasonality=extra_seasonality, flat=flat)
        ahead = raw.iloc[t + 1:t + 1 + horizon]
        yhat = model.predict(_future_frame(ahead['datetime'], regressor_cols, train))['yhat']
        shift = anchor_shift(model, train, regressor_cols) if anchor else 0.0
        actual[i] = np.log(ahead['adj_close'].to_numpy())
        pred[i] = yhat.to_numpy() + shift
        base[i] = float(np.log(raw['adj_close'].iloc[t]))
    return actual, pred, base


def calibration_origins(df, horizon, n_cal=N_CAL, stride=CAL_STRIDE):
    """The cleaned price frame and its calibration origins, or (frame, None)
    when there is too little history (as :func:`conformal_halfwidths`)."""
    raw = df.copy()
    raw['datetime'] = pd.to_datetime(raw['datetime'])
    raw = raw.sort_values('datetime').reset_index(drop=True)
    raw['adj_close'] = pd.to_numeric(raw['adj_close'], errors='coerce')
    raw = raw[raw['adj_close'] > 0].reset_index(drop=True)
    n = len(raw)
    origins = [n - 1 - horizon - stride * k for k in range(n_cal)]
    return raw, (None if origins[-1] < 120 else origins)


def shrink_band(actual, pred, base, alpha=0.05):
    """
    Per step: beta = clip(sum(p y) / sum(p p), 0, 1), the least-squares slope
    through the origin of the realised change y = actual - base on the
    predicted change p = pred - base over the calibration origins, and the
    finite-sample (1 - alpha) quantile of |y - beta p| (section 116; Campbell
    and Thompson 2008, as LightGBM's section 97 shrink). Returns (beta, half).
    """
    y = actual - base[:, None]
    p = pred - base[:, None]
    den = np.sum(p * p, axis=0)
    beta = np.where(den > 0, np.clip(np.sum(p * y, axis=0) / np.where(den > 0, den, 1.0), 0.0, 1.0), 0.0)
    n_cal = actual.shape[0]
    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    return beta, np.quantile(np.abs(y - beta * p), q_level, axis=0)


def conformal_halfwidths(df, interval, horizon, max_obs=750, alpha=0.05,
                         n_cal=N_CAL, stride=CAL_STRIDE, anchor=ANCHOR,
                         changepoint_prior_scale=CHANGEPOINT_PRIOR,
                         extra_seasonality=EXTRA_SEASONALITY, flat=False):
    """
    Per-step half-widths, in log price, of a split-conformal band for Prophet.

    Prophet's own interval is built from its trend and noise model, and it
    under-covered: on 20 NSE names, 2020-2024H1, its 95% band held 67% of
    5-day outcomes on 1.2.2 and 52% on 1.5.0 (README section 76). This
    measures the model's real errors instead. At each of ``n_cal`` past
    origins, ``stride`` sessions apart and ending ``horizon`` sessions before
    the last bar, Prophet is refitted exactly as for the forecast (same
    preparation, same ``max_obs`` window) on the data up to that origin, and
    the error at step h is ``|log(actual[t+h] / predicted[t+h])|``, scored on
    the dates that actually traded. The half-width for step h is the
    finite-sample ``(1 - alpha)`` quantile of its ``n_cal`` errors, the rule
    the neural models use (``model_utils.forecast_nn_conformal``).

    The refits use the forecast's own configuration, anchoring included, so
    the band measures the errors of the forecast that is served.

    Returns ``None`` when there is too little history for ``n_cal`` origins.
    """
    raw = df.copy()
    raw['datetime'] = pd.to_datetime(raw['datetime'])
    raw = raw.sort_values('datetime').reset_index(drop=True)
    raw['adj_close'] = pd.to_numeric(raw['adj_close'], errors='coerce')
    raw = raw[raw['adj_close'] > 0].reset_index(drop=True)
    n = len(raw)
    origins = [n - 1 - horizon - stride * k for k in range(n_cal)]
    if origins[-1] < 120:
        return None
    print(f"Calibrating the Prophet band on {n_cal} past origins...")
    parts = calibration_logs(raw, interval, horizon, origins, max_obs=max_obs, anchor=anchor,
                             changepoint_prior_scale=changepoint_prior_scale,
                             extra_seasonality=extra_seasonality, flat=flat)
    actual, pred, _ = parts
    errors = np.abs(actual - pred)
    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    return np.quantile(errors, q_level, axis=0)


def prophet_paths(df, interval, horizon, calibrate=True, anchor=ANCHOR,
                  changepoint_prior_scale=CHANGEPOINT_PRIOR,
                  extra_seasonality=EXTRA_SEASONALITY, flat=False, variants=(False,)):
    """
    Forecast paths for several ``shrink`` settings from one set of fits:
    {shrink: DataFrame}. ``flat`` changes the fits, so it is fixed for the
    call. With ``variants=(False,)`` and ``flat=False`` this is the served model.
    """
    interval = normalize_interval(interval)
    print(f"--- Starting Prophet Prediction ({interval}) ---")

    max_obs = 750 if interval == 'daily' else 500

    prophet_df = prepare_prophet_data(df, max_obs=max_obs)
    last_date = prophet_df['ds'].iloc[-1]

    cfg = dict(changepoint_prior_scale=changepoint_prior_scale,
               extra_seasonality=extra_seasonality)
    model, regressor_cols = train_prophet_model(prophet_df, interval, flat=flat, **cfg)
    shift = anchor_shift(model, prophet_df, regressor_cols) if anchor else 0.0
    base_df = generate_prophet_forecast(
        model, last_date, horizon, interval, regressor_cols, prophet_df, shift=shift,
    )

    parts = None
    if calibrate:
        raw, origins = calibration_origins(df, horizon)
        if origins is not None:
            print(f"Calibrating the Prophet band on {N_CAL} past origins...")
            parts = calibration_logs(raw, interval, horizon, origins, max_obs=max_obs,
                                     anchor=anchor, flat=flat, **cfg)
    last_log = float(prophet_df['y'].iloc[-1])

    outs = {}
    for shrink in tuple(dict.fromkeys(bool(v) for v in variants)):
        results_df = base_df.copy()
        native = results_df[['lower_bound', 'upper_bound']].copy()
        half, beta = None, None
        if parts is not None:
            actual, pred, base = parts
            if shrink:
                beta, half = shrink_band(actual, pred, base)
                change = np.log(results_df['predicted_price'].to_numpy()) - last_log
                results_df['predicted_price'] = np.exp(last_log + beta * change)
            else:
                n_cal = actual.shape[0]
                q_level = min(1.0, np.ceil((n_cal + 1) * (1 - 0.05)) / n_cal)
                half = np.quantile(np.abs(actual - pred), q_level, axis=0)
        if half is not None:
            p = results_df['predicted_price'].to_numpy()
            results_df['lower_bound'] = p * np.exp(-half)
            results_df['upper_bound'] = p * np.exp(half)
            results_df['volatility'] = (
                (results_df['upper_bound'] - results_df['lower_bound']) / (2 * 1.96))
        results_df.attrs['prophet'] = {
            'band_mode': 'conformal' if half is not None else 'native',
            'native_lower': [float(v) for v in native['lower_bound']],
            'native_upper': [float(v) for v in native['upper_bound']],
            'used_halfwidth': None if half is None else [float(v) for v in half],
            'n_calibration': N_CAL if half is not None else 0,
            'anchor_shift': shift,
            'config': dict(anchor=bool(anchor), **cfg),
        }
        if shrink or flat:
            results_df.attrs['prophet']['switches'] = dict(
                shrink=bool(shrink), flat=bool(flat),
                beta=None if beta is None else [float(v) for v in beta])
        outs[shrink] = results_df
    return outs


def run_prophet_prediction(df: pd.DataFrame, interval: str, horizon: int,
                           calibrate=True, anchor=ANCHOR,
                           changepoint_prior_scale=CHANGEPOINT_PRIOR,
                           extra_seasonality=EXTRA_SEASONALITY, shrink=None, flat=None):
    """
    Main function to run Prophet prediction.

    Orchestrates data preparation (log-transform target, compute
    exogenous regressors), model training (with 95% CI, dynamic
    changepoints, and volume/range regressors), and forecast
    generation with log-to-price inverse transform.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with columns datetime, open, high, low,
        close, adj_close, volume.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    calibrate : bool, optional
        When True (default) the 95% band is split-conformal
        (:func:`conformal_halfwidths`), at the cost of 20 point-only refits;
        Prophet's native band is kept in ``attrs['prophet']``. Falls back to
        the native band when the history is too short.
    anchor, changepoint_prior_scale, extra_seasonality : optional
        The configuration (section 89); ``**LEGACY`` gives the one before it.
    shrink, flat : bool or None, optional
        None (default) takes :func:`resolve_switches` (section 116): both on
        for daily bars (PROPHET_DAILY), both off for hourly.
        ``shrink``: each step's predicted change is scaled by the calibration
        origins' least-squares slope of realised on predicted change, clipped
        to [0, 1], and the band is calibrated on the shrunk errors (needs
        ``calibrate``). ``flat``: a flat trend with weekly seasonality only.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'trend', 'volatility'].
    """
    sw = resolve_switches(interval, horizon, shrink, flat)
    out = prophet_paths(df, interval, horizon, calibrate=calibrate, anchor=anchor,
                        changepoint_prior_scale=changepoint_prior_scale,
                        extra_seasonality=extra_seasonality, flat=sw['flat'],
                        variants=(sw['shrink'],))[sw['shrink']]
    print("Prophet Prediction Complete.")
    return out
```
