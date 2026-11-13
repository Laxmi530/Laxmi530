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
import pmdarima as pm
import warnings

from model_utils import (
    N_SIMS as _N_SIMS,
    normalize_interval,
    coerce_numeric_columns,
    winsorize,
    wilder_rsi,
    select_best_garch,
    simulate_price_paths,
    generate_future_dates,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================
# 1. Preprocessing + Exogenous Features
# ==========================================
def _get_exog_cols(interval, calendar=True):
    """
    Return the ordered list of exogenous column names for ARIMAX.

    Parameters
    ----------
    interval : str
        'daily' or 'hourly'. Hourly adds 'hour' to the feature set.
    calendar : bool, optional
        True (default) keeps the calendar regressors (day of week, month and,
        hourly, hour) as plain numbers; False drops them (README section 114).

    Returns
    -------
    list of str
        Ordered list of exogenous column names.
    """
    cols = [
        'log_volume_ratio', 'range_hl', 'volatility',
        'rsi', 'momentum_5', 'dayofweek', 'month',
    ]
    if interval == 'hourly':
        cols.append('hour')
    if not calendar:
        cols = [c for c in cols if c not in _CALENDAR_EXOG]
    return cols


# Market-context regressors (from market_context). Added to the exogenous set
# only when present on the (enriched) df. Like every bar statistic they enter
# lagged one bar (_lagged_design): the contemporaneous market return would
# explain part of today's move in-sample, shrink the residual the GARCH band is
# fitted to, and then have to be guessed (as 0) at forecast time, so the band
# would omit the market's own variance.
_MARKET_EXOG = ['mkt_ret_1', 'vix_change']

# Calendar regressors describe the bar being forecast and are known in
# advance; every other regressor is a bar statistic and enters LAGGED.
_CALENDAR_EXOG = ('dayofweek', 'month', 'hour')


def _lagged_design(df, exog_cols):
    """
    Regression design for ARIMAX: return r_t on the bar statistics of t-1.

    Momentum, volatility, RSI, range and the market return at bar t all
    contain r_t itself, so regressing r_t on them is a contemporaneous fit: it
    explains part of r_t with r_t, shrinks the residual the GARCH layer sees
    (and with it the band), and at forecast time has nothing to feed but a
    guessed future value. Lagging them by one bar makes every regressor known
    at the origin, so step 1 is forecast from the actual last bar. Calendar
    columns are the target bar's own and stay unlagged.

    Returns
    -------
    (pd.Series, np.ndarray)
        Log returns from the second row on, and the matching design matrix.
    """
    X = df[exog_cols].copy()
    lag = [c for c in exog_cols if c not in _CALENDAR_EXOG]
    X[lag] = X[lag].shift(1)
    return df['log_return'].iloc[1:], X.iloc[1:].to_numpy(dtype=np.float64)


def prepare_data(df, interval, max_obs=None):
    """
    Compute log returns (target) and exogenous regressors for ARIMAX.

    Exogenous features include scale-invariant technical indicators
    (log volume ratio, high-low range, rolling volatility, RSI via
    Wilder smoothing, 5-period log momentum) and calendar variables
    (day of week, month, hour for hourly data). Target and continuous
    regressors are winsorized at 1st/99th percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime, open, high, low, close, adj_close, volume.
    interval : str
        'daily' or 'hourly'; determines whether 'hour' feature is included.
    max_obs : int, optional
        Maximum number of recent observations to keep. If None, use all data.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index, log_return, and exogenous columns.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    coerce_numeric_columns(df)

    df['log_return'] = np.log(df['adj_close'] / df['adj_close'].shift(1))

    vol_ma = df['volume'].rolling(20, min_periods=1).mean()
    df['log_volume_ratio'] = np.log(
        (df['volume'] / vol_ma.replace(0, 1)).clip(lower=0.01)
    )

    df['range_hl'] = (df['high'] - df['low']) / df['adj_close']

    df['volatility'] = df['log_return'].rolling(10, min_periods=1).std()

    df['rsi'] = wilder_rsi(df['adj_close'])

    df['momentum_5'] = np.log(df['adj_close'] / df['adj_close'].shift(5))

    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    if interval == 'hourly':
        df['hour'] = df.index.hour

    df = df.dropna(subset=['log_return'])
    df = df[np.isfinite(df['log_return'])]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if max_obs and len(df) > max_obs:
        df = df.iloc[-max_obs:]

    df['log_return'] = winsorize(df['log_return'].values)
    for col in ['log_volume_ratio', 'range_hl', 'volatility', 'momentum_5']:
        if col in df.columns:
            df[col] = winsorize(df[col].values)

    return df


# ==========================================
# 2. Train ARIMAX + GARCH
# ==========================================
def train_arimax_garch(df, exog_cols):
    """
    Fit ARIMAX on log returns with exogenous variables and GARCH on residuals.

    Uses auto_arima with exogenous regressors, d=0 (log returns are
    already stationary), and BIC for model selection. Caps total ARMA
    order at 10 to prevent overfitting. Selects the best GARCH variant
    from an expanded search space on ARIMAX residuals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'log_return' and exogenous columns (from prepare_data).
    exog_cols : list of str
        Ordered list of exogenous column names.

    Returns
    -------
    tuple of (ARIMA, ARCHModelResult)
        arimax_model : fitted pmdarima ARIMAX model.
        garch_model : fitted GARCH volatility model.
    """
    print("--- Training ARIMAX + GARCH ---")

    returns, exog = _lagged_design(df, exog_cols)

    print("Fitting Auto-ARIMAX with exogenous regressors...")
    # pmdarima >= 2.0 takes the regressors as ``X``. It silently swallowed the
    # old ``exogenous=`` keyword, so until this was fixed "ARIMAX" fitted no
    # regressors at all and was plain ARIMA on the same returns.
    arimax_model = pm.auto_arima(
        returns,
        X=exog,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        max_order=10,
        d=0,
        seasonal=False,
        information_criterion='bic',
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False,
    )
    print(f"Best ARIMAX Order: {arimax_model.order}")

    residuals = arimax_model.resid()

    print("Selecting best volatility model for residuals...")
    garch_model = select_best_garch(residuals)

    return arimax_model, garch_model


# ==========================================
# 3. Future Exogenous Generation
# ==========================================
def _build_future_exog(df, future_dates, exog_cols, interval, centre=False):
    """
    Construct the exogenous matrix for the forecast horizon.

    The bar statistics enter the regression lagged one bar
    (:func:`_lagged_design`), so the regressors for step 1 are the **actual**
    values at the last observed bar; later steps need values of bars not yet
    seen and decay from there toward a neutral level (weight ``0.9**(k-1)``
    at step k):

    - log_volume_ratio : decay toward 0.0 (= average volume, since log(1) = 0)
    - range_hl : decay toward its long-term median
    - volatility : decay toward its long-term median
    - rsi : decay toward 50 (neutral equilibrium)
    - momentum_5 : decay toward 0 (no trend assumption)
    - dayofweek : exact from calendar
    - month : exact from calendar
    - hour : exact from calendar (hourly only)
    - market regressors : decay toward 0

    Parameters
    ----------
    df : pd.DataFrame
        Historical data with exogenous columns (from prepare_data).
    future_dates : list of datetime
        Future dates for the forecast horizon.
    exog_cols : list of str
        Ordered list of exogenous column names.
    interval : str
        'daily' or 'hourly'; determines whether 'hour' is included.
    centre : bool, optional
        False (default): the neutral levels above. True: every bar statistic
        decays toward its mean over the training rows instead, the assumed
        future value of Hyndman and Athanasopoulos, Forecasting: Principles
        and Practice, section 10.3 (README section 114). With no intercept in
        the model, a neutral level that differs from the training mean adds
        a drift of the coefficient times the difference at every step.

    Returns
    -------
    np.ndarray
        Exogenous matrix of shape (horizon, len(exog_cols)) with
        dtype float64.
    """
    last = df.iloc[-1]
    long_term_range_hl = float(df['range_hl'].median())
    long_term_vol = float(df['volatility'].median())
    neutral = {'log_volume_ratio': 0.0, 'range_hl': long_term_range_hl,
               'volatility': long_term_vol, 'rsi': 50.0, 'momentum_5': 0.0}
    if centre:
        neutral = {c: float(df[c].mean()) for c in neutral}

    rows = []
    for i, dt in enumerate(future_dates):
        # Step 1 (i = 0) regresses on the last observed bar itself: weight 1.
        weight = 0.9 ** i

        row = {c: float(last[c]) * weight + neutral[c] * (1 - weight) for c in neutral}
        row['dayofweek'] = dt.dayofweek
        row['month'] = dt.month
        if interval == 'hourly':
            row['hour'] = dt.hour

        # Any extra exog (e.g. market-context regressors) starts from its last
        # observed value and decays to its expected future value of 0.
        for col in exog_cols:
            if col not in row:
                v = float(last[col]) if col in df.columns else 0.0
                m = float(df[col].mean()) if (centre and col in df.columns) else 0.0
                m = m if np.isfinite(m) else 0.0
                row[col] = (v if np.isfinite(v) else 0.0) * weight + m * (1 - weight)

        rows.append(row)

    return np.array(pd.DataFrame(rows)[exog_cols], dtype=np.float64)


# ==========================================
# 4. Monte Carlo Simulation
# ==========================================
def _simulate_price_paths(arimax_model, garch_model, last_price, horizon,
                          future_exog, method='simulation'):
    """
    Monte Carlo simulation of future price paths for ARIMAX+GARCH.

    Computes the ARIMAX deterministic mean forecast (conditioned on the
    estimated future exogenous variables) and delegates to the shared
    GARCH path simulator.

    Parameters
    ----------
    arimax_model : pmdarima.ARIMA
        Fitted ARIMAX model for mean return forecasts.
    garch_model : arch.univariate.base.ARCHModelResult
        Fitted GARCH volatility model (mean='Zero').
    last_price : float
        Last observed adjusted close price.
    horizon : int
        Number of steps to simulate.
    future_exog : np.ndarray
        Exogenous matrix of shape (horizon, n_features) for the ARIMAX
        mean forecast.

    Returns
    -------
    dict
        'predicted_price', 'lower_bound', 'upper_bound', 'volatility' arrays.
    """
    arimax_mean = arimax_model.predict(n_periods=horizon, X=future_exog)
    return simulate_price_paths(arimax_mean, garch_model, last_price, horizon,
                                method=method)


# ==========================================
# 5. Forecast Generation
# ==========================================
def generate_forecasts(df, arimax_model, garch_model, exog_cols, horizon,
                       interval, centre=False, method='simulation'):
    """
    Generate future price forecasts via Monte Carlo simulation.

    Simulates thousands of return paths through the fitted ARIMAX+GARCH
    dynamics conditioned on estimated future exogenous variables, then
    extracts empirical percentiles for point forecasts and 95% confidence
    intervals. This approach naturally respects fat tails, skewness,
    and path-dependent volatility clustering from the fitted GARCH
    innovation distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame with adj_close, exogenous columns, datetime index.
    arimax_model : pmdarima.ARIMA
        Fitted ARIMAX model.
    garch_model : arch.univariate.base.ARCHModelResult
        Fitted GARCH volatility model.
    exog_cols : list of str
        Ordered list of exogenous column names.
    horizon : int
        Number of steps to forecast.
    interval : str
        'daily' or 'hourly'; affects date generation and exogenous logic.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price', 'lower_bound',
        'upper_bound', 'volatility'].
    """
    last_price = df['adj_close'].iloc[-1]
    last_date = df.index[-1]

    future_dates = generate_future_dates(last_date, horizon, interval)
    future_exog = _build_future_exog(df, future_dates, exog_cols, interval, centre=centre)

    print(f"Simulating {_N_SIMS} price paths for {horizon} steps...")
    sim = _simulate_price_paths(
        arimax_model, garch_model, last_price, horizon, future_exog,
        method=method,
    )

    future_data = []
    for i in range(horizon):
        future_data.append({
            'datetime': future_dates[i],
            'predicted_price': sim['predicted_price'][i],
            'lower_bound': sim['lower_bound'][i],
            'upper_bound': sim['upper_bound'][i],
            'volatility': sim['volatility'][i],
        })

    return pd.DataFrame(future_data)


# ==========================================
# 6. Main Orchestrator
# ==========================================
def _train_no_exog(df):
    """ARIMA + GARCH on ARIMAX's prepared returns, with no regressors at all."""
    print("--- Training ARIMA + GARCH (ARIMAX data, no regressors) ---")
    model = pm.auto_arima(
        df['log_return'], start_p=0, start_q=0, max_p=5, max_q=5,
        max_order=10, d=0, seasonal=False, information_criterion='bic',
        stepwise=True, suppress_warnings=True, error_action='ignore',
        trace=False,
    )
    print(f"Best ARIMA Order: {model.order}")
    return model, select_best_garch(model.resid())


#: The configuration before section 114, which every run keeps unless a switch
#: is given: neutral future regressor levels, the calendar regressors as plain
#: numbers, and shocks drawn from the fitted distribution.
ARIMAX_LEGACY = dict(centre=False, calendar=True, fhs=False)


def resolve_switches(interval, horizon, centre=None, calendar=None, fhs=None):
    """
    The switches a run uses. A switch left as None takes ARIMAX_LEGACY's value;
    an explicit value is kept. ``interval`` and ``horizon`` are taken so a
    horizon-dependent default can be added without changing callers.
    """
    given = dict(centre=centre, calendar=calendar, fhs=fhs)
    return {k: ARIMAX_LEGACY[k] if v is None else bool(v) for k, v in given.items()}


def arimax_paths(df, interval, horizon, calendar=True, variants=((False, False),)):
    """
    Forecast paths for several (centre, fhs) settings from one fit:
    {(centre, fhs): DataFrame}. ``calendar`` changes the regressors, so it is
    fixed for the call; ``centre`` changes only the assumed future regressors
    and ``fhs`` only the simulated shocks.
    """
    interval = normalize_interval(interval)
    exog_cols = _get_exog_cols(interval, calendar=calendar)
    max_obs = 750 if interval == 'daily' else 500
    clean_df = prepare_data(df, interval, max_obs=max_obs)

    # Add market-context regressors when the df was enriched (graceful no-op
    # otherwise). prepare_data carries them through unchanged.
    market_exog = [c for c in _MARKET_EXOG if c in clean_df.columns]
    if market_exog:
        exog_cols = exog_cols + market_exog
        print(f"ARIMAX using market-context regressors: {market_exog}")

    arimax, garch = train_arimax_garch(clean_df, exog_cols)
    return {(bool(c), bool(f)): generate_forecasts(
                clean_df, arimax, garch, exog_cols, horizon, interval,
                centre=bool(c), method='bootstrap' if f else 'simulation')
            for c, f in variants}


def run_arimax_prediction(df: pd.DataFrame, interval: str, horizon: int,
                          use_exog=True, centre=None, calendar=None, fhs=None):
    """
    Main function to run ARIMAX + GARCH prediction.

    ``use_exog=False`` fits the same pipeline with no regressors. That is
    exactly what this function did before the regressors were passed to
    pmdarima correctly (README §59), so it reproduces every figure published
    for "ARIMAX" before then; the meta-ensemble uses it for that reason.

    Orchestrates exogenous feature engineering (including RSI via
    Wilder smoothing and scale-invariant technical indicators),
    ARIMAX model fitting with complexity cap, GARCH volatility
    modelling from an expanded candidate search, and Monte Carlo
    forecast generation with distribution-aware confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with columns datetime, open, high, low,
        close, adj_close, volume.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    use_exog : bool, optional
        False fits no regressors (see above). Its shocks are drawn from the
        fitted distribution unless ``fhs=True`` is given explicitly, whatever
        the defaults below, so the meta-ensemble's member does not change.
    centre, calendar, fhs : bool or None, optional
        None (default) takes ARIMAX_LEGACY (README section 114). ``centre``:
        future bar statistics decay toward their training means rather than
        fixed neutral levels. ``calendar=False`` drops the day-of-week, month
        and hour regressors. ``fhs``: shocks drawn from the model's own
        standardized residuals (filtered historical simulation).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    print(f"--- Starting ARIMAX Prediction ({interval}) ---")

    if not use_exog:
        method = 'bootstrap' if fhs else 'simulation'
        max_obs = 750 if interval == 'daily' else 500
        clean_df = prepare_data(df, interval, max_obs=max_obs)
        model, garch = _train_no_exog(clean_df)
        last_price = clean_df['adj_close'].iloc[-1]
        future_dates = generate_future_dates(clean_df.index[-1], horizon, interval)
        print(f"Simulating {_N_SIMS} price paths for {horizon} steps...")
        sim = simulate_price_paths(model.predict(n_periods=horizon), garch,
                                   last_price, horizon, method=method)
        return pd.DataFrame([{
            'datetime': future_dates[i],
            'predicted_price': sim['predicted_price'][i],
            'lower_bound': sim['lower_bound'][i],
            'upper_bound': sim['upper_bound'][i],
            'volatility': sim['volatility'][i],
        } for i in range(horizon)])

    sw = resolve_switches(interval, horizon, centre, calendar, fhs)
    v = (sw['centre'], sw['fhs'])
    results_df = arimax_paths(df, interval, horizon, calendar=sw['calendar'], variants=(v,))[v]
    print("ARIMAX Prediction Complete.")
    return results_df
```
