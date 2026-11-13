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
    prepare_log_returns,
    winsorize,
    select_best_garch,
    simulate_price_paths,
    generate_future_dates,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================
# 1. Preprocessing (Stationarity + Cleaning)
# ==========================================
def prepare_data_statistical(df, max_obs=None):
    """
    Prepare data for ARIMA/GARCH: compute log returns and clean outliers.

    ARIMA/GARCH work best on returns, not raw prices. Applies winsorization
    at 1st/99th percentiles for robustness and optionally limits to recent
    observations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'datetime' and OHLCV columns including 'adj_close'.
    max_obs : int, optional
        Maximum number of recent observations to keep. If None, use all data.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index, log_return column, and winsorized values.
    """
    df = prepare_log_returns(df, max_obs=max_obs)
    df['log_return'] = winsorize(df['log_return'].values)
    return df


# ==========================================
# 2. Train Hybrid ARIMA-GARCH
# ==========================================
def train_arima_garch(df):
    """
    Fit ARIMA on log returns and GARCH on residuals.

    Uses auto_arima with d=0 (log returns are stationary) and BIC for
    model selection. Caps total ARMA order at 10 to prevent overfitting.
    Selects the best GARCH variant on ARIMA residuals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'log_return' column (from prepare_data_statistical).

    Returns
    -------
    tuple of (ARIMA, ARCHModelResult)
        arima_model : fitted pmdarima ARIMA model.
        garch_model : fitted GARCH volatility model.
    """
    print("--- Training Statistical Models ---")

    returns = df['log_return']

    print("Fitting Auto-ARIMA to find best parameters...")
    arima_model = pm.auto_arima(
        returns,
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
    print(f"Best ARIMA Order: {arima_model.order}")

    arima_residuals = arima_model.resid()

    print("Selecting best volatility model...")
    garch_model = select_best_garch(arima_residuals)

    return arima_model, garch_model


# ==========================================
# 3. Monte Carlo Simulation
# ==========================================
def _simulate_price_paths(arima_model, garch_model, last_price, horizon,
                          method='simulation'):
    """
    Monte Carlo simulation of future price paths.

    Computes the ARIMA deterministic mean forecast and delegates to the
    shared GARCH path simulator, which combines it with GARCH-simulated
    residual paths to produce realistic price distributions.

    Parameters
    ----------
    arima_model : pmdarima.ARIMA
        Fitted ARIMA model for mean return forecasts.
    garch_model : arch.univariate.base.ARCHModelResult
        Fitted GARCH volatility model (mean='Zero').
    last_price : float
        Last observed adjusted close price.
    horizon : int
        Number of steps to simulate.
    method : str, optional
        'simulation' (default) or 'bootstrap' (filtered historical
        simulation); see :func:`model_utils.simulate_price_paths`.

    Returns
    -------
    dict
        'predicted_price', 'lower_bound', 'upper_bound', 'volatility' arrays.
    """
    arima_mean = arima_model.predict(n_periods=horizon)
    return simulate_price_paths(arima_mean, garch_model, last_price, horizon,
                                method=method)


# ==========================================
# 4. Forecast Generation
# ==========================================
def generate_forecasts(df, arima_model, garch_model, horizon, interval,
                       method='simulation'):
    """
    Generate future price forecasts via Monte Carlo simulation.

    Simulates thousands of return paths through the fitted ARIMA+GARCH
    dynamics, then extracts empirical percentiles for point forecasts
    and confidence intervals. This approach naturally respects fat tails,
    skewness, and path-dependent volatility clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame with 'adj_close' and datetime index.
    arima_model : pmdarima.ARIMA
        Fitted ARIMA model.
    garch_model : arch.univariate.base.ARCHModelResult
        Fitted GARCH volatility model.
    horizon : int
        Number of steps to forecast.
    interval : str
        'daily' or 'hourly'; affects date increment and market-hours logic.
    method : str, optional
        'simulation' (default) or 'bootstrap' (filtered historical simulation).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price', 'lower_bound',
        'upper_bound', 'volatility'].
    """
    last_price = df['adj_close'].iloc[-1]
    last_date = df.index[-1]

    print(f"Simulating {_N_SIMS} price paths for {horizon} steps...")
    sim = _simulate_price_paths(arima_model, garch_model, last_price, horizon,
                                method=method)

    future_dates = generate_future_dates(last_date, horizon, interval)

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
# 5. Main Orchestrator (Call this from UI)
# ==========================================
#: The configuration before section 114, which every run keeps unless a switch
#: is given: shocks drawn from the fitted distribution.
ARIMA_LEGACY = dict(fhs=False)


def resolve_switches(interval, horizon, fhs=None):
    """
    The switches a run uses. A switch left as None takes ARIMA_LEGACY's value;
    an explicit value is kept. ``interval`` and ``horizon`` are taken so a
    horizon-dependent default can be added without changing callers.
    """
    return {'fhs': ARIMA_LEGACY['fhs'] if fhs is None else bool(fhs)}


def arima_paths(df, interval, horizon, variants=(False,)):
    """
    Forecast paths for several ``fhs`` settings from one fit: {fhs: DataFrame}.
    The fit does not depend on ``fhs``; only the simulated shocks do.
    """
    interval = normalize_interval(interval)
    max_obs = 750 if interval == 'daily' else 500
    clean_df = prepare_data_statistical(df, max_obs=max_obs)
    arima, garch = train_arima_garch(clean_df)
    return {bool(f): generate_forecasts(clean_df, arima, garch, horizon, interval,
                                        method='bootstrap' if f else 'simulation')
            for f in variants}


def run_statistical_prediction(df: pd.DataFrame, interval: str, horizon: int,
                               fhs=None):
    """
    Main function to run ARIMA+GARCH prediction.

    Orchestrates data preparation, model training, and Monte Carlo
    forecast generation with distribution-aware confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    fhs : bool or None, optional
        None (default) takes ARIMA_LEGACY (README section 114): shocks drawn
        from the fitted distribution. True draws them from the model's own
        standardized residuals (filtered historical simulation).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    sw = resolve_switches(interval, horizon, fhs)
    print(f"--- Starting Statistical Prediction ({interval}) ---")
    results_df = arima_paths(df, interval, horizon, variants=(sw['fhs'],))[sw['fhs']]
    print("Prediction Complete.")
    return results_df

```
