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
from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)
import warnings

from model_utils import (
    N_SIMS as _N_SIMS,
    normalize_interval,
    prepare_log_returns,
    generate_future_dates,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 1. Fit Markov Switching Model
# ==========================================
#: Seeds for statsmodels' random restart search, tried in turn (see below).
_SEARCH_SEEDS = (12345, 2024, 7)


def _fit_markov_switching(returns, k_regimes=2, order=2):
    """
    Fit a Markov Switching Autoregression on log returns.

    Tries the requested order first, then falls back to simpler
    specifications if optimisation fails. The model captures
    regime-dependent means and variances (e.g., low-volatility
    vs high-volatility regimes), which is critical for financial
    time series that exhibit bull/bear market transitions.

    Parameters
    ----------
    returns : pd.Series
        Log return series.
    k_regimes : int, optional
        Number of hidden regimes. Default 2.
    order : int, optional
        Autoregressive order within each regime. Default 2.

    Returns
    -------
    statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregressionResultsWrapper
        Fitted model results.

    Raises
    ------
    RuntimeError
        If all attempted specifications fail to converge.
    """
    ret_values = returns.values.astype(np.float64)

    # statsmodels draws the random restarts of ``search_reps`` from NumPy's
    # GLOBAL random state, so the fit depended on whatever ran before it: on
    # one Reliance series it failed outright under 1 of 30 states and settled
    # on AR(2) or AR(1) depending on the state (README section 68). Fixed seeds
    # make it reproducible, the caller's state is restored, and a specification
    # is retried with other seeds and then without random restarts before the
    # model is simplified.
    saved = np.random.get_state()
    try:
        for ar_order in [order, 1, 0]:
            mod = MarkovAutoregression(
                ret_values,
                k_regimes=k_regimes,
                order=ar_order,
                switching_variance=True,
            )
            for seed, reps in [(s, 20) for s in _SEARCH_SEEDS] + [(None, 0)]:
                try:
                    if seed is not None:
                        np.random.seed(seed)
                    result = mod.fit(maxiter=200, disp=False, search_reps=reps)
                    print(
                        f"Markov Switching AR({ar_order}) with {k_regimes} regimes "
                        f"converged (LL={result.llf:.2f})"
                    )
                    return result
                except Exception as e:
                    last = e
            print(f"  AR({ar_order}) failed: {last}, trying simpler...")
    finally:
        np.random.set_state(saved)

    raise RuntimeError("All Markov Switching specifications failed")


def regime_parameters(ms_result):
    """
    Each regime's mean and standard deviation of returns, from the fit.

    The model is fitted on a NumPy array, so ``ms_result.params`` is an array,
    not a Series, and looking it up by name (``params['sigma2[0]']``) always
    failed. The old code caught that and fell back to a mean of 0 and a
    standard deviation of 1% for every regime of every stock, so the served
    range never used the fit (README section 88). Parameters are looked up by
    position in ``model.param_names``, and a missing one is an error, not a
    silent default.
    """
    names = list(ms_result.model.param_names)
    values = np.asarray(ms_result.params, dtype=np.float64)
    k = ms_result.k_regimes

    def get(name):
        if name not in names:
            raise RuntimeError(f"Markov switching fit has no parameter {name!r}")
        return float(values[names.index(name)])

    means = np.array([get(f'const[{i}]') for i in range(k)])
    var = np.array([get(f'sigma2[{i}]') for i in range(k)])
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(var)) or np.any(var <= 0):
        raise RuntimeError("Markov switching fit gave a non-finite or non-positive variance")
    return means, np.sqrt(var)


# ==========================================
# 3. Monte Carlo Simulation with Regime Switching
# ==========================================
def _simulate_regime_paths(ms_result, last_price, horizon):
    """
    Monte Carlo simulation of future price paths under regime dynamics.

    For each simulation path:
    1. Start from the current smoothed regime probabilities.
    2. At each step, sample the next regime from the transition matrix.
    3. Draw a return from the regime-specific distribution
       (regime mean + regime std * N(0,1)).
    4. Accumulate log returns to build a price path.

    This naturally produces wider confidence intervals during
    high-volatility regimes and tighter ones during calm periods.

    Parameters
    ----------
    ms_result : MarkovAutoregressionResultsWrapper
        Fitted Markov Switching model.
    last_price : float
        Last observed adjusted close price.
    horizon : int
        Number of future steps to simulate.

    Returns
    -------
    dict
        'predicted_price' : np.ndarray (horizon,) — median price.
        'lower_bound'     : np.ndarray (horizon,) — 2.5th percentile.
        'upper_bound'     : np.ndarray (horizon,) — 97.5th percentile.
        'volatility'      : np.ndarray (horizon,) — std of simulated returns.
    """
    # A local generator: seeding NumPy's global state here changed what every
    # model run after this one drew (README section 68).
    rng = np.random.default_rng(42)

    k_regimes = ms_result.k_regimes
    regime_means, regime_stds = regime_parameters(ms_result)

    raw_tm = np.asarray(ms_result.regime_transition, dtype=np.float64)
    if raw_tm.ndim == 1:
        transition_matrix = raw_tm.reshape(k_regimes, k_regimes)
    else:
        transition_matrix = raw_tm
    if transition_matrix.shape[0] != k_regimes:
        transition_matrix = transition_matrix.T

    smoothed = ms_result.smoothed_marginal_probabilities
    if hasattr(smoothed, 'iloc'):
        current_probs = np.asarray(
            smoothed.iloc[-1], dtype=np.float64,
        ).flatten()
    else:
        current_probs = np.asarray(smoothed[-1], dtype=np.float64).flatten()

    current_probs = np.maximum(current_probs, 0.0)
    cp_sum = current_probs.sum()
    if cp_sum > 0:
        current_probs = current_probs / cp_sum
    else:
        current_probs = np.ones(k_regimes) / k_regimes

    price_paths = np.zeros((horizon, _N_SIMS))
    all_returns = np.zeros((horizon, _N_SIMS))

    for s in range(_N_SIMS):
        regime = rng.choice(k_regimes, p=current_probs)
        cum_return = 0.0

        for t in range(horizon):
            ret = regime_means[regime] + regime_stds[regime] * rng.standard_normal()
            all_returns[t, s] = ret
            cum_return += ret
            price_paths[t, s] = last_price * np.exp(cum_return)

            trans_probs = np.asarray(
                transition_matrix[:, regime], dtype=np.float64,
            ).flatten()
            trans_probs = np.maximum(trans_probs, 0.0)
            total = trans_probs.sum()
            if total > 0:
                trans_probs = trans_probs / total
            else:
                trans_probs = np.ones(k_regimes) / k_regimes
            regime = rng.choice(k_regimes, p=trans_probs)

    valid = np.all(np.isfinite(price_paths) & (price_paths > 0), axis=0)
    if valid.sum() >= 100:
        price_paths = price_paths[:, valid]
        all_returns = all_returns[:, valid]

    return {
        'predicted_price': np.median(price_paths, axis=1),
        'lower_bound': np.percentile(price_paths, 2.5, axis=1),
        'upper_bound': np.percentile(price_paths, 97.5, axis=1),
        'volatility': np.std(all_returns, axis=1),
    }


# ==========================================
# 4. Generate Forecasts
# ==========================================
def _generate_forecasts(df, ms_result, horizon, interval):
    """
    Generate future price forecasts with regime-aware confidence intervals.

    Combines the fitted Markov Switching model with Monte Carlo
    simulation, then builds the output DataFrame with business-day
    or market-hour date generation.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame with adj_close and datetime index.
    ms_result : MarkovAutoregressionResultsWrapper
        Fitted Markov Switching model.
    horizon : int
        Forecast steps.
    interval : str
        'daily' or 'hourly'.

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'predicted_price', 'lower_bound',
        'upper_bound', 'volatility'].
    """
    last_price = float(df['adj_close'].iloc[-1])
    last_date = df.index[-1]

    print(f"Simulating {_N_SIMS} regime-switching paths for {horizon} steps...")
    sim = _simulate_regime_paths(ms_result, last_price, horizon)

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
# 5. Main Orchestrator
# ==========================================
def run_regime_prediction(df, interval, horizon):
    """
    Main function to run Markov Regime Switching prediction.

    Fits a Markov Switching Autoregression that identifies hidden
    market regimes (e.g., bull/low-volatility vs bear/high-volatility)
    with regime-dependent means and variances. Forecasts are generated
    via Monte Carlo simulation that respects the regime transition
    probabilities, naturally widening confidence intervals in
    uncertain or volatile regimes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'predicted_price', 'lower_bound',
        'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    print(f"--- Starting Regime Switching Prediction ({interval}) ---")

    max_obs = 750 if interval == 'daily' else 500
    clean_df = prepare_log_returns(df, max_obs=max_obs)

    ms_result = _fit_markov_switching(clean_df['log_return'])

    results_df = _generate_forecasts(clean_df, ms_result, horizon, interval)

    print("Regime Switching Prediction Complete.")
    return results_df
```
