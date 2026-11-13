- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Cross-family meta-ensemble.

The multi-ticker backtest showed there is **no universal winner**: the tree/ML
models (Quantile-LGBM, XGBoost) won AAPL and MSFT, while the statistical models
(ARIMA/ARIMAX) won GOOGL, and the worst models (N-BEATS, Prophet) blew up by
3-10x on some tickers. The existing "Stacking Ensemble" only combines tree base
learners on the same features, so it inherits the tree family's ticker-specific
weaknesses.

This meta-ensemble instead combines **different model families** — two tree
models plus a statistical model (ARIMA+GARCH) — and aggregates them with the **median** of
the per-horizon point forecasts. The median is deliberately chosen over a mean
or a learned weighting: it is robust to a single constituent producing a wild
forecast, which is exactly the failure mode (one family misfiring on a given
ticker) we want to neutralise. Confidence bands are formed by averaging the
constituents' own intervals (all three emit calibrated bounds).

The result is a single forecast that tracks whichever family suits the current
ticker/regime without needing to know which in advance.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from model_utils import normalize_interval

#: How the members' forecasts are combined (README section 121).
#: ``point``: 'median' or 'mean' of the members' point forecasts per step.
#: ``band``: 'mean' (the average of the members' bounds, i.e. quantile
#: averaging), 'median' (the median of their bounds, as SCUM does for its
#: intervals) or 'pool' (the 2.5% and 97.5% quantiles of an equal-weight
#: mixture of the members' bands, each read as a normal in log price: a
#: linear pool, wider when the members disagree).
POINT_CHOICES = ('median', 'mean')
BAND_CHOICES = ('mean', 'median', 'pool')
#: The combination before section 121.
META_LEGACY = dict(point='median', band='mean')


def resolve_switches(interval, horizon, point=None, band=None):
    """
    The combination a run uses. A switch left as None takes META_LEGACY's
    value; an explicit value is kept. ``interval`` and ``horizon`` are taken
    so a default by bar size or horizon can be added without changing callers.
    """
    out = dict(point=META_LEGACY['point'] if point is None else str(point),
               band=META_LEGACY['band'] if band is None else str(band))
    if out['point'] not in POINT_CHOICES:
        raise ValueError(f"point must be one of {POINT_CHOICES}, not {out['point']!r}")
    if out['band'] not in BAND_CHOICES:
        raise ValueError(f"band must be one of {BAND_CHOICES}, not {out['band']!r}")
    return out


def pool_bounds(lower, upper, level=0.95, iters=80):
    """
    Per step, the central ``level`` interval of an equal-weight mixture of
    normals in log price, one per member, each centred on the middle of the
    member's log band with sd = half its log width / z. ``lower`` and
    ``upper`` are (members, steps) arrays of prices. Solved by bisection.
    """
    z = norm.ppf(0.5 + level / 2)
    lo, hi = np.log(np.asarray(lower, float)), np.log(np.asarray(upper, float))
    mu = (lo + hi) / 2
    sd = np.maximum((hi - lo) / (2 * z), 1e-12)

    def quantile(q):
        a = (mu - 10 * sd).min(axis=0)
        b = (mu + 10 * sd).max(axis=0)
        for _ in range(iters):
            m = (a + b) / 2
            below = norm.cdf((m[None, :] - mu) / sd).mean(axis=0) < q
            a, b = np.where(below, m, a), np.where(below, b, m)
        return np.exp((a + b) / 2)

    return quantile((1 - level) / 2), quantile((1 + level) / 2)

# Complementary, reliable constituents (tree x2 + statistical x1). Each emits
# 'predicted_price' plus calibrated 'lower_bound'/'upper_bound'.
#
# The statistical member is ARIMAX's pipeline WITHOUT regressors. pmdarima
# ignored ARIMAX's regressors (README §59), so the member every published
# meta-ensemble figure was measured with was ARIMA on ARIMAX's prepared
# returns; use_exog=False reproduces it exactly. With the regressors now
# actually fitted, ARIMAX measured ~3% worse RMSE than that on five NSE names
# (run_arimax_exog.py), so the ensemble keeps the model it was validated with
# rather than silently becoming a new one.
_CONSTITUENTS = [
    ('Quantile-LGBM', 'Quantile_model', 'run_quantile_prediction', {}),
    ('XGBoost', 'ML_model', 'run_ML_prediction', {'model_type': 'xgboost'}),
    ('ARIMA+GARCH', 'ARIMAX_model', 'run_arimax_prediction', {'use_exog': False}),
]


def _load(module_name, fn_name):
    """Import a constituent's run_* function lazily (avoids import cycles)."""
    import importlib
    return getattr(importlib.import_module(module_name), fn_name)


def run_constituents(df: pd.DataFrame, interval: str, horizon: int, xgb_kw=None):
    """
    Run each constituent and return ``[(name, output or None), ...]``.

    ``xgb_kw`` is passed to the XGBoost member on top of ``model_type``. None
    (default) passes nothing, so the member takes ``ML_model``'s defaults; since
    README section 101 those are larger leaves and a shrunk centre on daily bars
    beyond 13 sessions. ``ML_model.LGBM_LEGACY`` gives the member as it was
    before section 101 (README section 110). A constituent that raises gives
    None.
    """
    interval = normalize_interval(interval)
    outs = []
    for name, module_name, fn_name, kwargs in _CONSTITUENTS:
        if name == 'XGBoost' and xgb_kw:
            kwargs = dict(kwargs, **xgb_kw)
        try:
            fn = _load(module_name, fn_name)
            outs.append((name, fn(df, interval, horizon, **kwargs)))
        except Exception as exc:
            print(f"  meta-ensemble: '{name}' failed, skipping ({exc})")
            outs.append((name, None))
    return outs


def run_meta_ensemble_prediction(df: pd.DataFrame, interval: str, horizon: int,
                                 xgb_kw=None, point=None, band=None):
    """
    Median-combine complementary model families into one robust forecast.

    Runs each constituent (two tree models + one statistical model), aligns
    their forecasts by horizon step, and combines them with the per-step median
    for the point forecast and the per-step mean of the constituents' bounds for
    the interval. If a constituent raises or returns nothing usable, it is
    skipped, so the ensemble degrades gracefully rather than failing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with datetime + OHLCV columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    xgb_kw : dict, optional
        Extra switches for the XGBoost member; see :func:`run_constituents`.
    point, band : str, optional
        How the members are combined (:data:`POINT_CHOICES`,
        :data:`BAND_CHOICES`); None takes :func:`resolve_switches`'s default.

    Returns
    -------
    pd.DataFrame
        Columns ['datetime', 'predicted_price', 'lower_bound', 'upper_bound',
        'volatility']. Bound columns are present when at least one constituent
        supplied intervals.

    Raises
    ------
    RuntimeError
        If no constituent produced a usable forecast.
    """
    interval = normalize_interval(interval)
    print(f"--- Starting Meta-Ensemble Prediction ({interval}) ---")
    sw = resolve_switches(interval, horizon, point=point, band=band)
    return combine(run_constituents(df, interval, horizon, xgb_kw=xgb_kw), horizon, **sw)


def combine(outs, horizon, point='median', band='mean'):
    """
    Combine :func:`run_constituents`'s output per step: the ``point`` ('median'
    or 'mean') of the members' point forecasts, and the ``band`` ('mean',
    'median' or 'pool', see :data:`BAND_CHOICES`) of their bounds. The defaults
    are the combination before README section 121. Members that gave nothing
    usable are skipped.
    """
    if point not in POINT_CHOICES or band not in BAND_CHOICES:
        raise ValueError(f"point/band must be in {POINT_CHOICES}/{BAND_CHOICES}")
    point_stack = []
    lower_stack = []
    upper_stack = []
    datetimes = None
    used = []

    for name, out in outs:
        if (out is None or 'predicted_price' not in getattr(out, 'columns', [])
                or len(out) < horizon):
            print(f"  meta-ensemble: '{name}' gave no usable forecast, skipping")
            continue

        point_stack.append(out['predicted_price'].values[:horizon].astype(float))
        if datetimes is None:
            datetimes = out['datetime'].values[:horizon]
        if 'lower_bound' in out.columns and 'upper_bound' in out.columns:
            lower_stack.append(out['lower_bound'].values[:horizon].astype(float))
            upper_stack.append(out['upper_bound'].values[:horizon].astype(float))
        used.append(name)

    if not point_stack:
        raise RuntimeError("Meta-ensemble: no constituent produced a forecast.")

    print(f"Meta-ensemble combining {len(used)} models via {point} (band: {band}): {used}")

    stack = np.vstack(point_stack)
    median_price = np.median(stack, axis=0) if point == 'median' else np.mean(stack, axis=0)
    result = {'datetime': datetimes, 'predicted_price': median_price}

    if lower_stack:
        lows, ups = np.vstack(lower_stack), np.vstack(upper_stack)
        if band == 'mean':
            lower, upper = np.mean(lows, axis=0), np.mean(ups, axis=0)
        elif band == 'median':
            lower, upper = np.median(lows, axis=0), np.median(ups, axis=0)
        else:
            lower, upper = pool_bounds(lows, ups)
        # Guard ordering against any averaging artefact.
        lower = np.minimum(lower, median_price)
        upper = np.maximum(upper, median_price)
        result['lower_bound'] = lower
        result['upper_bound'] = upper
        result['volatility'] = (upper - lower) / (2 * 1.96)

    print("Meta-Ensemble Prediction Complete.")
    return pd.DataFrame(result)
```
