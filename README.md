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
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from model_utils import normalize_interval, generate_future_dates, lgb_threads
from ML_model import create_features
from direct_forecast import (
    horizon_feature_cols, make_horizon_targets, training_frame,
    compute_origin_row, conformal_calibration,
)

_QUANTILES = [0.025, 0.5, 0.975]
_ALPHA = 0.05

#: Every switch off: the model as validated before README section 111, which
#: every run keeps unless a switch is passed (section 111 tests the switches).
QUANTILE_LEGACY = dict(leaf_scale=False, shrink=False, cqr=False)


# ==========================================
# 1. Build Quantile LightGBM Models
# ==========================================
def _build_quantile_lgb(alpha, min_child_samples=20):
    """
    Create a LightGBM regressor configured for quantile loss.

    Parameters
    ----------
    alpha : float
        Target quantile in (0, 1). For example 0.5 gives the median,
        0.025 gives the 2.5th percentile (lower bound of 95 % CI).
    min_child_samples : int, optional
        Minimum rows per leaf. Default 20 (the shipped model).

    Returns
    -------
    lgb.LGBMRegressor
        Configured but unfitted LightGBM model.
    """
    return lgb.LGBMRegressor(
        objective='quantile',
        alpha=alpha,
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=int(min_child_samples),
        random_state=42,
        verbose=-1,
        **lgb_threads(),
    )


# ==========================================
# 2. Train Three Quantile Models
# ==========================================
def train_quantile_models(features_df, target_col, test_size=0.1,
                          min_child_samples=20):
    """
    Train three LightGBM models for the 2.5th, 50th, and 97.5th quantiles.

    Each model independently optimises quantile loss (pinball loss) for
    its target percentile.  A temporal train/test split avoids look-ahead
    bias, and early stopping prevents overfitting.

    Parameters
    ----------
    features_df : pd.DataFrame
        Engineered features DataFrame (datetime-indexed) with a target
        column and raw OHLCV columns.
    target_col : str
        Name of the target column (e.g., 'target_log_return').
    test_size : float, optional
        Fraction of data for temporal validation. Default 0.1.
    min_child_samples : int, optional
        Minimum rows per leaf for all three models. Default 20.

    Returns
    -------
    models : dict
        Mapping of quantile float -> fitted LGBMRegressor.
    feature_cols : list of str
        Ordered list of feature column names used during training.
    """
    raw_cols = {'open', 'high', 'low', 'close', 'adj_close', 'volume'}
    feature_cols = [
        c for c in features_df.columns
        if c != target_col and c not in raw_cols
    ]

    X = features_df[feature_cols]
    y = features_df[target_col]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {}

    for q in _QUANTILES:
        label = f"Q{q:.3f}"
        print(f"Training LightGBM {label}...")

        model = _build_quantile_lgb(q, min_child_samples)
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ]
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='quantile',
            callbacks=callbacks,
        )
        models[q] = model

    print(f"Quantile training complete (features: {len(feature_cols)})")
    return models, feature_cols


# ==========================================
# 3. Switches (README section 111)
# ==========================================
def resolve_switches(interval, horizon, leaf_scale=None, shrink=None, cqr=None):
    """
    The switches a run uses. A switch left as None takes QUANTILE_LEGACY (all
    off); an explicit True or False is kept. ``interval`` and ``horizon`` are
    taken so that a horizon-split default can be added here if one ships.
    """
    given = dict(leaf_scale=leaf_scale, shrink=shrink, cqr=cqr)
    return {k: QUANTILE_LEGACY[k] if v is None else bool(v) for k, v in given.items()}


def min_leaf(h, n_rows):
    """``leaf_scale``'s rows per leaf: at least min(20 h, a fifth of the rows)."""
    return int(max(20, min(20 * h, n_rows // 5)))


def adjust_quantiles(q_origin, cal, shrink, cqr, alpha=_ALPHA):
    """
    One step's (lower, median, upper) cumulative log returns after the switches.

    ``q_origin`` holds the three models' forecasts at the origin; ``cal`` is
    ``(y, P)`` from the calibration block (realised returns and an (m, 3)
    array of the calibration models' quantiles), or None. Both are sorted, so
    the quantiles never cross (the rearrangement of Chernozhukov,
    Fernandez-Val and Galichon, Econometrica 2010).

    * ``shrink``: the three quantiles move together by -(1 - beta) times the
      median, beta = clip(sum(p y) / sum(p p), 0, 1) from the calibration
      block's medians p (as ``ML_model``'s shrink; Campbell and Thompson 2008).
    * ``cqr``: conformalized quantile regression (Romano, Patterson and
      Candes, NeurIPS 2019): the scores E = max(lo - y, y - hi) on the
      calibration block, Q their finite-sample (1 - alpha) quantile, and the
      band [lo - Q, hi + Q]. Q is negative when the raw band over-covers, so
      the band can also narrow; it never crosses the median.

    With both off, or no calibration block, the sorted quantiles are returned
    unchanged.
    """
    lo, mid, hi = (float(v) for v in np.sort(np.asarray(q_origin, dtype=float)))
    if cal is None or not (shrink or cqr):
        return lo, mid, hi
    y, P = cal
    y = np.asarray(y, dtype=float)
    P = np.sort(np.asarray(P, dtype=float), axis=1)
    if shrink:
        p = P[:, 1]
        den = float(np.sum(p * p))
        beta = float(np.clip(np.sum(p * y) / den, 0.0, 1.0)) if den > 0 else 0.0
        shift = (1.0 - beta) * mid
        lo, mid, hi = lo - shift, mid - shift, hi - shift
        P = P - ((1.0 - beta) * p)[:, None]
    if cqr:
        E = np.maximum(P[:, 0] - y, y - P[:, 2])
        m = len(E)
        q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
        Q = float(np.quantile(E, q_level))
        lo, hi = min(lo - Q, mid), max(hi + Q, mid)
    return lo, mid, hi


def quantile_paths(df, interval, horizon, leaf_scale=False, variants=((False, False),)):
    """
    Forecasts for several ``(shrink, cqr)`` settings from one set of fits.

    Returns ``{(shrink, cqr): DataFrame}``. The calibration models (fit on the
    training rows before the most recent 20%, with ``h`` rows purged between,
    as ``direct_forecast.conformal_calibration``) are trained only when a
    variant needs them. ``run_quantile_prediction`` is this with one variant.
    """
    interval = normalize_interval(interval)
    variants = [(bool(s), bool(c)) for s, c in variants]

    df = df.copy()
    max_obs = 1000 if interval == 'daily' else 750
    if len(df) > max_obs:
        df = df.iloc[-max_obs:]

    features_df = create_features(df, interval)
    feature_cols = horizon_feature_cols(features_df)
    targets = make_horizon_targets(features_df, horizon)

    origin_X, last_price, last_date = compute_origin_row(
        df, feature_cols, interval,
    )
    future_dates = generate_future_dates(last_date, horizon, interval)
    need_cal = any(s or c for s, c in variants)

    print(f"Training {horizon} direct quantile horizon models...")

    rows = {v: [] for v in variants}
    for i, h in enumerate(range(1, horizon + 1)):
        target_name = f'target_h{h}'
        train_df = training_frame(
            features_df, feature_cols, targets[h], target_name,
        )
        mcs = min_leaf(h, len(train_df)) if leaf_scale else 20
        models, _ = train_quantile_models(train_df, target_name, min_child_samples=mcs)
        q_origin = [float(models[q].predict(origin_X)[0]) for q in _QUANTILES]

        cal = None
        if need_cal:
            def _fit_predict(proper_df, X_cal, _tn=target_name, _mcs=mcs):
                m, _ = train_quantile_models(proper_df, _tn, min_child_samples=_mcs)
                return np.column_stack([m[q].predict(X_cal) for q in _QUANTILES])
            cal = conformal_calibration(
                train_df, feature_cols, target_name, _fit_predict, embargo=h,
            )

        for v in variants:
            lo, mid, hi = adjust_quantiles(q_origin, cal, *v)
            lower_price, median_price, upper_price = (
                float(last_price * np.exp(x)) for x in (lo, mid, hi))
            rows[v].append({
                'datetime': future_dates[i],
                'predicted_price': median_price,
                'lower_bound': lower_price,
                'upper_bound': upper_price,
                'volatility': float((upper_price - lower_price) / (2 * 1.96)),
            })

    return {v: pd.DataFrame(r) for v, r in rows.items()}


# ==========================================
# 4. Main Orchestrator (Direct Multi-Horizon)
# ==========================================
def run_quantile_prediction(df, interval, horizon, leaf_scale=None, shrink=None,
                            cqr=None):
    """
    Main function to run quantile forecasting with LightGBM.

    Uses **direct multi-horizon** forecasting: for each step ``h`` three
    LightGBM models target the 2.5th / 50th / 97.5th quantiles of the
    *cumulative* log return ``log(P[t+h]/P[t])``, forecast directly from the
    observed features at the last bar. This yields native probabilistic
    intervals whose width grows naturally with horizon, and avoids the
    recursive error accumulation (and one-bar origin offset) of the previous
    step-by-step scheme.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with datetime, OHLCV columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps.
    leaf_scale, shrink, cqr : bool or None, optional
        README section 111's switches; None takes :func:`resolve_switches`'s
        default (all off, ``QUANTILE_LEGACY``). ``leaf_scale``: at least
        min(20 h, a fifth of the rows) per leaf (:func:`min_leaf`). ``shrink``
        and ``cqr``: see :func:`adjust_quantiles`.

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'predicted_price', 'lower_bound',
        'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    print(f"--- Starting Quantile Prediction ({interval}) ---")
    sw = resolve_switches(interval, horizon, leaf_scale, shrink, cqr)
    v = (sw['shrink'], sw['cqr'])
    out = quantile_paths(df, interval, horizon, leaf_scale=sw['leaf_scale'], variants=[v])[v]
    print("Quantile Prediction Complete.")
    return out
```
