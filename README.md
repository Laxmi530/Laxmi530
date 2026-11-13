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
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from model_utils import normalize_interval, generate_future_dates, lgb_threads, xgb_threads
from ML_model import create_features
from direct_forecast import (
    horizon_feature_cols, make_horizon_targets, training_frame,
    compute_origin_row, conformal_calibration, conformal_quantile,
)


# ==========================================
# 0. Switches (README section 113)
# ==========================================
#: The combiners the stack can use. 'ridge' is the shipped Ridge(alpha=1) with an
#: intercept. 'nnls' is least squares with non-negative weights and no intercept
#: (Breiman, 'Stacked regressions', Machine Learning 1996). 'mean' is the plain
#: average of the three members (the forecast combination puzzle: Smith and
#: Wallis 2009; Claeskens, Magnus, Vasnev and Wang 2016).
META_CHOICES = ('ridge', 'nnls', 'mean')
#: The configuration before section 113, which every run up to
#: gating.MAX_DIRECTIONAL_SESSIONS (and every hourly run) keeps.
STACK_LEGACY = dict(meta='ridge', shrink=False, leaf_scale=False, purge=False)
#: Daily bars beyond gating.MAX_DIRECTIONAL_SESSIONS: the Ridge combiner with a
#: centre shrunk toward no change, larger leaves and purged chunks. Chosen on 32
#: forecasts and confirmed on 72 fresh ones before the reserved period
#: (run_stack_improve.py, README section 113).
STACK_LONG = dict(meta='ridge', shrink=True, leaf_scale=True, purge=True)


def resolve_switches(interval, horizon, meta=None, shrink=None, leaf_scale=None,
                     purge=None):
    """
    The switches a run uses. A switch left as None takes the default: STACK_LONG
    on daily bars beyond gating.MAX_DIRECTIONAL_SESSIONS, otherwise
    STACK_LEGACY. An explicit value is kept.
    """
    from gating import MAX_DIRECTIONAL_SESSIONS
    long = normalize_interval(interval) == 'daily' and int(horizon) > MAX_DIRECTIONAL_SESSIONS
    base = STACK_LONG if long else STACK_LEGACY
    given = dict(meta=meta, shrink=shrink, leaf_scale=leaf_scale, purge=purge)
    out = {k: base[k] if v is None else v for k, v in given.items()}
    out['meta'] = str(out['meta'])
    if out['meta'] not in META_CHOICES:
        raise ValueError(f"meta must be one of {META_CHOICES}, not {out['meta']!r}")
    for k in ('shrink', 'leaf_scale', 'purge'):
        out[k] = bool(out[k])
    return out


def min_leaf(h, n_rows):
    """Rows per leaf with leaf_scale: min(20 h, a fifth of the rows), at least 20
    (as LightGBM's and XGBoost's section 97 and 101 setting)."""
    return int(max(20, min(20 * h, n_rows // 5)))


# ==========================================
# 1. Base Model Definitions
# ==========================================
def _get_base_models(min_leaf_rows=None):
    """
    Create the three base learners for the stacking ensemble.

    Returns XGBoost (pseudo-Huber loss), LightGBM (Huber loss), and
    Ridge regression. All gradient-boosting models are configured with
    early-stopping-compatible settings.

    Parameters
    ----------
    min_leaf_rows : int, optional
        None (default) keeps XGBoost's min_child_weight=5 and LightGBM's
        min_child_samples=20; a number sets both (the leaf_scale switch).

    Returns
    -------
    list of tuple
        Each tuple is (name: str, model: estimator, engine: str).
    """
    xgb_model = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5 if min_leaf_rows is None else min_leaf_rows,
        random_state=42,
        early_stopping_rounds=50,
        **xgb_threads(),
    )
    lgb_model = lgb.LGBMRegressor(
        objective='huber',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20 if min_leaf_rows is None else min_leaf_rows,
        random_state=42,
        verbose=-1,
        **lgb_threads(),
    )
    ridge_model = Ridge(alpha=1.0)

    return [
        ('xgboost', xgb_model, 'xgboost'),
        ('lightgbm', lgb_model, 'lightgbm'),
        ('ridge', ridge_model, 'ridge'),
    ]


def _fit_member(model, engine, X, y, X_es, y_es):
    """Fit one member; the tree models stop early on (X_es, y_es)."""
    if engine == 'xgboost':
        model.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
    elif engine == 'lightgbm':
        model.fit(
            X, y,
            eval_set=[(X_es, y_es)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    else:
        model.fit(X, y)
    return model


# ==========================================
# 2. Out-of-Fold Prediction Generation
# ==========================================
def _train_and_predict_oof(features_df, feature_cols, target_col,
                           min_leaf_rows=None, purge=0):
    """
    Train base models and generate out-of-fold meta-features.

    Splits data temporally into three chunks:
    - 60 % for base model training
    - 20 % for meta-feature generation (OOF predictions)
    - 20 % held out as final validation

    Each base model is trained on the first chunk, then predicts on
    the second chunk.  The stacked OOF predictions become meta-features
    for the meta-learner.

    Parameters
    ----------
    features_df : pd.DataFrame
        Engineered features with datetime index.
    feature_cols : list of str
        Ordered list of feature column names.
    target_col : str
        Target column name in features_df.
    min_leaf_rows : int, optional
        Rows per leaf for the tree members (see :func:`_get_base_models`).
    purge : int, optional
        Rows dropped from the end of each training chunk (the first chunk, and
        the first two chunks for the refit), so that no training label's h-step
        window reaches into the block after it. 0 (default) drops none.

    Returns
    -------
    meta_X : np.ndarray
        (n_oof, 3) stacked OOF predictions from base models.
    meta_y : np.ndarray
        (n_oof,) true target values for OOF portion.
    trained_base : list of tuple
        (name, fitted_model, engine) for each base model, retrained
        on the combined train+OOF data for final prediction.
    feature_cols : list of str
        Feature columns used.
    val_mae : dict
        Base-model-name -> MAE on the final validation chunk.
    """
    X_all = features_df[feature_cols].values
    y_all = features_df[target_col].values

    n = len(X_all)
    split_1 = int(n * 0.6)
    split_2 = int(n * 0.8)
    # The purge never leaves fewer than 50 rows to train on.
    cut_1 = split_1 - purge if split_1 - purge >= 50 else split_1
    cut_2 = split_2 - purge if split_2 - purge >= 50 else split_2

    X_base = X_all[:cut_1]
    y_base = y_all[:cut_1]
    X_oof = X_all[split_1:split_2]
    y_oof = y_all[split_1:split_2]
    X_val = X_all[split_2:]
    y_val = y_all[split_2:]

    oof_preds = []
    val_mae = {}

    for name, model, engine in _get_base_models(min_leaf_rows):
        _fit_member(model, engine, X_base, y_base, X_oof, y_oof)
        oof_preds.append(model.predict(X_oof))
        pred_val = model.predict(X_val)
        val_mae[name] = mean_absolute_error(y_val, pred_val)
        print(f"  {name} val MAE: {val_mae[name]:.6f}")

    meta_X = np.column_stack(oof_preds)
    meta_y = y_oof

    retrained = []
    X_retrain = X_all[:cut_2]
    y_retrain = y_all[:cut_2]

    for name, m, engine in _get_base_models(min_leaf_rows):
        retrained.append((name, _fit_member(m, engine, X_retrain, y_retrain, X_val, y_val), engine))

    return meta_X, meta_y, retrained, feature_cols, val_mae


# ==========================================
# 3. Meta-Learner Training
# ==========================================
def _train_meta_learner(meta_X, meta_y):
    """
    Train a Ridge meta-learner on stacked base-model predictions.

    Ridge regression is chosen for its stability and resistance to
    overfitting when the meta-feature space is very small (3 columns).

    Parameters
    ----------
    meta_X : np.ndarray
        (n_oof, 3) stacked OOF predictions.
    meta_y : np.ndarray
        (n_oof,) true target values.

    Returns
    -------
    Ridge
        Fitted meta-learner.
    """
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X, meta_y)
    meta_pred = meta.predict(meta_X)
    mae = mean_absolute_error(meta_y, meta_pred)
    print(f"  Meta-learner OOF MAE: {mae:.6f}")
    return meta


def combiner(meta, meta_X, meta_y):
    """
    A function mapping the members' predictions (rows x 3) to the stack's, for
    the combiner ``meta``: 'ridge' (the shipped Ridge with an intercept), 'nnls'
    (non-negative weights, no intercept) or 'mean' (the plain average).
    """
    if meta == 'ridge':
        return _train_meta_learner(meta_X, meta_y).predict
    if meta == 'nnls':
        w, _ = nnls(np.asarray(meta_X, dtype=float), np.asarray(meta_y, dtype=float))
        return lambda X: np.asarray(X, dtype=float) @ w
    if meta == 'mean':
        return lambda X: np.asarray(X, dtype=float).mean(axis=1)
    raise ValueError(f"meta must be one of {META_CHOICES}, not {meta!r}")


# ==========================================
# 4. Main Orchestrator (Direct Multi-Horizon)
# ==========================================
def stack_paths(df, interval, horizon, leaf_scale=False, purge=False,
                variants=(('ridge', False),)):
    """
    Forecast paths for several (meta, shrink) settings from one set of fits.

    The members do not depend on the combiner or on ``shrink``, so every
    setting in ``variants`` is read from the same fits; ``leaf_scale`` and
    ``purge`` change the fits and are fixed for the call. Returns
    {(meta, shrink): DataFrame}. With ``variants=(('ridge', False),)`` and both
    other switches off this is the shipped model.
    """
    interval = normalize_interval(interval)
    variants = tuple((str(m), bool(s)) for m, s in variants)
    metas = tuple(dict.fromkeys(m for m, _ in variants))
    for m in metas:
        if m not in META_CHOICES:
            raise ValueError(f"meta must be one of {META_CHOICES}, not {m!r}")

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
    origin_vals = origin_X.values   # base models are trained on numpy arrays
    future_dates = generate_future_dates(last_date, horizon, interval)

    print(f"Training {horizon} direct stacking-ensemble horizon models...")

    rows = {v: [] for v in variants}
    for i, h in enumerate(range(1, horizon + 1)):
        target_name = f'target_h{h}'
        train_df = training_frame(
            features_df, feature_cols, targets[h], target_name,
        )
        leaf = min_leaf(h, len(train_df)) if leaf_scale else None
        gap = h if purge else 0

        meta_X, meta_y, trained_base, _, _ = _train_and_predict_oof(
            train_df, feature_cols, target_name, leaf, gap,
        )
        base_preds = np.array([[model.predict(origin_vals)[0] for _, model, _ in trained_base]])
        cum = {m: float(combiner(m, meta_X, meta_y)(base_preds)[0]) for m in metas}

        def _stack_fit_predict(proper_df, X_cal, _tn=target_name, _leaf=leaf, _gap=gap):
            """Train the full stack on proper-train rows; predict the
            calibration rows with every combiner (one column each)."""
            mX, mY, base, _, _ = _train_and_predict_oof(
                proper_df, feature_cols, _tn, _leaf, _gap,
            )
            cal_vals = X_cal.values
            base_stack = np.column_stack([m.predict(cal_vals) for _, m, _ in base])
            return np.column_stack([combiner(m, mX, mY)(base_stack) for m in metas])

        parts = conformal_calibration(
            train_df, feature_cols, target_name, _stack_fit_predict, embargo=h,
        )
        fallback = 1.96 * float(np.std(train_df[target_name].values))

        for meta, shrink in variants:
            ret = cum[meta]
            half = None
            if parts is not None:
                y_cal, p_all = parts
                p_cal = p_all[:, metas.index(meta)]
                if shrink:
                    den = float(np.sum(p_cal * p_cal))
                    beta = float(np.clip(np.sum(p_cal * y_cal) / den, 0.0, 1.0)) if den > 0 else 0.0
                    ret *= beta
                    half = conformal_quantile(y_cal - beta * p_cal)
                else:
                    half = conformal_quantile(y_cal - p_cal)
            if half is None:
                half = fallback

            median = last_price * np.exp(ret)
            lower = last_price * np.exp(ret - half)
            upper = last_price * np.exp(ret + half)
            rows[(meta, shrink)].append({
                'datetime': future_dates[i],
                'predicted_price': float(median),
                'lower_bound': float(lower),
                'upper_bound': float(upper),
                'volatility': float((upper - lower) / (2 * 1.96)),
            })

    return {v: pd.DataFrame(r) for v, r in rows.items()}


def run_ensemble_prediction(df, interval, horizon, meta=None, shrink=None,
                            leaf_scale=None, purge=None):
    """
    Main function to run the stacking ensemble prediction.

    Uses **direct multi-horizon** forecasting: for each step ``h`` a full
    stacking ensemble (XGBoost + LightGBM + Ridge base learners with a Ridge
    meta-learner, trained on temporally-honest out-of-fold predictions) targets
    the *cumulative* log return ``log(P[t+h]/P[t])`` and forecasts directly from
    the observed features at the last bar. This removes the recursive error
    accumulation (and one-bar origin offset) that made the recursive ensemble a
    bottom performer in the walk-forward backtest, while preserving the
    leakage-aware OOF stacking design. Distribution-free **conformal**
    prediction intervals are added per horizon via a temporal split-conformal
    calibration of the full stack.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with datetime, OHLCV columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps.
    meta, shrink, leaf_scale, purge : optional
        None (default) takes :func:`resolve_switches`'s default (README
        section 113): STACK_LONG (shrink, leaf_scale and purge on) on daily
        bars beyond 13 sessions, otherwise STACK_LEGACY (the Ridge combiner,
        no shrink, the members' own leaf sizes, no purge).
        ``meta`` is 'ridge', 'nnls' or 'mean' (see :func:`combiner`);
        ``shrink`` shrinks each step's return toward no change by the
        calibration block's least-squares slope through the origin, clipped to
        [0, 1]; ``leaf_scale`` sets the members' rows per leaf to
        :func:`min_leaf`; ``purge`` drops h rows at the end of each training
        chunk inside the stack.

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'predicted_price', 'lower_bound',
        'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    sw = resolve_switches(interval, horizon, meta, shrink, leaf_scale, purge)
    print(f"--- Starting Stacking Ensemble Prediction ({interval}) ---")
    v = (sw['meta'], sw['shrink'])
    out = stack_paths(df, interval, horizon, sw['leaf_scale'], sw['purge'], variants=(v,))[v]
    print("Stacking Ensemble Prediction Complete.")
    return out
```
