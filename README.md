- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
The app's serving pipeline without Streamlit: fetch, run each model, assess
it, gate it.

``UI.py`` wraps these in ``st.cache_data`` and renders what they return;
``run_watchlist.py`` calls them on a schedule (README section 78). Both
paths therefore serve, gate and log a forecast the same way, which is what
lets the scheduled forecasts stand in for served ones in the forward
prediction log.
"""

import numpy as np
import pandas as pd

import config as _cfg
from ARIMA_GARCH import run_statistical_prediction
from ARIMAX_model import run_arimax_prediction
from CNN_LSTM_model import run_cnn_lstm_prediction
from chronos_model import run_chronos_prediction
from config import USE_ASYMMETRIC_CONE, USE_IV_ADJUST
from DL_model import run_DL_prediction
from Ensemble_model import run_ensemble_prediction
from fb_prophet_model import run_prophet_prediction
from HAR_RV_model import run_har_rv_prediction
from Hybrid_model import run_hybrid_prediction
from kalman_model import run_kalman_prediction
from meta_ensemble import run_meta_ensemble_prediction
from ML_model import run_ML_prediction
from NBeats_model import run_nbeats_prediction
from Quantile_model import run_quantile_prediction
from RegimeSwitching_model import run_regime_prediction
from TCN_model import run_tcn_prediction
from Transformer_model import run_transformer_prediction
from vol_lstm_model import run_vol_lstm_prediction

# Single source of truth for the model suite: (display name, function, kwargs),
# in execution order. The sidebar selector, the runner, and the cache key all
# derive from this list. NOTE: Meta-Ensemble internally re-runs Quantile-LGBM +
# XGBoost + ARIMA (ARIMAX's pipeline without regressors), so selecting it alongside those duplicates that compute;
# per-model caching (in UI.py) makes the duplication a cache hit rather than a
# recomputation.
MODEL_REGISTRY = [
    ("ARIMA-GARCH", run_statistical_prediction, {}),
    ("ARIMAX-GARCH", run_arimax_prediction, {}),
    ("Prophet", run_prophet_prediction, {}),
    ("LSTM (DL)", run_DL_prediction, {'model_type': 'LSTM'}),
    ("GRU (DL)", run_DL_prediction, {'model_type': 'GRU'}),
    ("XGBoost (ML)", run_ML_prediction, {'model_type': 'xgboost'}),
    ("LightGBM (ML)", run_ML_prediction, {'model_type': 'lightgbm'}),
    ("Transformer", run_transformer_prediction, {}),
    ("Chronos-2", run_chronos_prediction, {}),  # company_name injected at call
    ("N-BEATS", run_nbeats_prediction, {}),
    ("TCN", run_tcn_prediction, {}),
    ("CNN-LSTM", run_cnn_lstm_prediction, {}),
    ("Stacking Ensemble", run_ensemble_prediction, {}),
    ("Quantile (LGB)", run_quantile_prediction, {}),
    ("Regime Switching", run_regime_prediction, {}),
    ("Hybrid ARIMA+LSTM", run_hybrid_prediction, {}),
    # The only state-space model in the suite: parameters are re-estimated every
    # bar rather than frozen after a fit, and the interval falls out of the state
    # covariance instead of a conformal calibration (Klaas Ch. 4).
    ("Kalman (local trend)", run_kalman_prediction, {}),
    ("HAR-RV (volatility)", run_har_rv_prediction,
     {'iv_adjust': USE_IV_ADJUST, 'asymmetric': USE_ASYMMETRIC_CONE}),
    ("Vol-LSTM (volatility)", run_vol_lstm_prediction,
     {'asymmetric': USE_ASYMMETRIC_CONE}),
    ("Meta-Ensemble", run_meta_ensemble_prediction, {}),
]
MODEL_NAMES = [name for name, _, _ in MODEL_REGISTRY]
#: A note shown on a model's card whenever it runs (README sections 89 and 92).
MODEL_NOTES = {
    "Prophet": ("On daily NSE data before 2025, Prophet's own trend and seasonality forecast "
                "worse than no change (README section 89). Its predicted change is now shrunk "
                "toward no change by how well it predicted at 20 recent dates, often to none, "
                "and its error is then about that of a no-change forecast (README section 116 "
                "for daily bars, README section 118 for hourly bars). Weigh its range, not "
                "its line."),
}
#: Listed in the app but not selected by default. Empty since section 92: every
#: model starts selected and the user decides, with MODEL_NOTES on the card.
OFF_BY_DEFAULT = frozenset()
DEFAULT_MODELS = [m for m in MODEL_NAMES if m not in OFF_BY_DEFAULT]
_MODEL_LOOKUP = {name: (func, kwargs) for name, func, kwargs in MODEL_REGISTRY}

# Walk-forward confidence-gate settings (kept light so it stays interactive).
GATE_SPLITS = _cfg.GATE_SPLITS
GATE_MIN_TRAIN = _cfg.GATE_MIN_TRAIN
GATE_MAX_ROWS = _cfg.GATE_MAX_ROWS


def _call_args(company_name, model_name):
    func, kwargs = _MODEL_LOOKUP[model_name]
    kwargs = dict(kwargs)
    if model_name == "Chronos-2":
        kwargs["company_name"] = company_name
    return func, kwargs


def fetch_history(company_name, backend_interval, use_context, note=None):
    """
    Historical OHLCV for a company, optionally with market-context features.

    Resilient fetch: retries + last-known-good cache fallback, tagging
    ``attrs['source']`` (live / partial / cached_fallback), which the
    data-quality gate reads. Returns None when the backend returned nothing.
    ``note(msg)`` receives a skipped-enrichment message.
    """
    from get_data import get_whole_stock_data
    from safe_fetch import fetch_with_fallback
    hist_df = fetch_with_fallback(
        company_name, backend_interval, fetch_fn=get_whole_stock_data)
    if not isinstance(hist_df, pd.DataFrame) or hist_df.empty:
        return None
    if use_context:
        try:
            from market_context import add_market_context
            hist_df = add_market_context(hist_df, backend_interval)
        except Exception as ctx_err:
            if note:
                note(f"market-context enrichment skipped: {ctx_err}")
    return hist_df


def run_model(hist_df, company_name, backend_interval, horizon, model_name):
    """One model's prediction, its 95% band widened to be monotone (README section 70)."""
    func, kwargs = _call_args(company_name, model_name)
    out = func(hist_df, backend_interval, horizon, **kwargs)
    from model_utils import widen_to_monotone
    return widen_to_monotone(out)


def get_col_by_candidates(df, candidates):
    """
    Find the first existing column from a list of candidates (case-insensitive).

    Returns the actual column name from the DataFrame if found, else None.
    """
    df_cols = [c.lower() if isinstance(c, str) else str(c).lower()
               for c in df.columns]
    for cand in candidates:
        if cand.lower() in df_cols:
            return df.columns[df_cols.index(cand.lower())]
    return None


PRICE_COLS = ['predicted_price', 'prediction', 'forecast', 'yhat', 'pred']
HIST_PRICE_COLS = ['adj_close', 'adj close', 'close', 'price']


def final_price(pred_df):
    """The last predicted price, or NaN when the frame has none."""
    col = get_col_by_candidates(pred_df, PRICE_COLS)
    if col is None or not len(pred_df):
        return float('nan')
    return float(pd.to_numeric(pred_df[col], errors='coerce').iloc[-1])


def is_usable(pred_df):
    """
    A forecast the app may serve. A NaN/inf final price is a failed fit, not a
    forecast: kept, it read as "Bearish" (NaN >= 0 is False) and turned the risk
    panel's median NaN.
    """
    return (isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0
            and bool(np.isfinite(final_price(pred_df))))


def model_confidence(hist_df, company_name, backend_interval, horizon, model_name):
    """
    A model's recent skill from a light walk-forward backtest, and its bucket.

    The serve-time "predictability gate": per request, whether the model
    actually beats a random walk here.

    Returns
    -------
    dict
        {'bucket', 'improvement', 'dm_p', 'coverage', 'model_rmse',
         'baseline_rmse', 'n_eval', 'note'}.
    """
    import backtest as bt
    from backtest import confidence_bucket

    base = {'bucket': 'N/A', 'improvement': float('nan'), 'dm_p': float('nan'),
            'coverage': float('nan'), 'model_rmse': float('nan'),
            'baseline_rmse': float('nan'), 'n_eval': 0, 'note': ''}
    if hist_df is None or len(hist_df) < GATE_MIN_TRAIN + horizon + 10:
        base['bucket'] = 'No data'
        base['note'] = 'insufficient history for a walk-forward check'
        return base

    gate_df = hist_df.tail(GATE_MAX_ROWS).reset_index(drop=True)
    func, kwargs = _call_args(company_name, model_name)

    try:
        res = bt.walk_forward_backtest(
            gate_df, func, backend_interval, horizon,
            n_splits=GATE_SPLITS, min_train=GATE_MIN_TRAIN,
            predict_kwargs=kwargs, model_name=model_name, verbose=False,
            # The serve-time gate needs the RMSE/DM summary only. The economic
            # block is deliberately NOT surfaced here: with 3 origins its Sharpe
            # would be pure noise, and showing a noisy Sharpe next to a
            # calibrated confidence bucket would undercut the honesty the gate
            # exists to provide. Use tests/validation/run_strategy_eval.py, which
            # runs enough non-overlapping origins for the number to mean something.
            strategy=False,
        )
    except Exception as exc:
        base['note'] = f'assessment failed: {str(exc)[:80]}'
        return base

    s = res['summary']
    dm = res.get('dm', {})
    model_rmse = float(s.loc['rmse', 'model'])
    baseline_rmse = min(
        float(s.loc['rmse', c]) for c in ('naive', 'drift', 'ar1')
        if c in s.columns
    )
    improvement = ((baseline_rmse - model_rmse) / baseline_rmse
                   if baseline_rmse > 0 else 0.0)
    dm_p = float(dm.get('p_value', float('nan')))
    coverage = (float(s.loc['coverage', 'model'])
                if 'coverage' in s.index else float('nan'))

    return {
        'bucket': confidence_bucket(improvement, dm_p, coverage),
        'improvement': improvement,
        'dm_p': dm_p,
        'coverage': coverage,
        'model_rmse': model_rmse,
        'baseline_rmse': baseline_rmse,
        'n_eval': len(res.get('origins', [])),
        'note': '',
    }


def regime_novelty(hist_df, backend_interval):
    """
    How unlike its fitted history the latest market window is.

    Wraps ``autoencoder_regime.regime_novelty``. This is the signal the other two
    gates cannot see: ``data_quality`` catches a broken *input*, and the
    confidence bucket catches a model with no measured edge *in the past* — but
    neither knows that the present regime is off-manifold, where a
    backtest-derived bucket is an extrapolation rather than evidence. Every
    failure path returns ``available=False``, which is a no-op for the gate,
    so enabling this can only ever add information.
    """
    if hist_df is None:
        return {'available': False, 'verdict': 'unknown', 'note': 'no history'}
    try:
        import autoencoder_regime as ar
        return ar.regime_novelty(hist_df, backend_interval)
    except Exception as exc:
        return {'available': False, 'verdict': 'unknown',
                'note': f'novelty check unavailable: {str(exc)[:100]}'}


def last_price(hist_df):
    """The last observed price, the forecast's anchor, or None."""
    col = get_col_by_candidates(hist_df, HIST_PRICE_COLS)
    return float(hist_df[col].iloc[-1]) if col else None


#: A served band more than BAND_SANITY times wider, or narrower, than the
#: random-walk band the stock's own recent volatility implies is flagged
#: (README section 85). Vol-LSTM's exploding bands (section 83) went unseen
#: because every check looked at coverage, and a band wide enough covers all.
BAND_SANITY = 4.0
BAND_SANITY_WINDOW = 60


def band_sanity(pred_df, hist_df, ratio=BAND_SANITY, window=BAND_SANITY_WINDOW):
    """
    ``(q, flagged)``: the last step's band width over the random-walk band.

    The reference is the 95% band of a random walk with the volatility of the
    last ``window`` returns, ``exp(+-1.96 * sd * sqrt(h))`` around the last price,
    in the history's own bar units (so hourly is compared with hourly). ``q`` is
    NaN, and nothing is flagged, when the band or the history is missing.
    """
    nan = (float("nan"), False)
    if not isinstance(pred_df, pd.DataFrame) or not {"lower_bound", "upper_bound"} <= set(pred_df.columns):
        return nan
    col = get_col_by_candidates(hist_df, HIST_PRICE_COLS) if isinstance(hist_df, pd.DataFrame) else None
    if not col or len(pred_df) == 0:
        return nan
    p = pd.to_numeric(hist_df[col], errors="coerce").to_numpy(dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < 3:
        return nan
    sd = float(np.std(np.diff(np.log(p))[-window:], ddof=1))
    lo, hi = float(pred_df["lower_bound"].iloc[-1]), float(pred_df["upper_bound"].iloc[-1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or sd <= 0:
        return nan
    half = 1.959963984540054 * sd * np.sqrt(len(pred_df))
    honest = np.exp(half) - np.exp(-half)
    q = ((hi - lo) / p[-1]) / honest
    return float(q), bool(q > ratio or q < 1.0 / ratio)
```
