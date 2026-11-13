- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
# -----------------------------------------------------------------------------
# 1. IMPORTS
# -----------------------------------------------------------------------------

import streamlit as st

# Before any other import: if a required package is missing, show an install
# page (with a button that installs it) instead of a ModuleNotFoundError from
# deep inside a model module. dependencies.py is standard-library only.
import dependencies as _deps
_MISSING_PACKAGES = _deps.missing()
if _MISSING_PACKAGES:
    _deps.render_install_page(st, _MISSING_PACKAGES)      # ends with st.stop()

import pandas as pd
import datetime as _dt
import time as _time
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from get_data import company_list
from model_utils import compute_device_label, compute_status, NSE_BARS_PER_SESSION
# The serving pipeline (model registry, fetch, run, assess) lives in serving.py,
# without Streamlit, so the scheduled watchlist run (run_watchlist.py, README
# section 78) serves exactly what this page serves. This page caches and renders.
import serving as _serving
from serving import MODEL_NAMES, DEFAULT_MODELS, get_col_by_candidates

# Feature flags live in config.py — one registry, with the evidence for each
# default recorded next to it, and a STOCKAPP_* environment override per flag so
# an A/B run does not require editing (and accidentally committing) a constant.
# They were previously defined here and in ML_model.py, which forced the console
# banner to import a flag out of a model module purely to display it.
import config as _cfg
from config import (
    USE_MARKET_CONTEXT,
    USE_REGIME_NOVELTY,
    USE_PREDICTION_LOG,
)

# -----------------------------------------------------------------------------
# 2. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Forecasting App", layout="wide", page_icon="📈")

# The model suite (MODEL_REGISTRY, MODEL_NAMES) is defined in serving.py.

# -----------------------------------------------------------------------------
# 3. CUSTOM CSS
# -----------------------------------------------------------------------------
def local_css():
    """
    Inject custom CSS to style the sidebar buttons and page padding.

    Applies a green theme to the Submit button and a red one to Reset, with
    hover states, reduces the top padding, and scales the page down on narrow
    screens.

    The buttons are matched by their widget keys (Streamlit puts a stable
    ``st-key-<key>`` class on each keyed element). The previous selectors
    targeted ``[data-testid="column"]``, which Streamlit renamed to
    ``stColumn``, so the colours had silently stopped applying: Submit
    rendered in the theme's default red, the colour of a destructive action.
    """
    st.markdown("""
        <style>
        .st-key-btn_submit button {
            background-color: #28a745; color: white; border-color: #28a745;
        }
        .st-key-btn_submit button:hover {
            background-color: #218838; border-color: #1e7e34; color: white;
        }
        .st-key-btn_reset button {
            background-color: #dc3545; color: white; border-color: #dc3545;
        }
        .st-key-btn_reset button:hover {
            background-color: #c82333; border-color: #bd2130; color: white;
        }
        .block-container {
            padding-top: 2rem;
        }
        /* Phones and narrow windows: a smaller title and tighter gutters, so
           the headline is not three lines tall and tables keep their width. */
        @media (max-width: 640px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1rem; }
            h1 { font-size: 1.75rem !important; }
            h3 { font-size: 1.2rem !important; }
            [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        }
        </style>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. GRAPHING FUNCTION (Smart Column Detection)
# -----------------------------------------------------------------------------
def plot_predictions_with_history(original_df, prediction_df, company_name, interval,
                                  lookback=50, style="line"):
    """
    One model's chart: history (a line, or adjusted candles), the median
    forecast, its 95% range as a cone from the last close, and the random-walk
    reference, on a trading-session axis. Built by ``charts.forecast_chart``
    (README section 70); a missing date/price column shows an error and an
    empty figure.
    """
    import charts
    try:
        return charts.forecast_chart(original_df, prediction_df, company_name, interval,
                                     lookback=lookback, style=style)
    except ValueError as e:
        st.error(f"Could not identify Date/Price columns: {e}")
        return go.Figure()


# -----------------------------------------------------------------------------
# 5. BACKEND ORCHESTRATION
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_company_list():
    """
    Fetch and cache the list of available company names.

    Cached for 1 hour to avoid re-fetching on every Streamlit render
    cycle. The caching wrapper is separated from error handling so
    that ``st.error`` is not called inside a cached function.

    Returns
    -------
    list of str
        Sorted company names from BSE/NSE.
    """
    return company_list()


def get_available_companies():
    """
    Fetch the list of available company names for the sidebar dropdown.

    Delegates to a cached inner function and catches any errors,
    returning an empty list as a fallback.

    Returns
    -------
    list of str
        Sorted company names, or an empty list on error.
    """
    try:
        return _fetch_company_list()
    except Exception as e:
        st.error(f"Error fetching company list: {e}")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_history(company_name, backend_interval, use_context):
    """
    Fetch (and optionally enrich) historical data, cached per parameter set.

    Caching means a Streamlit rerun triggered by any widget interaction does not
    re-hit the network, and every model for a given company/interval reuses the
    same fetched history.

    Parameters
    ----------
    company_name : str
        Selected company name.
    backend_interval : str
        '1d' or '1h'.
    use_context : bool
        Whether to attach market-context features (see USE_MARKET_CONTEXT).

    Returns
    -------
    pd.DataFrame or None
        Historical OHLCV data, or None if the backend returned nothing.
    """
    # Resilient fetch: retries + last-known-good cache fallback, tagging
    # attrs['source'] (live / partial / cached_fallback) which the data-quality
    # gate reads. A transient yfinance hiccup degrades gracefully instead of
    # silently producing a broken forecast.
    return _serving.fetch_history(company_name, backend_interval, use_context,
                                  note=lambda m: _console.note("data", m))


@st.cache_data(ttl=3600, show_spinner=False)
def _run_one_model(company_name, backend_interval, horizon, model_name,
                   use_context):
    """
    Run a single model and return its prediction, cached per parameter set.

    Keyed on (company, interval, horizon, model, use_context) so reruns and
    re-selections are cache hits rather than recomputation — the fix for the
    UI re-running the whole suite on every interaction. Exceptions are not
    cached, so a transient failure is retried on the next run.

    Returns
    -------
    pd.DataFrame or None
        The model's prediction DataFrame, or None if history is unavailable.
    """
    hist_df = _fetch_history(company_name, backend_interval, use_context)
    if hist_df is None:
        return None
    # No model's 95% band may narrow as the horizon grows (README section 70).
    return _serving.run_model(hist_df, company_name, backend_interval, horizon, model_name)


# Data-quality gate: score the input before trusting any forecast (reviewer
# request). Bad data abstains or drops to the risk-cone-only mode rather than
# producing a confident-looking line off broken inputs.
from data_quality import assess_data_quality, gate_action

# Reproducibility: pin every RNG per run and record an auditable manifest.
from reproducibility import set_all_seeds, run_manifest, GLOBAL_SEED

# Forward prediction log: record every served forecast for later calibration-drift
# monitoring, live gate validation, and the paper portfolio (book Ch. 3).
import prediction_log as _plog
import forward_record as _fr

# Console output (console.py). The terminal trace used to show per-model chatter
# and nothing about what the app concluded; these calls mirror the UI's decision
# layer to stdout. Rendered from the same verdicts the browser shows, never
# recomputed, so the two channels cannot drift apart. STOCKAPP_QUIET=1 silences.
import console as _console
import architectures as _arch

_console.startup_banner(
    seed=GLOBAL_SEED,
    device=compute_device_label(),
    n_models=len(MODEL_NAMES),
    flags=_cfg.active_flags(),
    overrides=(_cfg.overridden()
               + [f"architecture:{n}" for n in _arch.OVERRIDDEN]),
    log_path=(_plog.LOG_FILE if USE_PREDICTION_LOG else None),
)
for _h in compute_status()['hints']:
    _console.note("compute", _h)


@st.cache_data(ttl=3600, show_spinner=False)
def _model_confidence(company_name, backend_interval, horizon, model_name,
                      use_context):
    """
    Assess a model's recent skill with a light walk-forward backtest (cached).

    Runs the existing ``backtest.walk_forward_backtest`` for one model on recent
    history and returns the metrics plus a confidence bucket. Cached per
    (company, interval, horizon, model) so it is paid once. This is the
    serve-time realisation of the reviewers' "predictability gate": expose, per
    request, whether the model actually beats a random walk here.

    Returns
    -------
    dict
        {'bucket', 'improvement', 'dm_p', 'coverage', 'model_rmse',
         'baseline_rmse', 'n_eval', 'note'}.
    """
    hist_df = _fetch_history(company_name, backend_interval, use_context)
    return _serving.model_confidence(hist_df, company_name, backend_interval,
                                     horizon, model_name)


def run_selected_models(company_name, ui_interval, horizon, selected_models):
    """
    Run the user-selected models, collecting predictions with a progress bar.

    The heavy work (fetch + each model) is delegated to cached functions, so
    this wrapper only orchestrates progress display and per-model error
    handling. Failed models surface as UI warnings without aborting the rest.

    Parameters
    ----------
    company_name : str
        Selected company name.
    ui_interval : str
        UI frequency label ('1 Day' or '1 Hour').
    horizon : int
        Number of forecast steps.
    selected_models : list of str
        Display names (subset of MODEL_NAMES) to run.

    Returns
    -------
    dict
        Mapping model name -> {'history': DataFrame, 'prediction': DataFrame}.
    """
    backend_interval = '1d' if ui_interval == '1 Day' else '1h'

    # Pin every RNG before any model trains, so a run is reproducible given the
    # seed (auditable, not predictive — see reproducibility.py).
    set_all_seeds(GLOBAL_SEED)

    with st.spinner(f"Fetching historical data ({backend_interval})..."):
        hist_df = _fetch_history(company_name, backend_interval, USE_MARKET_CONTEXT)
    if hist_df is None:
        st.error("Received empty data from backend.")
        return {}

    results = {}
    total = len(selected_models)
    progress_bar = st.progress(0.0)

    for i, name in enumerate(selected_models):
        progress_bar.progress(i / total, text=f"Running {name}... ({i + 1}/{total})")
        try:
            pred_df = _run_one_model(
                company_name, backend_interval, horizon, name, USE_MARKET_CONTEXT,
            )
            if pred_df is not None and len(pred_df) > 0:
                # A NaN/inf final price is a failed fit, not a forecast
                # (serving.is_usable).
                if _serving.is_usable(pred_df):
                    results[name] = {'history': hist_df, 'prediction': pred_df}
                else:
                    st.warning(f"Model {name} produced no usable forecast "
                               f"(non-finite price); left out of this run.")
        except Exception as e:
            st.warning(f"Model {name} failed: {e}")
        progress_bar.progress((i + 1) / total,
                              text=f"Completed {name} ({i + 1}/{total})")

    progress_bar.empty()
    return results


def _build_summary_table(model_results, last_actual_price, conf_map=None):
    """
    Build a comparison table summarising every model's final forecast.

    Parameters
    ----------
    model_results : dict
        Mapping of model name -> {'history': DataFrame,
        'prediction': DataFrame}.
    last_actual_price : float
        Last known actual price for computing change and direction.
    conf_map : dict, optional
        Mapping model -> confidence dict (from :func:`_model_confidence`). When
        a model's bucket is **'No edge'**, its directional call is **suppressed**
        (Sentiment → "No edge", Direction → "–"): asserting Bullish/Bearish off a
        point forecast that the walk-forward test says is no better than a random
        walk would contradict the app's own abstention logic. Pass None / empty
        to leave sentiment ungated (e.g. when the user disabled assessment).

    Returns
    -------
    pd.DataFrame or None
        Summary DataFrame with columns [Model, Predicted Price, Change (₹),
        Signal Score (%), Conviction, Sentiment, Direction], or None if no valid
        predictions exist.

        - **Signal Score (%)** — forecasted percentage change from the last
          actual price to the horizon; the quantity the bull/bear call is made on.
        - **Conviction** — the forecast move divided by half the model's own 95%
          interval width, i.e. how large the move is relative to the model's own
          uncertainty. Same sign as the move; ``|Conviction| >= 1`` means the
          current price falls outside the 95% interval (a high-conviction call).
          NaN for models that emit no interval.
        - **Sentiment** — "Bullish" / "Bearish" by the sign of the move, or
          "No edge" when the confidence gate found no edge over a random walk.
        - **Direction** — the matching ▲ / ▼ / – marker (kept as the last column).
    """
    conf_map = conf_map or {}
    rows = []
    for name, data in model_results.items():
        pred_df = data['prediction']
        price_col = get_col_by_candidates(
            pred_df,
            ['predicted_price', 'prediction', 'forecast', 'yhat', 'pred'],
        )
        if price_col and len(pred_df) > 0:
            final_price = float(pred_df[price_col].iloc[-1])
            change = final_price - last_actual_price
            pct = (change / last_actual_price) * 100

            # Conviction: forecast move relative to the model's own 95% band
            # (pure rule in gating.py; wide band -> |conviction| < 1).
            lower_col = get_col_by_candidates(pred_df, ['lower_bound', 'lower'])
            upper_col = get_col_by_candidates(pred_df, ['upper_bound', 'upper'])
            low = float(pred_df[lower_col].iloc[-1]) if lower_col else None
            up = float(pred_df[upper_col].iloc[-1]) if upper_col else None
            conv = _conviction(change, low, up)
            conviction = round(conv, 2) if conv == conv else np.nan   # NaN-safe

            # Gate the directional call by the shared abstention rule: "No edge"
            # means don't assert a direction.
            bucket = conf_map.get(name, {}).get('bucket')
            sentiment, direction = _gated_direction(change, bucket)

            rows.append({
                'Model': name,
                'Predicted Price': round(final_price, 2),
                'Change (₹)': round(change, 2),
                'Signal Score (%)': round(pct, 2),
                'Conviction': conviction,
                'Sentiment': sentiment,
                'Direction': direction,
            })
    return pd.DataFrame(rows) if rows else None


def _build_consensus(summary_df):
    """
    Summarise where the models collectively land.

    Given the project's own finding that no single model reliably wins, the
    collective view (how many models lean up vs. down, and the median forecast)
    is more informative than any one row.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of :func:`_build_summary_table`.

    Returns
    -------
    dict
        {'n_models', 'n_up', 'n_down', 'n_noedge', 'median_price', 'median_pct'}.
    """
    n = len(summary_df)
    n_up = int((summary_df['Sentiment'] == 'Bullish').sum())
    n_down = int((summary_df['Sentiment'] == 'Bearish').sum())
    return {
        'n_models': n,
        'n_up': n_up,
        'n_down': n_down,
        'n_noedge': n - n_up - n_down,
        'median_price': float(summary_df['Predicted Price'].median()),
        'median_pct': float(summary_df['Signal Score (%)'].median()),
    }


# Pure risk math lives in risk_metrics.py (no Streamlit dep) so it can be unit
# tested directly. The UI just renders what these return.
from risk_metrics import empirical_risk as _empirical_risk
from risk_metrics import risk_from_interval as _risk_from_interval
from risk_metrics import breakeven_move as _breakeven_move
from risk_metrics import widen_log_returns as _widen_log_returns
from risk_metrics import widen_interval as _widen_interval
from risk_metrics import results_sessions as _results_sessions
from risk_metrics import results_widening as _results_widening
from risk_metrics import RESULTS_DAY0_VAR, RESULTS_DAY1_VAR


def build_risk_panel(model_results, last_actual, threshold_pct=2.0, scale=1.0):
    """
    Aggregate forecast distributions into consensus risk metrics.

    **Empirical first.** Any model that exposes a fat-tailed return sample via
    ``prediction.attrs['risk_log_returns']`` (HAR-RV's and Vol-LSTM's conformal
    residuals) is used directly — samples are pooled and probabilities counted off the
    empirical distribution, preserving the tails. Only if *no* model exposes
    such a sample do we fall back to the log-normal interval approximation
    (:func:`_risk_from_interval`), and the returned ``method`` flag says which
    path was taken so the UI can caption it honestly.

    Parameters
    ----------
    model_results : dict
        Mapping model -> {'history', 'prediction'}.
    last_actual : float
        Last observed price.
    threshold_pct : float, optional
        Move size (in %) for the gain/loss probabilities. Default 2.0.
    scale : float, optional
        Widen every distribution about its median by this factor, for a
        results session inside the horizon (``risk_metrics.results_widening``).

    Returns
    -------
    dict or None
        Consensus metrics (incl. 'method' and 'n_used'), or None if nothing
        usable is available.
    """
    # --- Empirical path: pool every model's fat-tailed return sample. ---
    samples, src_models = [], []
    for name, data in model_results.items():
        pred = data['prediction']
        sample = pred.attrs.get('risk_log_returns') if hasattr(pred, 'attrs') else None
        if sample is not None and len(sample) >= 20:
            samples.append(np.asarray(sample, dtype=float))
            src_models.append(name)
    if samples:
        pooled = _widen_log_returns(np.concatenate(samples), scale)
        m = _empirical_risk(last_actual, pooled, threshold_pct)
        m.update({
            'method': 'empirical',
            'n_used': len(src_models),
            'source_models': src_models,
            'threshold_pct': threshold_pct,
            'var5_price': last_actual * (1.0 + m['var5_pct']),
            'cvar5_price': last_actual * (1.0 + m['cvar5_pct']),
        })
        return m

    # --- Fallback path: log-normal read of each model's interval. ---
    keys = ('p_up', 'p_gain', 'p_loss', 'var5_pct', 'p_stop_hit', 'stop_1in5_pct',
            'p_trail_hit', 'p_target_touch')
    acc = {k: [] for k in keys}
    lows, highs = [], []
    for data in model_results.values():
        pred = data['prediction']
        pcol = get_col_by_candidates(
            pred, ['predicted_price', 'prediction', 'forecast', 'yhat', 'pred'])
        lcol = get_col_by_candidates(pred, ['lower_bound', 'lower'])
        ucol = get_col_by_candidates(pred, ['upper_bound', 'upper'])
        if not (pcol and lcol and ucol) or len(pred) == 0:
            continue
        _lo, _hi = _widen_interval(float(pred[pcol].iloc[-1]), float(pred[lcol].iloc[-1]),
                                   float(pred[ucol].iloc[-1]), scale)
        r = _risk_from_interval(
            float(pred[pcol].iloc[-1]), _lo, _hi, last_actual, threshold_pct)
        if r is None:
            continue
        for k in keys:
            acc[k].append(r[k])
        lows.append(r['range_low'])
        highs.append(r['range_high'])

    if not acc['p_up']:
        return None
    med = lambda a: float(np.median(a))
    var5 = med(acc['var5_pct'])
    return {
        'method': 'gaussian',
        'n_used': len(acc['p_up']),
        'source_models': [],
        'p_up': med(acc['p_up']),
        'p_gain': med(acc['p_gain']),
        'p_loss': med(acc['p_loss']),
        'p_stop_hit': med(acc['p_stop_hit']),
        'stop_1in5_pct': med(acc['stop_1in5_pct']),
        'p_trail_hit': med(acc['p_trail_hit']),
        'p_target_touch': med(acc['p_target_touch']),
        'threshold_pct': threshold_pct,
        'var5_pct': var5,
        'var5_price': last_actual * (1.0 + var5),
        'range_low': med(lows),
        'range_high': med(highs),
    }


def _honesty_banner(conf_map):
    """
    Build an aggregate reliability banner from the per-model confidence buckets.

    Encodes the reviewers' (and the backtest's) core message: when models don't
    reliably beat a random walk, say so plainly and steer the user to the range
    rather than the point line.

    Parameters
    ----------
    conf_map : dict
        Mapping model name -> confidence dict (from :func:`_model_confidence`).

    Returns
    -------
    tuple(str, str) or None
        (streamlit level: 'error'|'warning'|'success', message), or None.
    """
    buckets = [c.get('bucket') for c in conf_map.values()]
    n = len(buckets)
    if n == 0:
        return None
    reliable = sum(b in ('High', 'Medium') for b in buckets)
    # The check behind the buckets is a few walk-forward dates on this one
    # stock (config.GATE_SPLITS, 3 by default): enough to catch a broken model,
    # not to prove a good one (README section 98).
    k = max((int(c.get('n_eval') or 0) for c in conf_map.values()), default=0)
    check = (f"a quick check of {k} recent forecast dates" if k
             else "a quick check of recent forecast dates")
    if reliable == 0:
        return ('error',
                "**No model showed a statistically reliable edge over a random "
                "walk** for this ticker & horizon in recent walk-forward tests. "
                "Treat the **range / interval** as the output — not the point "
                "forecast.")
    if reliable < n / 2:
        return ('warning',
                f"Only **{reliable} of {n}** models did better than a no-change "
                f"forecast in {check} here, which is not proof. Weight the "
                "**range** over the point line.")
    return ('success',
            f"**{reliable} of {n}** models did better than a no-change forecast "
            f"in {check} here: a small sample, not proof.")


#: Under every run that can show an up/down call (README section 98). The larger
#: tests (sections 87, 90 and 91) found no model whose direction held up across
#: separate weeks.
DIRECTION_CAVEAT = ("No model's up/down call has yet proven reliable across "
                    "separate weeks of testing. Read a Bullish or Bearish label "
                    "as a pointer, not a prediction.")


# Pure serving-gate logic lives in gating.py (no Streamlit) so the trust layer is
# unit-tested directly. The UI just renders what these return.
from gating import (
    gated_direction as _gated_direction,
    conviction as _conviction,
    apply_data_quality_gate as _apply_dq_gate,
    apply_novelty_gate as _apply_novelty_gate,
    confidence_reason as _confidence_reason,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _regime_novelty(company_name, backend_interval, use_context):
    """
    Score how unlike its fitted history the latest market window is (cached).

    Wraps ``autoencoder_regime.regime_novelty``. This is the signal the other two
    gates cannot see: ``data_quality`` catches a broken *input*, and the
    confidence bucket catches a model with no measured edge *in the past* — but
    neither knows that the present regime is off-manifold, where a
    backtest-derived bucket is an extrapolation rather than evidence.

    Cached per (company, interval) because fitting the autoencoder costs a few
    seconds and the answer is identical for every model in the run. Every failure
    path returns ``available=False``, which is a no-op for the gate, so enabling
    this can only ever add information.
    """
    return _serving.regime_novelty(
        _fetch_history(company_name, backend_interval, use_context), backend_interval)


def _render_fx_notice(hist_df):
    """
    Surface *how* prices were converted to INR, so the figure is auditable.

    Reads the ``fx_conversion`` audit tag attached by ``get_data`` and renders a
    caption (normal paths) or a prominent warning/error (degraded paths). The
    honest distinction the reviewer asked for: per-date historical conversion is
    period-accurate, whereas the live-uniform fallback rescales the *whole*
    history by today's rate (distorting historical levels) and the failed path
    leaves prices in the original currency.
    """
    info = getattr(hist_df, 'attrs', {}).get('fx_conversion') if hist_df is not None else None
    if not info:
        return
    method = info.get('method')
    cur = info.get('source_currency', '?')
    if method == 'none':
        return                                   # native INR — nothing to flag
    if method == 'historical':
        rate = info.get('latest_rate')
        msg = (f"💱 Prices converted **{cur} → INR at each date's own historical "
               f"rate** (period-accurate; latest ≈ ₹{rate}/{cur}).")
        if info.get('rows_filled_live'):
            msg += (f" {info['rows_filled_live']} early row(s) had no historical "
                    "rate and used today's rate as a fallback.")
        st.caption(msg)
    elif method == 'live_uniform':
        st.warning(
            f"💱 **Historical {cur}→INR rates were unavailable, so the *entire* "
            f"history was converted at today's rate (₹{info.get('rate')}/{cur}).** "
            "Day-to-day returns are preserved, but **historical price *levels* are "
            "rescaled and not period-accurate** — read absolute ₹ levels with care."
        )
    elif method == 'unconverted':
        st.error(
            f"💱 **Currency conversion failed — prices are shown in the original "
            f"{cur}, not INR.** The ₹ labels do not apply to this run."
        )


@st.cache_data(ttl=3600, show_spinner=False)
def _dividend_notice(ticker, last_price, dates_iso):
    """Announced ex-dividend dates inside the horizon (cached; advisory only)."""
    import corporate_events as ce
    return ce.dividend_notice(ticker, last_price, list(dates_iso))


def _render_dividend_notice(hist_df, model_results, last_actual):
    """
    Say so when an announced ex-dividend date falls inside the forecast window.

    Mladjenovic, Ch. 6: on the ex-date the price drops by about the dividend.
    The forecasts are total-return prices, so without this note a user would
    read the quoted price's drop as a forecast miss (corporate_events.py).
    """
    ticker = getattr(hist_df, 'attrs', {}).get('ticker') if hist_df is not None else None
    if not ticker or last_actual is None or not model_results:
        return
    try:
        pred = list(model_results.values())[0]['prediction']
        dates = tuple(pd.to_datetime(pred['datetime']).dt.strftime('%Y-%m-%d'))
    except Exception:
        return
    note = _dividend_notice(ticker, float(last_actual), dates)
    if note:
        st.info("💰 " + note['message'].replace("Rs ", "₹"))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _listing_info(symbol):
    """NSE listing date, face value and ISIN (cached a day; advisory)."""
    import corporate_events as ce
    return ce.listing_info(symbol)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _recent_actions(end_iso):
    """Five years of NSE corporate actions for every company, fetched once (cached)."""
    import bhavcopy as bc
    end = pd.Timestamp(end_iso)
    try:
        return bc.corporate_actions(end - pd.DateOffset(years=5), end + pd.Timedelta(days=120),
                                    snapshot=False)
    except Exception:
        return None


def _render_listing_dividends(hist_df, last_actual):
    """
    The company's NSE listing and its dividend record (README section 81).

    Listing date, face value and ISIN from NSE's listing file; every cash
    dividend of the last five years and any announced ahead, from NSE's
    corporate-action feed. The IPO issue price is not in either source.
    """
    import corporate_events as ce
    ticker = getattr(hist_df, 'attrs', {}).get('ticker') if hist_df is not None else None
    symbol = ce.nse_symbol(ticker)
    if not symbol:
        return
    with st.expander("🏷️ Listing and dividends (NSE)"):
        info = _listing_info(symbol)
        if info:
            listed = pd.Timestamp(info['listed']) if info['listed'] else None
            age = (f" ({(pd.Timestamp.today() - listed).days / 365.25:.0f} years ago)"
                   if listed is not None else "")
            st.markdown(f"**Listed on NSE:** {listed:%d %b %Y}{age} · **face value** ₹{info['face_value']:g}"
                        f" · **ISIN** {info['isin']} · series {info['series']}"
                        if listed is not None else f"**ISIN** {info['isin']}")
        else:
            st.caption("NSE's listing file did not answer; listing details unavailable.")
        acts = _recent_actions(pd.Timestamp.today().normalize().date().isoformat())
        if acts is None:
            st.caption("NSE's corporate-action feed did not answer; dividend history unavailable.")
            return
        divs = ce.dividend_history(symbol, acts)
        s = ce.dividend_summary(divs, last_actual)
        if s is None:
            st.markdown("**No cash dividend** with an ex-date in the last five years.")
            return
        line = (f"**Last 12 months:** ₹{s['ttm_dividend']:,.2f} per share, a **{s['ttm_yield']:.2%}** "
                f"yield at the last price ₹{float(last_actual):,.2f} · {s['n_5y']} dividends in five years")
        if s['upcoming']:
            nxt = s['upcoming'][0]
            line += f" · **next:** ₹{nxt['dividend']:,.2f} going ex on {nxt['ex_date']}"
        st.markdown(line)
        table = divs.head(12).assign(**{
            "Ex-date": lambda d: d["ex_date"].dt.strftime("%d %b %Y"),
            "₹ per share": lambda d: d["dividend"].map(lambda v: f"{v:,.2f}"),
            "As announced": lambda d: d["subject"]})[["Ex-date", "₹ per share", "As announced"]]
        st.dataframe(table, hide_index=True, width='stretch')
        st.caption("Source: NSE listing file and corporate-action feed. Amounts are per share as "
                   "declared; the yield divides them by any later split or bonus first. The IPO "
                   "issue price is not in either source, so it is not shown.")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _results_dates(ticker, start_iso, end_iso):
    """Results board meetings for one ticker from NSE's calendar (cached; advisory)."""
    import corporate_events as ce
    return ce.upcoming_results(ticker, start_iso, end_iso)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _option_view(ticker, horizon_days, threshold, model_sigma_h, event_iso):
    """The live option-chain view (cached 15 minutes; advisory)."""
    import option_market as om
    return om.option_view(ticker, horizon_days, threshold, model_sigma_h, event_iso)


def _render_option_view(ticker, horizon_days, thr, risk, meetings):
    """
    What the options market prices, next to the model (Benklifa, Think Like an
    Option Trader). F&O stocks only; risk-neutral, live, never a model input.
    """
    if not ticker or risk is None:
        return
    lo, hi = risk.get('range_low'), risk.get('range_high')
    model_sig = (np.log(hi / lo) / (2 * 1.959964)) if lo and hi and lo > 0 else None
    first = min(meetings).date().isoformat() if meetings else None
    v = _option_view(ticker, float(horizon_days), thr / 100.0,
                     float(model_sig) if model_sig else None, first)
    if not v:
        return
    u = v['used']
    with st.expander("🎯 What the options market prices (live NSE option chain)"):
        o1, o2, o3 = st.columns(3)
        o1.metric(f"Implied volatility ({u['expiry']} expiry)", f"{v['iv'] * 100:.1f}%",
                  help="From the at-the-money straddle, the most liquid options, with "
                       "the forward from put-call parity: the market's forecast of the "
                       "stock's annualised volatility to that expiry.")
        o2.metric("Expected move to expiry (straddle)", f"±{u['move_pct'] * 100:.1f}%",
                  help="The straddle's price as a share of the stock price: what the "
                       "market charges to own a move in either direction, about 0.8 "
                       "standard deviations.")
        o3.metric("1σ move over your horizon", f"±{v['sigma_h'] * 100:.1f}%",
                  (f"panel ±{model_sig * 100:.1f}%" if model_sig else None),
                  delta_color="off",
                  help="The implied volatility scaled to this forecast's horizon, "
                       "against the Risk & Probability panel's own 1σ.")
        lines = []
        ratio = v.get('ratio_to_model')
        if ratio and np.isfinite(ratio):
            lines.append(
                f"The options market's spread is **{ratio:.2f}×** this panel's. Implied "
                "volatility usually runs above what is later realised (India VIX ran a "
                "median 27% above NIFTY's realised volatility), so a ratio near 1.3 is "
                "ordinary. One well below 1 means the models expect more movement than "
                "the market is paying for, often because a recent spike, such as a "
                "results day, is still in their history.")
        lines.append(f"Risk-neutral P(gain > +{thr:.1f}%) from the options: "
                     f"**{v['p_gain'] * 100:.0f}%** (the panel says "
                     f"{risk['p_gain'] * 100:.0f}%).")
        em = v.get('event_move')
        if em:
            lines.append(
                f"📅 The market prices the results day on {first} as a "
                f"**±{em['move'] * 100:.1f}%** move (1σ), {em['ratio']:.1f}× an ordinary "
                f"session's variance. The panel's historical average is "
                f"{RESULTS_DAY0_VAR:.1f}× on the day and {RESULTS_DAY1_VAR:.1f}× the "
                "session after.")
        elif v.get('event'):
            lines.append("📅 The first expiry's volatility stands well above the second's: "
                         "the market is pricing an event before the first expiry.")
        put = v.get('put')
        if put:
            lines.append(
                f"**Defined risk instead of a stop:** a {put['strike']:,.0f} put "
                f"({put['expiry']}) costs **{put['cost_pct'] * 100:.2f}%** of the price and "
                f"caps the loss at **{put['max_loss_pct'] * 100:.1f}%** through expiry, gaps "
                f"included. A −{thr:.1f}% stop costs nothing up front, but this panel puts "
                f"the chance of it being hit at {risk.get('p_stop_hit', float('nan')) * 100:.0f}%, "
                "and a gap can sell below it.")
        st.markdown("\n\n".join(lines))
        st.caption(f"NSE option chain as of {v['asof']}. Option prices are risk-neutral: "
                   "they include what traders pay for protection, so they are not a "
                   "forecast (README §48). NSE keeps no option history, so none of this "
                   "can be backtested here; it is shown beside the forecast, never fed "
                   "into it.")


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def _market_breadth(day_iso):
    """10-session NSE advance/decline against frozen 2018-2024 quintiles (advisory)."""
    import breadth
    return breadth.live_breadth(today=day_iso)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _company_inputs(ticker):
    """Annual statements, info and dividends for one ticker (cached; shared by
    the company check and the stock screens, so they are fetched once)."""
    import stock_checklist as sc
    try:
        return sc.fetch_inputs(ticker)
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _company_checklist(ticker, last_price):
    """Mladjenovic's company check for one ticker (cached; advisory only)."""
    import stock_checklist as sc
    inputs = _company_inputs(ticker)
    if inputs is None:
        return None
    return sc.company_checklist(ticker, last_price=last_price, fetch_fn=lambda _t: inputs)


@st.cache_data(ttl=3 * 3600, show_spinner="Reading NSE's insider-trading disclosures…")
def _insider_view(symbol, day_iso):
    """The last 90 days of this stock's SEBI insider-trading disclosures (cached; advisory)."""
    import insiders
    return insiders.live_view(symbol)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def _nifty_close(day_iso):
    """About a year of NIFTY 50 closes to ``day_iso`` (cached; advisory)."""
    import oneil
    try:
        end = pd.Timestamp(day_iso)
        return oneil.load_nifty(end - pd.Timedelta(days=300), end)["close"]
    except Exception:
        return None


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def _distribution_count(day_iso):
    """NIFTY distribution days in the last 20 sessions (cached; advisory)."""
    import oneil
    return oneil.live_distribution_count()


def _render_oneil_check(hist_df, interval):
    """
    The O'Neil-disciple checklist (Morales & Kacher, Ch. 5 and 7) for this
    stock, each signal next to what it measured on NSE (README section 65).
    Daily bars only; never an input to a forecast.
    """
    import oneil
    if interval != '1d':
        return
    chk = oneil.checklist(hist_df)
    if not chk:
        return
    if chk['pp_proper']:
        pp = "Yes: a proper pocket pivot (volume signature, above the 50-day, from the 10-day)"
    elif chk['pp_core']:
        pp = (f"A pocket pivot by volume, but extended: the day's low is "
              f"{chk['extension'] * 100:.1f}% above the 10-day average, which the book calls "
              "a 'do not buy' pivot")
    else:
        pp = "No"
    gap = chk['gap']
    if gap:
        gap_txt = (f"Yes, on {gap['date']}" + (" in a stock near its highs" if gap['leader'] else "")
                   + (f"; the gap day's low (₹{gap['low']:,.2f}) has held" if gap['held']
                      else f"; the gap day's low (₹{gap['low']:,.2f}) has been broken, the book's sell signal"))
    else:
        gap_txt = "None in the last 21 sessions"
    ma_txt = (f"{chk['close_vs_ma50'] * 100:+.1f}% from the 50-day average, "
              f"{chk['close_vs_ma10'] * 100:+.1f}% from the 10-day")
    if chk.get('heavy_break'):
        ma_txt += f"; broke below the 50-day on heavy volume on {chk['heavy_break']}, the book's sell guide"
    rows = [
        {"Signal": "Pocket pivot (Ch. 5)", f"Last session ({chk['date']})": pp,
         "The book's rule": "Up day on volume at least the largest down-day volume of the prior "
                            "10 sessions, above the 50-day, starting from the 10-day average",
         "Measured on NSE, 2018-24": oneil.MEASURED["pocket_pivot"]},
        {"Signal": "Major gap-up (Ch. 5)", f"Last session ({chk['date']})": gap_txt,
         "The book's rule": "Buy a gap of 0.75x the 40-day ATR on 1.5x volume in a leading stock; "
                            "sell if the gap day's low breaks",
         "Measured on NSE, 2018-24": oneil.MEASURED["gap_up"]},
        {"Signal": "Moving-average sell guides (Ch. 5)", f"Last session ({chk['date']})": ma_txt,
         "The book's rule": "A close below the 50-day (or the 10-day, for a stock that has obeyed it) "
                            "on heavy volume is a sell guide",
         "Measured on NSE, 2018-24": oneil.MEASURED["ma_sell"]},
    ]
    mkt = _distribution_count(_dt.date.today().isoformat())
    if mkt:
        rows.append({
            "Signal": "Market direction: NIFTY distribution days (Ch. 7)",
            f"Last session ({chk['date']})": (f"{mkt['count']} in the last {mkt['sessions']} sessions to "
                                              f"{mkt['as_of']}" + (": the model's sell condition (5) is met"
                                                                   if mkt['sell'] else "")),
            "The book's rule": "Index down 0.1%+ on higher volume, or churning; five in 20 sessions "
                               "is a sell signal, a follow-through day the buy signal",
            "Measured on NSE, 2018-24": oneil.MEASURED["market"]})
    with st.expander("📈 The O'Neil-disciple checklist (Morales & Kacher), with what it measured on NSE"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption("The book's buy and sell signals, coded to its published rules and tested on "
                   "200 liquid NSE stocks and NIFTY, 2018 to mid-2024 (README §65). Only the "
                   "50-day sell guide worked in its own direction, and not decisively; the rest "
                   "did not beat their controls after costs. So they are shown as the book's "
                   "checklist, never as calls, and none feeds the forecast. Volume is rupee "
                   "turnover; the market's is NSE's total equity turnover.")


def _render_company_check(hist_df, last_actual, horizon, interval):
    """
    The fundamentals the models cannot see, as information beside the forecast.

    Mladjenovic, Ch. 21 and Appendix B: six measurable hallmarks of a great
    stock and the ratio rules of thumb, from the latest annual report; Ch. 3-4:
    what this horizon is called and which risks no price model covers. Nothing
    here feeds a forecast, and none of it is validated as a return predictor
    (README section 32: no point-in-time statement history to test it on).
    """
    import stock_checklist as sc
    ticker = getattr(hist_df, 'attrs', {}).get('ticker') if hist_df is not None else None

    with st.expander("📘 Company check: the book's hallmarks of a great stock "
                     "(information, not a forecast input)"):
        st.caption(sc.horizon_note(horizon, interval))
        chk = (_company_checklist(ticker, float(last_actual) if last_actual else None)
               if ticker else None)
        if not chk:
            st.caption("No annual statements were available for this stock, so "
                       "the company check is skipped.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Hallmarks met (Ch. 21)", f"{chk['met']} of {chk['measured']}",
                      help="Measured from the latest annual report, dividends and "
                           "NSE's F&O list. The book: a stock with half of the "
                           "ten hallmarks is a super-solid choice.")
            for col, key, name in ((m2, 'profitability', "Profitability"),
                                   (m3, 'solvency', "Solvency")):
                good, scored = chk['minimum'][key]
                col.metric(f"{name} tests passed (App. B)",
                           f"{good} of {scored}" if scored else "n/a",
                           help="Appendix B: a solid company should at minimum "
                                "comfortably pass the profitability and solvency "
                                "ratios.")
            _sz = chk.get('size') or {}
            if _sz.get('band'):
                _mc = _sz.get('market_cap')
                st.markdown(
                    f"**Size (Ch. 1, 14): {_sz['band']}**, {_sz['basis']}"
                    + (f" · market value ₹{_mc / 1e7:,.0f} cr"
                       if _mc and np.isfinite(_mc) else ""))
                if _sz.get('checks'):
                    st.dataframe(
                        pd.DataFrame([{"": c['icon'], "Small-cap rule (Ch. 14)": c['label'],
                                       "This company": c['value'], "The book's rule": c['rule']}
                                      for c in _sz['checks']]),
                        hide_index=True, width='stretch')
                    st.caption("The book on small caps: buying one before there is "
                               "evidence of growth is speculating, not investing, and "
                               "no investor should put a large part of their money in "
                               "them.")
            st.dataframe(
                pd.DataFrame([{"": h['icon'], "Hallmark": h['label'],
                               "This company": h['value'], "The book's rule": h['rule']}
                              for h in chk['hallmarks']]),
                hide_index=True, width='stretch')
            where = ", ".join(x for x in (chk.get('sector'), chk.get('industry')) if x)
            st.markdown(
                "**Your call: the four hallmarks no statement can score**"
                + (f" ({where})" if where else "") + "\n"
                + "\n".join(f"- **{name}.** {q}" for name, q in chk['judgement']))
            st.dataframe(
                pd.DataFrame([{"Group": g, "": c['icon'], "Ratio (App. B)": c['label'],
                               "This company": c['value'], "The book's rule": c['rule']}
                              for g, checks in chk['ratios'] for c in checks]),
                hide_index=True, width='stretch')
            st.caption(
                f"✅ meets the book's rule · ⚠️ falls short · ❌ red flag · ➖ not "
                f"applicable or not reported · ℹ️ context only. From yfinance's "
                f"latest annual statements (year ending {chk['fiscal_year_end']}); "
                "figures can differ from sites that use trailing-quarter or "
                "standalone numbers. Ratios vary by industry, and the book warns "
                "against deciding on any single one. This is the book's checklist, "
                "not a tested predictor of returns.")

    with st.expander("🧭 Risks this forecast cannot see (Ch. 4)"):
        st.dataframe(
            pd.DataFrame([{"Risk": r, "What it means": m, "In this app": a}
                          for r, m, a in sc.RISKS]),
            hide_index=True, width='stretch')
        st.caption("The book's advice for the rest: learn before you buy, keep "
                   "money you will need within a year out of stocks, and "
                   "diversify.")


def _render_hough_check(hist_df, interval):
    """
    Hough, *Your Next Great Stock*: which of the book's screens this stock
    passes (Part Three), the checks on a screen survivor (Ch. 24) and the
    reasons to sell (Ch. 25). From the latest annual statements and the daily
    price history; information beside the forecast, never an input to it
    (README section 67).
    """
    import hough
    ticker = getattr(hist_df, 'attrs', {}).get('ticker') if hist_df is not None else None
    inputs = _company_inputs(ticker) if ticker else None
    if not inputs:
        return
    daily = hist_df if interval == '1d' else None
    from corporate_events import nse_symbol
    _sym = nse_symbol(ticker)
    ins = _insider_view(_sym, _dt.date.today().isoformat()) if _sym else None
    try:
        res = hough.build_screens(inputs, hist=daily, insider=ins,
                                  market=hough.market_returns(_nifty_close(_dt.date.today().isoformat())))
    except Exception:
        res = None
    if not res:
        return

    def _crit_text(c):
        mark = "✅" if c['ok'] is True else ("❌" if c['ok'] is False else "➖")
        return f"{mark} {c['label']}: {c['value']}"

    summary, detail = [], []
    for s in res['screens']:
        failed = [c for c in s['criteria'] if c['ok'] is False]
        unknown = [c for c in s['criteria'] if c['ok'] is None]
        if s['status'] == hough.FAIL:
            why = "; ".join(f"{c['label']} ({c['value']})" for c in failed[:2])
        elif s['status'] == hough.CANT:
            why = unknown[0]['value'] if unknown else ""
        elif s['status'] == hough.PARTLY:
            why = "not measurable here: " + "; ".join(c['label'] for c in unknown)
        else:
            why = "every criterion met"
        summary.append({"": s['icon'], "Screen": f"{s['name']} ({s['chapter']})",
                        "Result": s['status'], "Why": why})
        for c in s['criteria'] + s['refinements']:
            detail.append({"Screen": s['name'], "Criterion": _crit_text(c),
                           "Kind": "recipe" if c in s['criteria'] else "optional refinement"})

    m = res['metrics']
    fs = res['fscore']
    with st.expander("🔎 Stock screens: which of Hough's recipes this stock passes "
                     "(information, not a forecast input)"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Screens passed", f"{sum(s['status'] == hough.PASS for s in res['screens'])} of "
                  f"{sum(s['status'] != hough.CANT for s in res['screens'])}",
                  help="Of the book's screens that can be run from yfinance data and NSE's insider "
                       "disclosures; the rest need analyst estimates, R&D or a whole-market ranking.")
        c2.metric("Piotroski F-score", f"{fs['score']} of {fs['scored']}" if fs['score'] is not None else "n/a",
                  help="Ch. 22: nine signals of improving profitability, strength and efficiency. "
                       "Piotroski's winners scored 8 or 9.")
        c3.metric("Return on invested capital", f"{m['roic'] * 100:.1f}%" if np.isfinite(m['roic']) else "n/a",
                  help="Ch. 24: after-tax operating profit over invested capital; 15% or more is a "
                       "good sign.")
        c4.metric("Net payout yield", f"{m['net_payout_yield'] * 100:.1f}%"
                  if np.isfinite(m['net_payout_yield']) else "n/a",
                  help="Ch. 16: dividends plus buybacks minus new shares, over market value.")
        t1, t2, t3, t4, t5 = st.tabs(["Screens (Part 3)", "Survivor checks (Ch. 24)",
                                      "When to sell (Ch. 25)", "F-score (Ch. 22)", "Insider trades (Ch. 19)"])
        with t1:
            st.dataframe(pd.DataFrame(summary), hide_index=True, width='stretch')
            st.dataframe(pd.DataFrame(detail), hide_index=True, width='stretch')
            st.caption("The book: a screen produces a short research list, not a buy list. Its "
                       "returns are from US studies of portfolios of screen survivors; none of "
                       "these screens can be back-tested on NSE here, because yfinance keeps "
                       "about a year of statement history (README §32).")
        with t2:
            # The check and its result first, so a phone shows them without scrolling.
            st.dataframe(pd.DataFrame([{"": c['icon'], "Check": c['label'], "This company": c['value'],
                                        "Step": g, "The book's rule": c['rule']}
                                       for g, c in res['survivor']]),
                         hide_index=True, width='stretch')
            st.caption("Ch. 24's three steps: cut the deal breakers, get to know the business "
                       "(the 10-K, here the annual report), then try to talk yourself out of it: "
                       "cheap and growing, strong, well managed.")
        with t3:
            if res['sell']:
                st.dataframe(pd.DataFrame([{"": c['icon'], "Sign": c['label'], "This company": c['value'],
                                            "The book's rule": c['rule']} for c in res['sell']]),
                             hide_index=True, width='stretch')
            st.markdown(
                "**The book's selling rules (Ch. 25).** Decide up front whether this is a growth "
                "(popular) or a value (unpopular) holding. Sell a **growth** stock when sales and "
                "earnings slow sharply or it misses estimates, not because its P/E is high. Sell "
                "a **value** stock when its P/E approaches the industry's, when it starts to make "
                "investors too happy. Either way, rethink it after a big acquisition, an earnings "
                "restatement without firings, recurring one-time charges, or falling margins and "
                "returns on capital. Above all, don't sell merely because it has gone up.")
        with t4:
            if fs['score'] is None:
                st.caption(f"The F-score is not computed: {fs['why']}.")
            else:
                st.dataframe(pd.DataFrame([{"": c['icon'], "Signal": c['label'], "This company": c['value']}
                                           for c in fs['checks']]),
                             hide_index=True, width='stretch')
                st.caption(f"FY{fs['year'][:4]} against the year before. {fs['rule']}")
        with t5:
            import insiders
            if not ins:
                st.caption("NSE's insider-trading disclosures could not be read just now.")
            else:
                b, sl = ins['buy'], ins['sell']
                st.markdown(
                    f"**Disclosed {ins['from']} to {ins['to']}:** {b['n']} open-market purchases by "
                    f"{b['buyers']} insiders (₹{b['value'] / 1e5:,.1f} lakh), {sl['n']} sales by "
                    f"{sl['buyers']} (₹{sl['value'] / 1e5:,.1f} lakh). The book's screen: at least "
                    f"two purchases over ₹{insiders.MIN_VALUE / 1e5:.0f} lakh in total, under 5% of the "
                    "shares; " + ("**met**." if b['meets'] else "not met."))
                tr = ins['trades']
                if len(tr):
                    st.dataframe(pd.DataFrame({
                        "Disclosed": tr['disclosed'].dt.strftime('%d-%b-%Y'),
                        "Who": tr['person'], "Category": tr['category'],
                        "Trade": tr['side'].str.capitalize(),
                        "Shares": tr['shares'].map(lambda v: f"{v:,.0f}" if v == v else "—"),
                        "Value": tr['value'].map(lambda v: f"₹{v / 1e5:,.1f} lakh" if v == v else "—"),
                    }), hide_index=True, width='stretch')
                other = ins['all_modes']
                other = other[~other['mode'].isin([insiders.OPEN_MARKET_BUY, insiders.OPEN_MARKET_SELL])]
                if len(other):
                    st.caption("Not counted (not a decision to buy or sell at the market price): "
                               + ", ".join(f"{k} {v}" for k, v in other['mode'].value_counts().items())
                               + ". Designated employees' and relatives' trades are left out too.")
                st.caption(insiders.MEASURED['leaders'] + " Source: SEBI insider-trading "
                           "disclosures published by NSE; open-market trades by key managerial "
                           "personnel, directors and promoters.")
        st.caption(
            f"✅ meets the rule · ❌ fails it · ⚠️ short of it · ➖ not measurable here · ℹ️ context. "
            f"Latest annual statements (year ending {res['fiscal_year_end']}), the book's own 'easy "
            f"method' where it asks for trailing twelve months. Dollar cut-offs converted at "
            f"₹{hough.USD_INR:.0f}/$, not inflation-adjusted. EPS is net income per year-end share, "
            "because yfinance restates bonus issues in some statements and not others.")


def _render_history_notice(hist_df):
    """
    Surface the available history span, esp. yfinance's ~730-day hourly cap.

    Hourly models can train on far less history than daily ones; this makes the
    lookback explicit (and warns when it is short for a meaningful fit).
    """
    info = getattr(hist_df, 'attrs', {}).get('history') if hist_df is not None else None
    if not info or info.get('interval') != '1h':
        return                                   # only relevant for hourly
    rows = info.get('rows', 0)
    sessions = rows // NSE_BARS_PER_SESSION
    msg = (f"🕒 Hourly history: **{rows} bars** over ~{info.get('span_days')} days "
           f"(~{sessions} sessions), {info.get('start')} → {info.get('end')}. "
           "Hourly lookback is **capped by the data provider** (much shorter than "
           "daily), so hourly models train on less history.")
    if rows < 700:        # < ~100 sessions: thin for fitting hourly models
        st.warning(msg + " ⚠️ This is a **short window** for training hourly "
                   "models — treat results with extra caution.")
    else:
        st.caption(msg)


def _render_data_quality(report):
    """
    Render the data-quality gate banner and a per-check expander.

    Surfaces the reviewer's data-quality reason codes: a green caption when the
    data is clean, a warning when there are caveats, and a prominent error when
    the input is too broken to forecast (the app then withholds directional calls
    and falls back to the risk cone — see the main flow).
    """
    if report is None:
        return
    act = gate_action(report)
    level = act["level"]
    headline = f"🩺 {act['headline']}"
    if level == "success":
        st.caption(headline)
    else:
        getattr(st, level)(
            f"**{act['headline']}**"
            + (f" — {act['detail']}" if act.get("detail") else "")
        )
    # Always offer the full reason list for auditability.
    if report.codes:
        with st.expander("Data-quality details"):
            for c in report.codes:
                icon = {"info": "ℹ️", "warn": "⚠️", "critical": "⛔"}.get(
                    c["severity"], "•")
                st.markdown(f"{icon} **{c['code']}** — {c['message']}")
            if report.checks:
                st.caption("Raw checks: " + ", ".join(
                    f"{k}={v}" for k, v in report.checks.items()))


def _format_prediction_table(pred_df, interval):
    """
    A model's forecast rows, readable at any width: dates without a midnight
    time on daily bars, rupee prices to two decimals, plain column names.
    """
    # Only the forecast and its 95% band. Models also return internals such as
    # ``volatility`` (a rupee spread in some, a return standard deviation in
    # others) and Prophet's ``trend``; shown raw under one heading they read as
    # comparable when they are not (README section 68). Nothing downstream
    # reads them; the band carries each model's uncertainty in rupees.
    keep = [c for c in ('datetime', 'predicted_price', 'lower_bound', 'upper_bound') if c in pred_df]
    out = pred_df[keep].copy() if 'predicted_price' in keep else pred_df.copy()
    if 'datetime' in out:
        fmt = '%Y-%m-%d %H:%M' if interval == '1h' else '%a %d-%b-%Y'
        out['datetime'] = pd.to_datetime(out['datetime']).dt.strftime(fmt)
    out = out.rename(columns={'datetime': 'Date' if interval != '1h' else 'Date / time',
                              'predicted_price': 'Forecast (₹)',
                              'lower_bound': 'Lower 95% (₹)',
                              'upper_bound': 'Upper 95% (₹)'})
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            # float32 (Chronos) keeps showing 1210.739990 after round(2).
            out[c] = out[c].astype('float64').round(2)
    return out


def _style_summary(summary_df):
    """
    Color-code and format the summary table for readability.

    Greens/reds the Change (₹), Signal Score (%) and Sentiment columns by sign,
    color-codes the Confidence bucket if present, and formats prices in rupees.
    Returns a pandas Styler, which ``st.dataframe`` renders directly.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of :func:`_build_summary_table`.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled table.
    """
    green, red = '#1e7e34', '#c82333'
    _conf_colors = {
        'High': green, 'Medium': '#b8860b', 'Low': '#d2691e',
        'No edge': red, 'No data': '#6c757d', 'N/A': '#6c757d',
    }

    def _sign_color(v):
        if pd.isna(v):
            return ''
        return f"color: {green if v >= 0 else red}; font-weight: 600"

    grey = '#6c757d'

    def _sentiment_color(v):
        c = {'Bullish': green, 'Bearish': red}.get(v, grey)
        return f"color: {c}; font-weight: 600"

    def _arrow_color(v):
        c = {'▲': green, '▼': red}.get(v, grey)
        return f"color: {c}; font-weight: 600"

    def _conf_color(v):
        return f"color: {_conf_colors.get(v, '#6c757d')}; font-weight: 600"

    # Decision columns first, so a narrow screen shows them without scrolling:
    # the guide says to read Confidence and Basis, not the row order.
    lead = [c for c in ('Model', 'Confidence', 'Direction', 'Predicted Price',
                        'Change (₹)', 'Signal Score (%)', 'Conviction', 'Sentiment',
                        'Basis') if c in summary_df.columns]
    summary_df = summary_df[lead + [c for c in summary_df.columns if c not in lead]]
    styler = (
        summary_df.style
        .map(_sign_color, subset=['Change (₹)', 'Signal Score (%)', 'Conviction'])
        .map(_sentiment_color, subset=['Sentiment'])
        .map(_arrow_color, subset=['Direction'])
        .format({
            'Predicted Price': '₹{:,.2f}',
            'Change (₹)': '{:+,.2f}',
            'Signal Score (%)': '{:+.2f}%',
            'Conviction': '{:+.2f}',
        }, na_rep='—')
    )
    if 'Confidence' in summary_df.columns:
        styler = styler.map(_conf_color, subset=['Confidence'])
    if 'Basis' in summary_df.columns:
        styler = styler.map(
            lambda v: f"color: {grey}; font-style: italic", subset=['Basis'])
    return styler


# -----------------------------------------------------------------------------
# 6. SESSION STATE & CALLBACKS
# -----------------------------------------------------------------------------

if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False


# Sidebar defaults live in session state, set once, so the widgets below take
# no ``default=`` / ``value=``. Setting both (as Reset did) makes Streamlit log
# a "created with a default value but also had its value set via the Session
# State API" warning with a stack trace in the terminal on every Reset.
_DEFAULT_PERIOD = 5                  # the daily horizon a fresh start shows
# Every model starts selected (README section 92); serving.OFF_BY_DEFAULT is empty.
st.session_state.setdefault('sb_models', list(DEFAULT_MODELS))
st.session_state.setdefault('sb_assess', True)


def reset_callback():
    """
    Reset all sidebar controls and session state to defaults.

    Clears the submitted flag, resets company selection to the first
    available, frequency to '1 Day', the horizon to the same 5 days a fresh
    start shows (it used to be 3), and re-selects the default models
    (``serving.DEFAULT_MODELS``: all of them since README section 92).
    """
    st.session_state['submitted'] = False
    companies = get_available_companies()
    if companies:
        st.session_state['sb_company'] = companies[0]
    st.session_state['sb_freq'] = '1 Day'
    st.session_state['sb_period'] = _DEFAULT_PERIOD
    st.session_state['sb_models'] = list(DEFAULT_MODELS)
    st.session_state['sb_assess'] = True


def submit_callback():
    """
    Mark the form submitted and snapshot the current sidebar selections.

    Snapshotting the parameters (rather than reading the live widgets in the
    main panel) means that fiddling with a control *after* submitting does not
    silently recompute results on the changed parameters — results only update
    on the next explicit Submit.
    """
    st.session_state['submitted'] = True
    # One token per Submit: the results page re-runs on every widget change
    # (the threshold box, the sidebar), and the prediction log and console
    # summary must be written once per Submit, not once per re-run.
    import uuid as _uuid
    st.session_state['run_token'] = _uuid.uuid4().hex
    st.session_state['run_company'] = st.session_state.get('sb_company')
    st.session_state['run_freq'] = st.session_state.get('sb_freq', '1 Day')
    st.session_state['run_period'] = st.session_state.get('sb_period', 3)
    st.session_state['run_models'] = list(
        st.session_state.get('sb_models', MODEL_NAMES)
    )
    st.session_state['run_assess'] = st.session_state.get('sb_assess', True)


# -----------------------------------------------------------------------------
# 7. SIDEBAR UI
# -----------------------------------------------------------------------------

local_css()

st.sidebar.image("./UI asset/logo_1.png", width='stretch')
st.sidebar.title("Configuration")
_compute = compute_status()
st.sidebar.caption(f"⚙️ Compute: **{compute_device_label(_compute)}**")
for _hint in _compute["hints"]:
    st.sidebar.caption(f"ℹ️ {_hint}")

company_options = get_available_companies()
selected_company = st.sidebar.selectbox(
    "Search Company", options=company_options, key='sb_company',
)

selected_freq = st.sidebar.radio(
    "Select Frequency", options=['1 Day', '1 Hour'], key='sb_freq',
)

# The horizon is chosen in **trading days** for both frequencies. For daily the
# value is the number of session days (= forecast steps); for hourly it is
# converted to session bars (days × ~7) internally — so the option *values* are
# the actual step counts the models forecast, while the labels read in days.
_B = NSE_BARS_PER_SESSION   # 7 hourly bars per NSE session
from gating import MAX_DIRECTIONAL_SESSIONS as _MAX_CALL, apply_horizon_gate as _apply_horizon_gate
from gating import long_horizon_models as _long_horizon_models
horizon_label = "Forecast Horizon (Trading Days)"
if selected_freq == '1 Day':
    # Beyond 13 sessions the range is the forecast: no up/down call is made
    # (gating.apply_horizon_gate, README section 83).
    horizon_options = [3, 5, 7, 9, 11, 13, 20, 40, 60, 90]  # days = steps
    horizon_fmt = lambda d: (f"{d} days" if d <= _MAX_CALL
                             else f"{d} days (range only, ~{d / 21:.0f} month{'s' if d > 31 else ''})")
    horizon_help = ("Number of future NSE trading days to forecast (weekends and "
                    "exchange holidays are skipped). Beyond 13 days no up/down call "
                    "is made: the 95% range is the forecast.")
else:
    horizon_options = [1 * _B, 2 * _B, 3 * _B, 5 * _B]     # hourly bars = steps
    horizon_fmt = lambda b: (f"{b // _B} day{'s' if b // _B > 1 else ''} "
                             f"({b} hourly bars)")
    horizon_help = ("Chosen in trading days; each NSE session has ~7 hourly bars "
                    "(09:15-15:30), so the model forecasts days × 7 bars. "
                    "Weekends and exchange holidays are skipped.")

# Keep the persisted selection valid when the option set changes (freq toggled),
# since the two frequencies expose different value sets under one widget key.
if st.session_state.get('sb_period') not in horizon_options:
    st.session_state['sb_period'] = horizon_options[1]     # 5 days / 2 days hourly

selected_period = st.sidebar.selectbox(
    horizon_label, options=horizon_options, key='sb_period',
    format_func=horizon_fmt, help=horizon_help,
)

selected_models = st.sidebar.multiselect(
    "Models to run",
    options=MODEL_NAMES,
    key='sb_models',
    help="Running all models can take several minutes (the neural nets are "
         "slow). Deselect models you don't need for a faster run.",
)
st.sidebar.caption(f"{len(selected_models)} of {len(MODEL_NAMES)} models selected")

assess_confidence = st.sidebar.checkbox(
    "Assess forecast confidence",
    key='sb_assess',
    help="Runs a quick walk-forward backtest per model to check whether it "
         "actually beats a random-walk baseline here, and labels each forecast "
         "High / Medium / Low / No edge. Adds time on the first run for a "
         "ticker/horizon (results are cached). Turn off for a faster run.",
)

st.sidebar.markdown("---")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    st.button(
        "Submit", on_click=submit_callback, key="btn_submit",
        width='stretch', type="primary",
    )
with col_btn2:
    st.button(
        "Reset", on_click=reset_callback, key="btn_reset",
        width='stretch', type="secondary",
    )


def _render_forward_record():
    """
    Sidebar panel: the forward record, scored and fed from the app itself.

    Replaces the two scheduled tasks of README sections 77 and 78 (section 82):
    one button scores the forecasts that have matured, one starts this week's
    watchlist forecast in its own process.
    """
    if not USE_PREDICTION_LOG:
        return
    status = _fr.record_status()
    run = _fr.current_run()
    if run:
        head = "running"
    elif status["watchlist_due"]:
        head = "watchlist due"
    else:
        head = "up to date"
    with st.sidebar.expander(f"📒 Forward record ({head})", expanded=False):
        st.caption(
            f"{status['rows']} forecasts logged: {status['resolved']} scored, "
            f"{status['pending']} waiting for their date."
        )
        if st.button("Score matured forecasts", key="btn_fr_score", width='stretch',
                     help="Fetch the prices that have arrived and fill in the "
                          "outcomes of every forecast whose date has passed. Safe "
                          "to run any number of times."):
            with st.spinner("Fetching prices and scoring..."):
                try:
                    res = _fr.score_matured()
                    st.success(f"Scored {res['resolved']}; {res['still_pending']} "
                               "still waiting.")
                except Exception as exc:
                    st.error(f"Scoring failed: {exc}")
        if status["last_watchlist"] is None:
            st.caption("Watchlist: never run. It forecasts 10 fixed companies "
                       "once a week (README section 78).")
        else:
            st.caption(f"Watchlist: last run {status['days_since_watchlist']} days "
                       f"ago ({status['last_watchlist']:%d %b %Y}).")
        if run:
            st.info(f"Watchlist running since {run['started'].replace('T', ' ')}.")
            for line in _fr.last_lines(run["log"]):
                st.caption(line)
        else:
            if st.button("Run this week's watchlist", key="btn_fr_watch",
                         width='stretch', disabled=not status["watchlist_due"],
                         help="Starts run_watchlist.py in its own process: a few "
                              "hours, the app stays usable, and it carries on if "
                              "this tab is closed. Best after the Friday close. "
                              "Enabled once the last run is 7 days old."):
                r = _fr.launch_watchlist()
                st.success(f"Started (process {r['pid']}). Progress shows here.")


_render_forward_record()

# -----------------------------------------------------------------------------
# 8. MAIN UI LAYOUT
# -----------------------------------------------------------------------------

st.title("Multi-Model Forecasting Results")

st.warning(
    "⚠️ **Caution:** These predictions are based on historical data "
    "patterns and mathematical models. They may not reflect real-world "
    "market volatility, news events, or economic changes. "
    "Use for educational purposes only."
)

if st.session_state['submitted']:

    # Use the parameters snapshotted at submit time (see submit_callback), not
    # the live sidebar widgets, so tweaking a control doesn't recompute results.
    run_company = st.session_state.get('run_company')
    run_freq = st.session_state.get('run_freq', '1 Day')
    run_period = st.session_state.get('run_period', 3)
    run_models = st.session_state.get('run_models', list(MODEL_NAMES))
    run_assess = st.session_state.get('run_assess', True)
    backend_interval = '1d' if run_freq == '1 Day' else '1h'
    # Beyond two weeks the band is the forecast, and models whose band failed
    # the long-horizon coverage test are left out (README section 83).
    run_models, _left_out = _long_horizon_models(run_models, run_period, backend_interval)

    # True on the first render after a Submit, False on the re-runs a widget
    # change triggers (see submit_callback).
    _fresh_run = st.session_state.get('seen_token') != st.session_state.get('run_token')
    st.session_state['seen_token'] = st.session_state.get('run_token')

    if _left_out:
        _note = (f"**Left out at {run_period} trading days:** {', '.join(_left_out)}. "
                 "Their 95% range held fewer than 90% of outcomes at 20-90 days on "
                 "NSE data before 2025, so it is not shown as a forecast this far "
                 "ahead (README section 83).")
        (st.warning if not run_models else st.info)(_note)
    if not run_models:
        if not _left_out:
            st.warning("No models selected. Pick at least one model and Submit.")
    else:
        # Frame this Submit in the console so one run's per-model trace is
        # separable from the next in a long scroll.
        _t0 = _time.perf_counter()
        if _fresh_run:
            _console.run_header(run_company, backend_interval, run_period,
                                run_models,
                                timestamp=_dt.datetime.now().strftime("%H:%M:%S"))
        model_results = run_selected_models(
            run_company, run_freq, run_period, run_models,
        )

        if not model_results:
            st.warning("No models generated results successfully.")
        else:
            # Resolve the history sample and last actual price first, so the
            # data-quality gate can run *before* we trust any forecast.
            hist_sample = list(model_results.values())[0]['history']
            price_col_hist = get_col_by_candidates(
                hist_sample, ['adj_close', 'adj close', 'close', 'price'],
            )
            last_actual = (
                float(hist_sample[price_col_hist].iloc[-1])
                if price_col_hist else None
            )

            # Data-quality gate (reviewer request): score the input. Degraded
            # data withholds directional calls and steers the user to the risk
            # cone; broken data abstains. See data_quality.assess_data_quality.
            dq_report = assess_data_quality(
                hist_sample, backend_interval, run_period)
            dq_degraded = dq_report.mode in ('risk_cone_only', 'abstain')

            # Confidence gate: assess each model's recent walk-forward skill.
            conf_map = {}
            # Past the longest directional horizon the call is withheld, so a
            # walk-forward at that horizon would be slow and change nothing.
            _long = backend_interval == '1d' and run_period > _MAX_CALL
            if run_assess and not _long:
                assessed = list(model_results.keys())
                cbar = st.progress(
                    0.0, text="Assessing forecast confidence (walk-forward)...")
                for j, mname in enumerate(assessed):
                    try:
                        conf_map[mname] = _model_confidence(
                            run_company, backend_interval, run_period, mname,
                            USE_MARKET_CONTEXT,
                        )
                    except Exception as exc:
                        conf_map[mname] = {'bucket': 'N/A',
                                           'note': str(exc)[:80]}
                    cbar.progress((j + 1) / len(assessed),
                                  text=f"Assessed {mname} ({j + 1}/{len(assessed)})")
                cbar.empty()

            # On degraded data, withhold every directional call by forcing the
            # gate bucket to "No edge" (pure rule in gating.apply_data_quality_gate,
            # unit-tested) — reuses the abstention gating even if assessment is off.
            conf_map = _apply_dq_gate(conf_map, list(model_results), dq_degraded)

            # Regime-novelty gate: if the current market window is off-manifold
            # relative to what the models were fitted on, every model's
            # backtest-derived bucket is an extrapolation, so an
            # "unprecedented" verdict withholds the directional call the same
            # way degraded data does. An "unusual" verdict only annotates the
            # Basis column — 95th-percentile windows are a normal monthly event
            # in fat-tailed data, so gating on them would abstain far too often.
            novelty = {'available': False, 'verdict': 'unknown'}
            if USE_REGIME_NOVELTY:
                with st.spinner("Checking market-regime novelty..."):
                    novelty = _regime_novelty(
                        run_company, backend_interval, USE_MARKET_CONTEXT)
                conf_map = _apply_novelty_gate(
                    conf_map, list(model_results), novelty)
            conf_map = _apply_horizon_gate(conf_map, list(model_results), run_period,
                                           backend_interval)
            if _long:
                st.info(f"**Long horizon ({run_period} trading days):** no up/down call is "
                        "made past two weeks. Read the **95% range** as the forecast; the "
                        "median line is shown but is not a prediction of direction "
                        "(README section 83).")

            # Show the aggregate skill banner only when the data itself is
            # trustworthy; on degraded data the data-quality banner carries
            # the message and a positive skill banner would contradict it.
            # Drawn after the novelty gate for the same reason: an
            # "unprecedented" regime has just withheld every call, so a banner
            # built from the pre-gate buckets could announce an edge.
            if novelty.get('verdict') == 'unprecedented':
                st.warning(
                    "**Unseen market regime** — the latest window is unlike "
                    "anything the models were fitted on, so every directional "
                    "call is withheld. Treat the **range / interval** as the "
                    "output.")
            elif run_assess and not dq_degraded:
                banner = _honesty_banner(conf_map)
                if banner:
                    getattr(st, banner[0])(banner[1])
            if not _long:
                st.caption(DIRECTION_CAVEAT)

            # ── Forward prediction log ──────────────────────────────────
            # Written AFTER the gates so each row carries the verdicts that were
            # in force when it was served — which is what makes it possible to
            # ask later whether the GATES worked, not just whether the numbers
            # were close. Never raises; a log failure is not a forecast failure.
            _n_logged = 0
            # Once per Submit: changing the threshold or a sidebar control
            # re-runs this page, and each re-run used to append the whole run
            # to the log again under a new run_id.
            if USE_PREDICTION_LOG and last_actual is not None and _fresh_run:
                import uuid as _uuid
                _run_id = _uuid.uuid4().hex[:12]
                for _mname, _mdata in model_results.items():
                    try:
                        _pred = _mdata['prediction']
                        _hist = _mdata['history']
                        _origin = pd.to_datetime(_hist['datetime']).max()
                        _plog.log_forecast(
                            ticker=run_company, interval=backend_interval,
                            horizon=run_period, model=_mname,
                            prediction_df=_pred, anchor_price=float(last_actual),
                            origin_datetime=_origin,
                            confidence_bucket=conf_map.get(_mname, {}).get('bucket'),
                            dq_mode=dq_report.mode,
                            novelty_verdict=novelty.get('verdict'),
                            seed=GLOBAL_SEED, run_id=_run_id)
                        _n_logged += 1
                    except Exception as _log_exc:
                        _console.note("log", f"skipped {_mname} ({_log_exc})")

            if last_actual is not None:
                summary_df = _build_summary_table(
                    model_results, last_actual, conf_map)
                if summary_df is not None:
                    # Hourly horizon is chosen in trading days but counted in
                    # bars internally (~7 bars/session); label it in both.
                    if run_freq == '1 Day':
                        period_label = f"{run_period} trading days"
                    else:
                        _days = max(1, run_period // 7)
                        period_label = (f"{_days} trading day"
                                        f"{'s' if _days > 1 else ''}, "
                                        f"{run_period} hourly bars")

                    # Insert the Confidence bucket + a plain-language reason code
                    # (before Direction) when assessed. The reason ties the label
                    # to the DM test / coverage / data-quality, so no row is ever
                    # presented as a bare (or implied "best") winner.
                    if conf_map:
                        loc = summary_df.columns.get_loc('Direction')
                        summary_df.insert(
                            loc, 'Basis',
                            summary_df['Model'].map(
                                lambda m: _confidence_reason(conf_map.get(m, {}))
                            ),
                        )
                        summary_df.insert(
                            loc, 'Confidence',
                            summary_df['Model'].map(
                                lambda m: conf_map.get(m, {}).get('bucket', '—')
                            ),
                        )

                    st.markdown("### Model Comparison")

                    # Data-quality gate banner first — if the input is degraded
                    # the directional calls below are already withheld.
                    _render_data_quality(dq_report)

                    # Audit: how were these ₹ prices converted? (period-accurate
                    # historical vs degraded live-uniform vs unconverted) and how
                    # much history is available (hourly ~730-day cap).
                    _render_fx_notice(hist_sample)
                    _render_history_notice(hist_sample)
                    _render_dividend_notice(hist_sample, model_results, last_actual)

                    # Reproducibility manifest — seed + versions + params, so any
                    # forecast shown here can be reproduced and traced.
                    _hist_info = getattr(hist_sample, 'attrs', {}).get('history', {})
                    # Architectures: each deep model records what it actually
                    # built on its prediction frame's attrs (see
                    # architectures.attach). Lifting them here closes a hole in
                    # the reproducibility claim — the manifest pinned the seed
                    # and package versions but not the network shape, so it
                    # could not in fact reproduce a deep fit.
                    _archs = {}
                    for _mname, _mdata in model_results.items():
                        _a = getattr(_mdata['prediction'], 'attrs', {}).get(
                            'architecture')
                        if _a:
                            _archs[_mname] = _a
                    _manifest = run_manifest(
                        ticker=run_company, interval=backend_interval,
                        horizon=run_period, seed=GLOBAL_SEED,
                        fetched_at=_hist_info.get('end'),
                        # Which numbers, not just which ticker: adjusted
                        # history is restated on every dividend/split, so the
                        # same ticker+seed can be fed different data (§21).
                        data=hist_sample,
                        training_rows=int(len(hist_sample)),
                        models=list(model_results.keys()),
                        architectures=_arch.manifest_block(_archs),
                        architecture_overrides=_arch.OVERRIDDEN,
                    )
                    with st.expander("🔬 Run details (reproducibility)"):
                        st.caption(
                            f"Seed **{_manifest['seed']}** · Python "
                            f"{_manifest['python']} · {_manifest['training_rows']} "
                            f"training rows · data through {_manifest['fetched_at']} "
                            f"· run {_manifest['run_at_utc'][:19]}Z"
                            + (f" · data `{_manifest['data']['sha256']}`"
                               if (_manifest.get('data') or {}).get('sha256')
                               else "")
                        )
                        st.json(_manifest, expanded=False)

                    # Consensus view — the collective read across models.
                    cons = _build_consensus(summary_df)

                    # ── Console mirror of the decision layer ────────────────
                    # Everything below was previously browser-only: someone
                    # watching cmd saw 19 models announce "Prediction Complete"
                    # and could not tell whether a single one had beaten a
                    # random walk. Rendered from the same conf_map, the same
                    # _honesty_banner and the same _confidence_reason the UI
                    # uses, so the console cannot claim something the page
                    # doesn't. _honesty_banner is re-read here (rather than
                    # reusing the one rendered above) because it must reflect
                    # the buckets AFTER the data-quality and novelty gates.
                    _cbanner = _honesty_banner(conf_map)
                    (_console.run_summary if _fresh_run else (lambda **_k: None))(
                        ticker=run_company, interval=backend_interval,
                        horizon=run_period,
                        dq_mode=dq_report.mode, dq_score=dq_report.score,
                        conf_rows=[
                            (m, conf_map.get(m, {}).get('bucket', '—'),
                             _confidence_reason(conf_map.get(m)))
                            for m in summary_df['Model']
                        ] if conf_map else [],
                        consensus=cons,
                        verdict=_console.strip_markdown(
                            _cbanner[1] if _cbanner else ''),
                        novelty_verdict=(novelty.get('verdict')
                                         if USE_REGIME_NOVELTY else None),
                        log_path=_plog.LOG_FILE if _n_logged else None,
                        n_logged=_n_logged,
                        elapsed=_time.perf_counter() - _t0,
                    )
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Last Actual Price", f"₹{last_actual:,.2f}")
                    m2.metric(
                        f"Median Forecast ({period_label})",
                        f"₹{cons['median_price']:,.2f}",
                        f"{cons['median_pct']:+.2f}%",
                    )
                    m3.metric("Models Bullish",
                              f"{cons['n_up']} / {cons['n_models']}")
                    m4.metric("Models Bearish",
                              f"{cons['n_down']} / {cons['n_models']}")
                    if cons.get('n_noedge', 0) > 0:
                        st.caption(
                            f"↳ **{cons['n_noedge']} / {cons['n_models']}** "
                            "model(s) showed **no edge** over a random walk in "
                            "the confidence check, so their directional call is "
                            "withheld (shown as “No edge / –”) rather than "
                            "asserting a Bullish/Bearish view the evidence "
                            "doesn’t support."
                        )

                    st.dataframe(
                        _style_summary(summary_df),
                        hide_index=True,
                        width='stretch',
                        # Pinned, so the model name stays on screen when the
                        # table is scrolled sideways on a phone.
                        column_config={"Model": st.column_config.TextColumn(pinned=True)},
                    )
                    # Standing data-snooping disclaimer: operationalize the
                    # project's Reality-Check / SPA finding so the table is never
                    # read as crowning a "best" model. The lowest-RMSE row is not
                    # a confirmed winner once you correct for trying many models.
                    st.info(
                        "🎯 **This is a risk instrument, not an oracle.** Ranking "
                        "models by forecast or backtest RMSE does **not** identify "
                        "a truly superior model here: across the full set, "
                        "data-snooping-robust tests (White's Reality Check & "
                        "Hansen's SPA) found **no selection-bias-free edge over a "
                        "random walk** at these horizons. So read the **Confidence** "
                        "and **Basis** columns, not the row order — and treat the "
                        "**calibrated range / risk cone** as the deliverable, not "
                        "the point forecast."
                    )
                    st.caption(
                        "- **Signal Score (%)** is each model's forecasted "
                        "percentage change from the last actual price to the "
                        f"end of the {period_label} horizon — the "
                        "value the call is based on.\n"
                        "- **Sentiment** is "
                        "**Bullish** when the Signal Score is ≥ 0% and "
                        "**Bearish** when it is < 0% — but is shown as "
                        "**“No edge”** (with a “–” Direction) whenever the "
                        "confidence check finds the model no better than a "
                        "random walk, so no direction is asserted on noise.\n"
                        "- **Conviction** is the "
                        "forecast move divided by half the model's own 95% "
                        "interval width: it has the same sign as the move, and "
                        "**|Conviction| ≥ 1** means the move is larger than the "
                        "model's uncertainty (the current price falls outside "
                        "the 95% band) — a higher-conviction call. The further "
                        "from 0, the stronger the signal; values near 0 are "
                        "within noise. (Shown as “—” for models with no interval.)"
                    )
                    if conf_map:
                        st.caption(
                            "**Confidence** comes from a quick walk-forward "
                            "backtest on recent history:\n"
                            "- **High** — clearly beats the random walk with a "
                            "significant Diebold-Mariano test and sound coverage.\n"
                            "- **Medium** — a modest, weakly-significant edge.\n"
                            "- **Low** — better than the walk but within noise.\n"
                            "- **No edge** — no better than simply assuming the "
                            "price stays put.\n\n"
                            "For Low / No-edge rows, trust the **range**, not the "
                            "point forecast."
                        )
                    st.info(
                        "These are statistical forecasts conditioned on past "
                        "prices — read them as a **distribution of plausible "
                        "outcomes** (the shaded 95% band on each chart, and the "
                        "range around the median), not a precise prediction. "
                        "The grey dotted line on each chart is the naive "
                        "random-walk baseline for reference."
                    )
                    st.divider()

                    # ── Risk & Probability (distribution-first headline) ──────
                    st.markdown(f"### Risk & Probability "
                                f"(next {period_label})")
                    # In a narrow column: full-width, this one small box stretched
                    # across a 2,560-px screen.
                    _thr_col, _ = st.columns([1, 3])
                    thr = _thr_col.number_input(
                        "Move threshold for probabilities (%)",
                        min_value=0.5, max_value=25.0, value=2.0, step=0.5,
                        help="Used for P(gain >) and P(loss >) below.",
                    )
                    # Results inside the horizon (Benklifa Ch. 5, a "known
                    # unknown"): widen the panel by the measured results-day
                    # variance (risk_metrics.results_widening). Charts and the
                    # per-model tables are left as the models produced them.
                    _h_days = (run_period if backend_interval == '1d'
                               else run_period / NSE_BARS_PER_SESSION)
                    _ticker = getattr(hist_sample, 'attrs', {}).get('ticker')
                    _pred0 = list(model_results.values())[0]['prediction']
                    _fdates = pd.to_datetime(_pred0['datetime']).dt.normalize()
                    _last_day = (pd.to_datetime(hist_sample['datetime'].iloc[-1]).normalize()
                                 if 'datetime' in hist_sample else _fdates.min())
                    _sessions = [_last_day] + sorted(set(_fdates))
                    _meetings = (_results_dates(
                        _ticker, _last_day.date().isoformat(),
                        (_last_day + pd.Timedelta(days=45)).date().isoformat())
                        if _ticker else [])
                    _n0, _n1, _first_mtg = _results_sessions(_sessions, _meetings)
                    _scale = _results_widening(_h_days, _n0, _n1)
                    if _scale > 1.0:
                        st.info(
                            f"📅 **Results on {_first_mtg:%d %b %Y}** (NSE's board-meeting "
                            "calendar) fall inside this horizon. On NSE a results day "
                            f"carries {RESULTS_DAY0_VAR:.1f}× an ordinary session's variance, "
                            f"and the session after it {RESULTS_DAY1_VAR:.1f}× (many companies "
                            "report after the close). A 95% band that ignores this covered "
                            "only 82-93% of such windows, so **this panel is widened "
                            f"×{_scale:.2f}** (back to ~95%, measured out of sample). The "
                            "per-model charts below are not widened.")
                    risk = build_risk_panel(model_results, last_actual, thr, scale=_scale)
                    if risk is None:
                        st.caption("No model emitted an interval, so probability "
                                   "estimates are unavailable for this run.")
                    else:
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("P(price up)", f"{risk['p_up'] * 100:.0f}%")
                        r2.metric(f"P(gain > +{thr:.1f}%)",
                                  f"{risk['p_gain'] * 100:.0f}%")
                        r3.metric(f"P(loss > {thr:.1f}%)",
                                  f"{risk['p_loss'] * 100:.0f}%")
                        r4.metric("95% downside (VaR)",
                                  f"₹{risk['var5_price']:,.2f}",
                                  f"{risk['var5_pct'] * 100:+.1f}%")
                        st.markdown(
                            f"**Likely range (95%):** ₹{risk['range_low']:,.2f} "
                            f"– ₹{risk['range_high']:,.2f}"
                        )
                        # Stop-loss read (Mladjenovic Ch. 17): a stop triggers
                        # on the PATH, so its hit chance is about twice the
                        # end-point P(loss) above (risk_metrics.stop_hit_probability).
                        if np.isfinite(risk.get('p_stop_hit', float('nan'))):
                            s1, s2 = st.columns(2)
                            s1.metric(f"P(a −{thr:.1f}% stop is hit)",
                                      f"{risk['p_stop_hit'] * 100:.0f}%",
                                      help="A stop triggers if the price TOUCHES "
                                           "the level at any point before the "
                                           "horizon, not only if it ends there, so "
                                           "this is about twice P(loss). Measured "
                                           "on NSE: on target in a crash, about a "
                                           "third too high in a rising market.")
                            _s5 = risk.get('stop_1in5_pct', float('nan'))
                            if np.isfinite(_s5) and _s5 > 0:
                                s2.metric("Stop with a 1-in-5 chance of being hit",
                                          f"−{_s5 * 100:.1f}%  (₹{last_actual * (1 - _s5):,.2f})",
                                          help="Sized by this stock's volatility "
                                               "over this horizon, rather than one "
                                               "percentage for every stock: on NSE a "
                                               "10% stop is hit within a month 11% "
                                               "of the time on the quietest fifth of "
                                               "names and 42% on the most volatile.")
                        # Trailing stop and target touch (Ch. 17's trailing stops
                        # and limit orders): both trigger on the path too.
                        if np.isfinite(risk.get('p_trail_hit', float('nan'))):
                            t1, t2 = st.columns(2)
                            t1.metric(f"P(a −{thr:.1f}% trailing stop is hit)",
                                      f"{risk['p_trail_hit'] * 100:.0f}%",
                                      help="A trailing stop sits x% below the "
                                           "highest price since you set it, so it "
                                           "rises with the stock and is hit more "
                                           "often than a fixed stop. On NSE the "
                                           "book's 8% trailing stop was hit within "
                                           "a month 72% of the time on the median "
                                           "stock, and within a quarter 96%.")
                            t2.metric(f"P(+{thr:.1f}% is touched)",
                                      f"{risk['p_target_touch'] * 100:.0f}%",
                                      help="A sell order (limit order) at +x% "
                                           "fills if the price TOUCHES it before "
                                           "the horizon: about twice P(gain), "
                                           "which only counts where it ends. "
                                           "Measured on NSE: well calibrated in a "
                                           "calm market and a crash, a little low "
                                           "at a month.")
                        # Thomsett's break-even (Stock Market Math, Ch. 4): the
                        # move a trade needs to come out level after round-trip
                        # costs, tax and inflation, and the chance it gets there.
                        _be = _breakeven_move(_h_days)
                        _risk_be = build_risk_panel(model_results, last_actual,
                                                    _be['move'] * 100.0, scale=_scale)
                        if _risk_be is not None:
                            b1, b2 = st.columns(2)
                            b1.metric("Break-even move (costs, tax, inflation)",
                                      f"+{_be['move'] * 100:.2f}%  (₹{last_actual * (1 + _be['move']):,.2f})",
                                      help=f"Round-trip delivery costs "
                                           f"{_be['cost'] * 100:.2f}% (statutory, "
                                           "mostly STT, before brokerage and "
                                           "spread) plus inflation over the "
                                           f"horizon grossed up for tax "
                                           f"({_be['inflation'] * 100:.0f}% a year, "
                                           f"{_be['tax'] * 100:.1f}% short-term "
                                           "gains tax): "
                                           f"{_be['inflation_part'] * 100:.2f}%. "
                                           "Below this a winning trade still loses "
                                           "money in real, after-tax terms.")
                            b2.metric("P(the move clears break-even)",
                                      f"{_risk_be['p_gain'] * 100:.0f}%",
                                      help="The chance the price ENDS above the "
                                           "break-even move, from the same "
                                           "distribution as the rest of this "
                                           "panel. Compare it with P(price up): "
                                           "the gap is what costs take.")
                        # Market breadth (Thomsett Ch. 8): context for the whole
                        # market, not an input to this stock's cone.
                        _render_option_view(_ticker, _h_days, thr, risk, _meetings)
                        _mb = _market_breadth(_dt.date.today().isoformat())
                        if _mb:
                            st.caption(
                                "🌐 " + _mb['message'] + " Weak breadth predicts a "
                                "more volatile market beyond what past volatility "
                                "and India VIX say, but for a single stock it "
                                "improved the volatility forecast by under 1%, so "
                                "this stock's range is not adjusted for it.")
                        # Downside-risk row (B2) — only on the empirical path,
                        # where the up/down asymmetry is a real read of the
                        # fat-tailed distribution rather than a symmetric Gaussian.
                        if risk.get('method') == 'empirical' and 'cvar5_price' in risk:
                            d1, d2, d3, d4 = st.columns(4)
                            d1.metric("Expected shortfall (worst 5%)",
                                      f"₹{risk['cvar5_price']:,.2f}",
                                      f"{risk['cvar5_pct'] * 100:+.1f}%")
                            d2.metric("Downside vol",
                                      f"{risk['down_semidev'] * 100:.1f}%")
                            d3.metric("Upside vol",
                                      f"{risk['up_semidev'] * 100:.1f}%")
                            skew = risk.get('vol_skew', float('nan'))
                            d4.metric("Vol skew (down/up)",
                                      "—" if skew != skew else f"{skew:.2f}",
                                      help="> 1 means the downside is the fatter "
                                           "side of the outcome distribution.")
                        if risk.get('method') == 'empirical':
                            src = ", ".join(risk.get('source_models', [])) or "HAR-RV"
                            st.caption(
                                "Probabilities are counted **directly off the "
                                "empirical, fat-tailed return distribution** from "
                                f"the conformal volatility model ({src}) — no "
                                "normal-distribution assumption, so the tail risk "
                                "the model estimated is preserved. **95% downside "
                                "(VaR)** is the empirical 5th-percentile price — a "
                                "worst-case-but-not-extreme reference; **expected "
                                "shortfall** is the *average* outcome in that worst "
                                "5%, and **downside / upside vol** split the "
                                "distribution's dispersion (their ratio, **vol "
                                "skew**, exceeds 1 when the downside is the fatter "
                                "side). Probabilities reflect model uncertainty, not "
                                "real-world certainty."
                            )
                        else:
                            st.caption(
                                f"⚠️ Approximate read across {risk['n_used']} "
                                "models: no fat-tailed volatility model (e.g. "
                                "**HAR-RV**) was in this run, so each model's 95% "
                                "band is approximated as a **log-normal** "
                                "distribution and the median probability is shown. "
                                "Add **HAR-RV** to the selection for the exact, "
                                "fat-tailed empirical read. **95% downside (VaR)** "
                                "is the 5th-percentile price. Probabilities "
                                "reflect model uncertainty, not certainty."
                            )
                    st.divider()

                    # What the price models cannot see (Mladjenovic): the
                    # company's statements, and the risks no forecast covers.
                    _render_oneil_check(hist_sample, backend_interval)
                    _render_company_check(hist_sample, last_actual,
                                          run_period, backend_interval)
                    _render_listing_dividends(hist_sample, last_actual)
                    _render_hough_check(hist_sample, backend_interval)
                    st.divider()

            # How history is drawn on every chart below, and every model at once.
            import charts as _charts
            _style = "candles" if st.radio(
                "History drawn as", ["Line", "Candles"], horizontal=True, key="chart_style",
                help="Candles show each session's open, high, low and close, adjusted for "
                     "dividends and splits like the forecasts.") == "Candles" else "line"
            if len(model_results) > 1:
                st.markdown("### All models together")
                _cmp = _charts.comparison_chart(model_results, run_company, run_freq,
                                                lookback=30, style=_style)
                if _cmp is not None:
                    st.plotly_chart(_cmp, key="chart_all_models", width='stretch')
                    st.caption("Each model's median path from the last close, the median of "
                               "them (black) and the median of their 95% ranges (grey), against "
                               "the random walk. Agreement between models is not evidence of "
                               "skill: read it with the Confidence column above.")
                st.divider()

            for model_name, data in model_results.items():

                st.markdown(f"### Model: {model_name}")

                hist_df = data['history']
                pred_df = data['prediction']

                # A band far from what the stock's own volatility implies is
                # flagged: coverage alone cannot see a band wide enough to cover
                # everything (README section 85).
                _q, _odd = _serving.band_sanity(pred_df, hist_df)
                if _odd:
                    st.warning(
                        f"**Implausible range:** this model's 95% range is {_q:.1f}x "
                        f"the width this stock's own volatility over the last "
                        f"{_serving.BAND_SANITY_WINDOW} sessions implies for a no-change "
                        "forecast. Treat it with caution (README section 85).")
                    if _fresh_run:
                        _console.note("band", f"{model_name}: range {_q:.1f}x the volatility-implied width")
                if model_name in _serving.MODEL_NOTES:
                    st.info(_serving.MODEL_NOTES[model_name])

                # Chart and table as tabs, each at the full width, instead of
                # side by side: in a 2:3 split the table showed two of its
                # columns on a laptop and the chart was cramped on a tablet.
                tab_chart, tab_table = st.tabs(["📈 Chart", "📋 Predicted values"])

                with tab_table:
                    # Size the table to exactly the forecast rows (one per selected
                    # day/bar) so there are no empty filler rows; cap so a long
                    # hourly horizon scrolls instead of dominating the page.
                    _shown = _format_prediction_table(pred_df, backend_interval)
                    _n_rows = len(_shown)
                    _tbl_height = min(600, (_n_rows + 1) * 35 + 3)
                    st.dataframe(_shown, height=_tbl_height, width='stretch',
                                 hide_index=True)

                with tab_chart:
                    try:
                        fig = plot_predictions_with_history(
                            original_df=hist_df,
                            prediction_df=pred_df,
                            company_name=run_company,
                            interval=run_freq,
                            lookback=50,
                            style=_style,
                        )
                        st.plotly_chart(
                            fig, key=f"chart_{model_name}", width='stretch',
                        )
                    except Exception as e:
                        st.error(f"Error plotting graph for {model_name}: {e}")

                st.divider()

else:
    st.info("Select parameters and click Submit to generate forecasts.")



<!---
Laxmi530/Laxmi530 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
