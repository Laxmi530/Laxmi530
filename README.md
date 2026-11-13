- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Amazon Chronos-2 foundation model, run zero-shot with engineered covariates.

Pretraining contamination -- read this before trusting a Chronos result
------------------------------------------------------------------------
Every other model in this project is fitted inside the walk-forward, so a
backtest can only reward it for what it learned from the past. A pretrained
foundation model is different: it may score well because its pretraining
corpus contained the very series it is now being tested on, which is
memorisation rather than forecasting. (*Large Language Models in Finance*,
§13.4.6, writes observed skill as forecasting + contamination + noise.)

The local model card lists the training sources (a subset of Chronos Datasets,
a subset of GIFT-Eval Pretrain, and synthetic data) and cites a technical
report dated October 2025. It gives no data cutoff and does not say whether
any NSE equity series is included, so the contamination term is UNKNOWN for
this project -- not zero.

That leaves the current verdict intact: contamination can only inflate apparent
skill, and Chronos-2 shows no edge over the random walk (DM p = 0.61). It does
NOT leave a future positive result intact. If Chronos ever appears to beat the
random walk, re-run the comparison on data published after October 2025 before
believing it; the bhavcopy cache already covers 2025-11 to 2026-08. That span
is inside the reserved final test period (bhavcopy.HOLDOUT_START), so open it
deliberately with allow_holdout=True and a purpose; the opening is logged.
"""

import contextlib
import io
import os
import pandas as pd
import numpy as np
from model_utils import wilder_rsi
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

model_chronos_bolt_small = r"C:\Users\lrouta001\OneDrive - PwC\Desktop\AI Models\autogluon-chronos-bolt-small"
# The local weights folder differs from machine to machine; set
# STOCKAPP_CHRONOS_PATH to point at it elsewhere (README section 79).
model_amazon_chronos_2 = os.environ.get(
    "STOCKAPP_CHRONOS_PATH",
    r"C:\Users\lrouta001\OneDrive - PwC\Desktop\AI Models\amazon-chronos-2")

# ─── Constants ──────────────────────────────────────────────────────────────────
_VOLATILITY_WINDOW = 10
_SMA_SHORT = 5
_SMA_LONG = 20
_RSI_PERIOD = 14
_MAX_CONTEXT_DAILY = 2000
_MAX_CONTEXT_HOURLY = 4000
_QUANTILE_LEVELS = [0.025, 0.5, 0.975]

_pipeline_cache = None


def _get_pipeline():
    """
    Lazily load and cache the Chronos-2 pipeline.

    Loads the pretrained model on first call and caches it for
    subsequent calls, avoiding the memory and startup cost when
    the module is imported but prediction is not invoked.

    Returns
    -------
    Chronos2Pipeline
        Loaded and ready-to-use Chronos-2 pipeline instance.
    """
    global _pipeline_cache
    if _pipeline_cache is None:
        from chronos import BaseChronosPipeline

        # Use the GPU when one is available (Chronos-2 is a transformer that
        # benefits markedly from it); fall back to CPU otherwise.
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        print(f"Loading Chronos-2 pipeline on {device}...")
        _pipeline_cache = BaseChronosPipeline.from_pretrained(
            model_amazon_chronos_2, device_map=device,
        )
        print("Chronos-2 pipeline loaded.")
    return _pipeline_cache


# ─── Feature Engineering ────────────────────────────────────────────────────────

def _add_covariates(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Add engineered covariates as past-only context for Chronos-2.

    Chronos-2 natively supports multivariate input: any numeric column
    in the DataFrame (besides the target) is used as a past-only
    covariate. Features are chosen to carry genuinely new information
    beyond raw OHLCV, using scale-invariant representations that
    generalise across price levels.

    Covariates added:
    - log_return : single-period log return
    - volatility : rolling standard deviation of log returns
    - log_volume_ratio : log(volume / 20-period MA), centered at 0
    - range_pct : (high - low) / price, normalized intraday range
    - rsi : Relative Strength Index via Wilder smoothing
    - sma_ratio : short SMA / long SMA, captures trend direction
    - momentum_5 : 5-period log return for medium-term trend
    - dayofweek, month : calendar seasonality
    - hour : intra-day seasonality (hourly only)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'adj_close' and optionally 'volume',
        'high', 'low', and a datetime column.
    interval : str
        '1d' or '1h'; determines whether 'hour' covariate is added.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional covariate columns, NaN/Inf cleaned
        via forward-fill then back-fill.
    """
    df = df.copy()
    price = df["adj_close"].astype(float)

    df["log_return"] = np.log(price / price.shift(1))

    df["volatility"] = (
        df["log_return"]
        .rolling(_VOLATILITY_WINDOW, min_periods=1)
        .std()
    )

    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(_SMA_LONG, min_periods=1).mean()
        df["log_volume_ratio"] = np.log(
            (vol / vol_ma.replace(0, 1)).clip(lower=0.01)
        )

    if "high" in df.columns and "low" in df.columns:
        df["range_pct"] = (
            (df["high"].astype(float) - df["low"].astype(float)) / price
        )

    df["rsi"] = wilder_rsi(price, period=_RSI_PERIOD)

    sma_short = price.rolling(_SMA_SHORT, min_periods=1).mean()
    sma_long = price.rolling(_SMA_LONG, min_periods=1).mean()
    df["sma_ratio"] = sma_short / sma_long.replace(0, 1)

    df["momentum_5"] = np.log(price / price.shift(5))

    ts_col = "original_datetime" if "original_datetime" in df.columns else "datetime"
    ts = pd.to_datetime(df[ts_col])
    df["dayofweek"] = ts.dt.dayofweek.astype(float)
    df["month"] = ts.dt.month.astype(float)
    if interval == "1h":
        df["hour"] = ts.dt.hour.astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    return df


# ─── Data Preparation ───────────────────────────────────────────────────────────

def prepare_timeseries_data(
    df: pd.DataFrame, company_name: str, interval: str = "1d",
) -> pd.DataFrame:
    """
    Prepare timeseries data for the Chronos-2 pipeline.

    Both intervals keep the bars as they traded and give the model a plain
    step counter (required by Chronos-2's regular-grid assumption): the
    real timestamps go to 'original_datetime', and 'datetime' becomes a
    regular grid, business days for daily data (no rows are added for
    exchange holidays, README section 123) and 1-hour spacing for hourly.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with a 'datetime' column and OHLCV data.
    company_name : str
        Company display name, used to derive the ticker for the 'id'
        column required by Chronos-2's ``predict_df`` API.
    interval : str, optional
        '1d' for daily or '1h' for hourly. Default is '1d'.

    Returns
    -------
    pd.DataFrame
        DataFrame ready for Chronos-2 with proper timestamp frequency,
        an 'id' column, and an 'original_datetime' column with the real
        timestamps.

    Raises
    ------
    ValueError
        If *interval* is not '1d' or '1h'.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if interval == "1d":
        df = df.drop_duplicates(subset="datetime", keep="last")
        df["datetime"] = df["datetime"].dt.normalize()
        df = df.drop_duplicates(subset="datetime", keep="last").reset_index(drop=True)
        df = df.ffill()
        # The sessions as they traded, on a plain step counter (README section
        # 123). This used to be df.asfreq("B").ffill(): every NSE holiday became
        # a flat, zero-return day in the history, about 14 a year, and the
        # forecast's dates ran on weekdays, so a five-step forecast from
        # 2026-11-06 included the 10 November holiday. The real dates are kept
        # in original_datetime, for the calendar covariates and for the
        # forecast's dates (_real_future_dates).
        df["original_datetime"] = df["datetime"]
        df["datetime"] = pd.bdate_range(start=df["datetime"].iloc[0], periods=len(df))

    elif interval == "1h":
        df = df.drop_duplicates(subset="datetime", keep="last")
        df["original_datetime"] = df["datetime"]
        start_time = df["datetime"].iloc[0]
        uniform_timestamps = pd.date_range(
            start=start_time, periods=len(df), freq="h",
        )
        df["datetime"] = uniform_timestamps
        print(
            f"Hourly data: Replaced {len(df)} irregular timestamps "
            f"with uniform 1h spacing"
        )
    else:
        raise ValueError(f"Invalid interval: {interval}. Use '1d' or '1h'.")

    # The id only labels the single series for predict_df; any string works.
    # It used to be get_ticker(company_name), which downloads the full NSE and
    # BSE listings on every call -- at every walk-forward origin of the
    # confidence gate -- so a network blip failed the forecast for a label.
    df["id"] = str(company_name or "ID")
    print(f"Interval: {interval} | Final shape: {df.shape}")
    return df


# ─── Output Cleaning ────────────────────────────────────────────────────────────

def clean_prediction_output(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Chronos-2 prediction output into the standard model format.

    Renames columns to match the project-wide convention, drops
    metadata columns, and derives a ``volatility`` column from the
    confidence interval width for consistency with other models.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Raw prediction DataFrame from ``pipeline.predict_df``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'volatility'].
    """
    pred_df = pred_df.copy()

    if "predictions" in pred_df.columns:
        pred_df = pred_df.rename(columns={"predictions": "predicted_price"})

    for col in ("target_name", "id"):
        if col in pred_df.columns:
            pred_df = pred_df.drop(columns=[col])

    # Quantile columns are named by their level. Older chronos releases used
    # float names (0.025); chronos-forecasting 2.2.2 uses strings ('0.025').
    # Matching only floats silently dropped the band: no lower/upper bound, a
    # zero volatility column, no 95% band on the chart and no Conviction.
    def _level(c):
        try:
            v = float(c)
        except (TypeError, ValueError):
            return None
        return v if 0.0 < v < 1.0 else None

    q_cols = [c for _, c in sorted((_level(c), c) for c in pred_df.columns
                                   if _level(c) is not None)]
    if len(q_cols) >= 2:
        pred_df = pred_df.rename(
            columns={q_cols[0]: "lower_bound", q_cols[-1]: "upper_bound"},
        )
        middle_cols = q_cols[1:-1]
        if middle_cols:
            pred_df = pred_df.drop(columns=middle_cols, errors="ignore")

    if "lower_bound" in pred_df.columns and "upper_bound" in pred_df.columns:
        ci_width = pred_df["upper_bound"] - pred_df["lower_bound"]
        pred_df["volatility"] = (ci_width / (2 * 1.96)).clip(lower=0.0)
    else:
        pred_df["volatility"] = 0.0

    numeric_cols = pred_df.select_dtypes(include=["float64", "float32"]).columns
    pred_df[numeric_cols] = pred_df[numeric_cols].round(2)
    return pred_df.reset_index(drop=True)


# ─── Forecast Dates ─────────────────────────────────────────────────────────────

def _real_future_dates(pred_df, prepared, prediction_length, interval):
    """
    The forecast dated on the next NSE sessions (daily) or hourly bars after
    the last real bar, from ``model_utils.generate_future_dates`` as every other
    model's forecast is (README section 123). The model itself sees a step
    counter, so its own timestamps are not dates. The hourly mapping this
    replaced skipped weekends but not exchange holidays.
    """
    from model_utils import generate_future_dates
    if "original_datetime" not in prepared.columns or "datetime" not in pred_df.columns:
        return pred_df
    last = pd.Timestamp(prepared["original_datetime"].iloc[-1])
    dates = generate_future_dates(last, prediction_length, "daily" if interval == "1d" else "hourly")
    pred_df = pred_df.copy()
    if len(pred_df) == len(dates):
        pred_df["datetime"] = pd.to_datetime(dates)
    return pred_df


# ─── Switches of README section 122 ────────────────────────────────────────────

#: The configuration before README section 122: OHLCV and the engineered
#: covariates as past-only covariates, the model's own 2.5%/97.5% quantiles as
#: the band, and its median as the centre.
CHRONOS_LEGACY = dict(covariates=True, conformal=False, shrink=False)
#: Calibration origins for ``conformal`` and ``shrink``: as Prophet's (README
#: section 116), 20 past origins 5 sessions apart.
N_CAL = 20
CAL_STRIDE = 5


def resolve_switches(interval, horizon, covariates=None, conformal=None, shrink=None):
    """
    The switches a run uses. A switch left as None takes CHRONOS_LEGACY's
    value; an explicit value is kept. ``conformal`` and ``shrink`` are daily
    only. ``interval`` and ``horizon`` are taken so a default by bar size or
    horizon can be added without changing callers.
    """
    given = dict(covariates=covariates, conformal=conformal, shrink=shrink)
    out = {k: CHRONOS_LEGACY[k] if v is None else bool(v) for k, v in given.items()}
    daily = interval in ("1d", "1D", "daily", "1 Day")
    if (out["conformal"] or out["shrink"]) and not daily:
        raise ValueError("conformal and shrink are for daily bars only")
    return out


def _target_only(frame):
    """The predict_df frame without any covariate: the id, the timestamps and the target."""
    return frame[[c for c in ("id", "datetime", "adj_close", "original_datetime") if c in frame.columns]]


def _frame(df, company_name, interval, covariates=True):
    """The input frame for predict_df, built as run_chronos_prediction builds it."""
    max_obs = _MAX_CONTEXT_DAILY if interval == "1d" else _MAX_CONTEXT_HOURLY
    if len(df) > max_obs:
        df = df.tail(max_obs).reset_index(drop=True)
    with contextlib.redirect_stdout(io.StringIO()):
        out = prepare_timeseries_data(df, company_name, interval=interval)
    out = _add_covariates(out, interval) if covariates else _target_only(out)
    return out.drop(columns=["original_datetime"], errors="ignore")


def calibration(df, horizon, covariates=True, n_cal=N_CAL, stride=CAL_STRIDE):
    """
    Daily bars: at ``n_cal`` past origins ``stride`` sessions apart, the last
    ending ``horizon`` sessions before the end of ``df``, Chronos-2's log median
    for the next ``horizon`` steps, the log actual prices of the next
    ``horizon`` sessions and the log price at the origin, as
    fb_prophet_model.calibration_logs returns them (the served forecast is
    scored the same way). The origins are forecast in one predict_df call;
    cross-learning is off, so each is forecast from its own history only.
    Returns (actual, pred, base), or None when there is too little history.
    """
    import fb_prophet_model as F
    raw, origins = F.calibration_origins(df, horizon, n_cal=n_cal, stride=stride)
    if origins is None:
        return None
    frames = [_frame(raw.iloc[:t + 1], f"cal{i}", "1d", covariates) for i, t in enumerate(origins)]
    pred_df = _get_pipeline().predict_df(
        pd.concat(frames, ignore_index=True), prediction_length=horizon,
        quantile_levels=[0.5], id_column="id", timestamp_column="datetime",
        target="adj_close", validate_inputs=True)
    pred = np.vstack([np.log(pred_df.loc[pred_df["id"] == f"cal{i}", "predictions"].to_numpy(float)[:horizon])
                      for i in range(len(origins))])
    close = raw["adj_close"].to_numpy(float)
    actual = np.vstack([np.log(close[t + 1:t + 1 + horizon]) for t in origins])
    base = np.log(close[origins])
    return actual, pred, base


def recalibrate(out, last_price, cal, conformal, shrink, alpha=0.05):
    """
    The forecast ``out`` with its band replaced by the finite-sample
    ``1 - alpha`` quantile of the calibration errors, in log price, around its
    centre (``conformal``); with ``shrink`` the centre is first shrunk toward
    the last price by a slope per step, beta = clip(sum(p y) / sum(p p), 0, 1),
    and the band comes from the shrunk errors (Campbell and Thompson 2008;
    fb_prophet_model.shrink_band, README section 116). ``cal`` is
    :func:`calibration`'s output; None leaves ``out`` as it is.
    """
    if cal is None or not (conformal or shrink):
        return out
    import fb_prophet_model as F
    actual, pred, base = cal
    if shrink:
        beta, half = F.shrink_band(actual, pred, base, alpha=alpha)
    else:
        n = actual.shape[0]
        beta = np.ones(actual.shape[1])
        half = np.quantile(np.abs(actual - pred), min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), axis=0)
    b0 = np.log(float(last_price))
    h = len(half)
    centre = b0 + beta * (np.log(out["predicted_price"].to_numpy(float)[:h]) - b0)
    out = out.iloc[:h].copy()
    out["predicted_price"] = np.exp(centre)
    out["lower_bound"] = np.exp(centre - half)
    out["upper_bound"] = np.exp(centre + half)
    out["volatility"] = (out["upper_bound"] - out["lower_bound"]) / (2 * 1.96)
    out.attrs["switches"] = dict(conformal=bool(conformal), shrink=bool(shrink), beta=beta.tolist())
    return out.reset_index(drop=True)


def chronos_paths(df, interval, prediction_length, company_name="ID", covariates=True,
                  variants=((False, False),)):
    """
    Forecasts for several (conformal, shrink) settings from one forecast and,
    when needed, one batch of calibration forecasts: {(conformal, shrink):
    DataFrame}. ``covariates`` changes the inputs, so it is fixed for the call.
    With ``variants=((False, False),)`` and ``covariates=True`` this is the
    served model.
    """
    base = run_chronos_prediction(df, interval, prediction_length, company_name,
                                  covariates=covariates)
    cal = None
    if any(c or s for c, s in variants):
        cal = calibration(df, prediction_length, covariates=covariates)
    last = float(pd.to_numeric(df["adj_close"], errors="coerce").dropna().iloc[-1])
    return {(bool(c), bool(s)): recalibrate(base, last, cal, c, s) for c, s in variants}


def run_chronos_prediction(
    df: pd.DataFrame,
    interval: str,
    prediction_length: int,
    company_name: str = "ID",
    covariates=None,
    conformal=None,
    shrink=None,
) -> pd.DataFrame:
    """
    Full Chronos-2 prediction pipeline.

    Orchestrates context limiting, timestamp preparation, covariate
    engineering, model inference, timestamp remapping (hourly), and
    output cleaning. The pipeline lazily loads the Chronos-2 model
    on first invocation to avoid unnecessary memory usage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with columns datetime, open, high, low,
        close, adj_close, volume.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    prediction_length : int
        Number of future steps to predict.
    company_name : str, optional
        Company display name for ticker resolution. Default is 'ID'.
    covariates, conformal, shrink : bool, optional
        README section 122's switches (:func:`resolve_switches`); None takes
        CHRONOS_LEGACY's value. ``covariates=False`` gives the model the
        target alone; ``conformal`` and ``shrink`` (daily only) recalibrate
        the band, and shrink the centre, on 20 past origins.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'volatility'].
    """
    interval = "1d" if interval in ("1d", "1D", "daily", "1 Day") else "1h"

    if isinstance(df, str):
        print(df)
        return pd.DataFrame()

    sw = resolve_switches(interval, prediction_length, covariates, conformal, shrink)
    if sw["conformal"] or sw["shrink"]:
        key = (sw["conformal"], sw["shrink"])
        return chronos_paths(df, interval, prediction_length, company_name, sw["covariates"],
                             variants=[key])[key]

    max_obs = _MAX_CONTEXT_DAILY if interval == "1d" else _MAX_CONTEXT_HOURLY
    if len(df) > max_obs:
        df = df.tail(max_obs).reset_index(drop=True)

    print(f"--- Starting Chronos-2 Prediction ({interval}) ---")
    print(company_name)

    df = prepare_timeseries_data(df, company_name, interval=interval)

    df = _add_covariates(df, interval) if sw["covariates"] else _target_only(df)

    input_df = df.drop(columns=["original_datetime"], errors="ignore")

    pipeline = _get_pipeline()

    try:
        pred_df = pipeline.predict_df(
            input_df,
            prediction_length=prediction_length,
            quantile_levels=_QUANTILE_LEVELS,
            id_column="id",
            timestamp_column="datetime",
            target="adj_close",
            validate_inputs=(interval == "1d"),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Chronos-2 prediction failed for '{company_name}' "
            f"(interval={interval}, horizon={prediction_length}): {exc}"
        ) from exc

    pred_df = _real_future_dates(pred_df, df, prediction_length, interval)

    pred_df = clean_prediction_output(pred_df)
    print("Chronos-2 Prediction Complete.")
    return pred_df
```
