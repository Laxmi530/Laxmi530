- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Forward prediction log, paper portfolio, and calibration-drift monitoring.

Motivated by *Hands-On Deep Learning for Finance* (Troiano et al.), Chapter 3,
"Going live" (pp. 135-142): the deployment arc from prototype through a
**paper portfolio** — the strategy run on fictitious capital under live
conditions — to a soft launch, with continuous **benchmarking of live results
against model diagnostics**.

The gap this fills
------------------
Every evaluation in this project so far is **backward**: pick historical origins,
re-fit, score against what already happened. That is the right way to build a
model and a completely inadequate way to *operate* one, because it can only ever
tell you how the model would have done on data that already existed when you
tested it.

Nothing here recorded what the app **actually served**, so three questions were
unanswerable:

1. *Is the cone still calibrated?* A conformal interval is calibrated at fit
   time and decays silently as the regime moves. Coverage measured on 2021-2026
   backtest origins says nothing about coverage on the forecasts served last
   month.
2. *Did the confidence gate work?* The gate labels each forecast High / Medium /
   Low / No edge. Whether those labels actually sorted realised skill can only be
   checked against outcomes that arrived *after* the label was assigned.
3. *Would trading the served signals have made money?* ``strategy_metrics``
   simulates a portfolio backwards over chosen origins. A paper portfolio runs
   the signals the app really emitted, in the order it really emitted them, with
   no origin selection at all.

This module answers all three by writing an **append-only** log at serve time and
joining realised outcomes later.

Why append-only matters
-----------------------
The entire value of a forward log is that it **cannot be retro-fitted**. A
prediction is written before its outcome exists, and is never rewritten — only
annotated with the realised price once that price is knowable. That single
property is what makes the resulting coverage and P&L numbers immune to the
data-snooping the rest of this project spends so much effort correcting for:
there is no origin selection, no re-fitting, and no choice of which forecasts to
count. Every served forecast is in the sample, including the embarrassing ones.

Design
------
* **Storage** is JSONL under ``.prediction_log/`` — one line per (run, model,
  horizon step), human-readable, append-only, trivially greppable, and safe to
  delete. Same "best-effort local artifact" posture as ``safe_fetch``'s cache.
* **Never breaks a serving run.** Every logging call is wrapped; a failed write
  prints and returns rather than propagating. A forecast that cannot be logged is
  still a forecast.
* **Injectable clock and fetcher** so the tests run offline and deterministically.
* **Idempotent resolution.** ``resolve_outcomes`` can be run any number of times;
  it only fills rows whose target has matured and which are still unresolved.
"""

import json
import os
import time
import uuid

import numpy as np
import pandas as pd

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       ".prediction_log")
LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")

# Coverage band considered healthy for a nominal 95% interval. Wide on purpose:
# with a few dozen resolved points the binomial noise on a 0.95 rate is large,
# and an alert that fires on noise trains people to ignore alerts.
_COVERAGE_TARGET = 0.95
_MIN_RESOLVED_FOR_DRIFT = 30
# Minimum spread in realised directional accuracy between the worst and best
# confidence bucket before the gate is called informative. Below this the labels
# are not discriminating, however tidily they happen to be ordered.
_GATE_MIN_SPREAD = 0.02

_FIELDS = [
    "run_id", "logged_at_utc", "ticker", "interval", "horizon", "model",
    "origin_datetime", "anchor_price", "step", "target_datetime",
    "predicted_price", "lower_bound", "upper_bound",
    "confidence_bucket", "dq_mode", "novelty_verdict", "seed",
    "resolved_at_utc", "realised_price", "source",
]


# ==========================================
# Writing
# ==========================================
def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_forecast(ticker, interval, horizon, model, prediction_df,
                 anchor_price, origin_datetime, confidence_bucket=None,
                 dq_mode=None, novelty_verdict=None, seed=None,
                 run_id=None, now=None, path=None, source="app"):
    """
    Append one served forecast to the log — one row per horizon step.

    Called at serve time, **before** any outcome exists. Records not just the
    numbers but the *context in which they were served*: the confidence bucket,
    the data-quality mode, and the novelty verdict. Without that context the log
    could tell you whether forecasts were accurate but not whether the **gates**
    were doing their job, which is the more valuable question for a product whose
    central feature is abstention.

    Parameters
    ----------
    ticker : str
        Company/ticker identifier as the app knows it.
    interval : str
        '1d' / '1h' (as passed to the models).
    horizon : int
        Number of steps forecast.
    model : str
        Display name from the model registry.
    prediction_df : pd.DataFrame
        The model's output: 'datetime', 'predicted_price', and optionally
        'lower_bound' / 'upper_bound'.
    anchor_price : float
        Last observed price at the forecast origin — needed later to score
        direction, and stored rather than recomputed so a data revision cannot
        silently change what the forecast was measured against.
    origin_datetime : datetime-like
        Timestamp of the last observed bar.
    confidence_bucket, dq_mode, novelty_verdict : str, optional
        The gate verdicts in force when this forecast was served.
    seed : int, optional
        RNG seed (from ``reproducibility``), so a row can be reproduced.
    run_id : str, optional
        Groups all models from one Submit. Generated if omitted.
    source : str, optional
        "app" for a forecast a user asked for, "watchlist" for the scheduled
        run on the fixed watchlist (README section 78). Rows written before
        the field existed read back as "app".
    now : str/Timestamp, optional
        Injectable "now" for tests.
    path : str, optional
        Log file path (injectable for tests).

    Returns
    -------
    str or None
        The ``run_id`` written, or None if logging failed (never raises).
    """
    path = path or LOG_FILE
    try:
        if prediction_df is None or len(prediction_df) == 0:
            return None
        run_id = run_id or uuid.uuid4().hex[:12]
        ts = (pd.Timestamp.now("UTC") if now is None
              else pd.Timestamp(now)).isoformat()

        has_band = ("lower_bound" in prediction_df.columns
                    and "upper_bound" in prediction_df.columns)

        rows = []
        for i, (_, r) in enumerate(prediction_df.iterrows(), start=1):
            rows.append({
                "run_id": run_id,
                "logged_at_utc": ts,
                "ticker": str(ticker),
                "interval": str(interval),
                "horizon": int(horizon),
                "model": str(model),
                "origin_datetime": str(pd.Timestamp(origin_datetime)),
                "anchor_price": float(anchor_price),
                "step": int(i),
                "target_datetime": str(pd.Timestamp(r["datetime"])),
                "predicted_price": float(r["predicted_price"]),
                "lower_bound": (float(r["lower_bound"]) if has_band else None),
                "upper_bound": (float(r["upper_bound"]) if has_band else None),
                "confidence_bucket": confidence_bucket,
                "dq_mode": dq_mode,
                "novelty_verdict": novelty_verdict,
                "seed": (int(seed) if seed is not None else None),
                "resolved_at_utc": None,
                "realised_price": None,
                "source": str(source),
            })

        _ensure_dir() if path == LOG_FILE else os.makedirs(
            os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return run_id
    except Exception as exc:            # logging must never break a forecast
        print(f"prediction_log: write skipped ({exc})")
        return None


# ==========================================
# Reading
# ==========================================
def read_log(path=None):
    """
    Load the whole log as a DataFrame (empty frame with the right columns if none).

    Malformed lines are skipped rather than fatal: an append-only file written by
    a long-running app is exactly the kind of artifact that can end up with one
    truncated line after a crash, and losing the other 10,000 rows to it would be
    a poor trade.
    """
    path = path or LOG_FILE
    if not os.path.exists(path):
        return pd.DataFrame(columns=_FIELDS)
    rows, bad = [], 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    bad += 1
    except Exception as exc:
        print(f"prediction_log: read failed ({exc})")
        return pd.DataFrame(columns=_FIELDS)
    if bad:
        print(f"prediction_log: skipped {bad} malformed line(s)")
    df = pd.DataFrame(rows)
    for c in _FIELDS:
        if c not in df.columns:
            df[c] = None
    df["source"] = df["source"].fillna("app")
    return df[_FIELDS]


def _rewrite(df, path, attempts=5, backoff=0.05):
    """
    Atomically rewrite the log (used only to annotate resolved outcomes).

    ``os.replace`` is atomic on POSIX, but on Windows it raises
    ``PermissionError: [WinError 5]`` whenever another process holds even a
    momentary handle on the destination - an antivirus scanner or the search
    indexer touching a file that was just written is enough. It is a race, so it
    fails perhaps one run in ten and succeeds on retry, which is exactly the
    profile that makes it hard to attribute: the resolve step crashes, leaves a
    stray ``.tmp`` beside the log, and appears to be a different problem each
    time. It surfaced here as intermittent test failures in three DIFFERENT
    drift-report tests before being traced to this line.

    Retrying with exponential backoff is the standard Windows-safe pattern. The
    temporary file is removed if every attempt fails, so a failed resolve cannot
    leave litter that a later run mistakes for real data.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps({k: (None if pd.isna(r[k]) else r[k])
                                 for k in _FIELDS}) + "\n")
    last = None
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:          # Windows handle contention
            last = exc
            time.sleep(backoff * (2 ** attempt))
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise last


# ==========================================
# Resolution
# ==========================================
def _wall_clock(values):
    """
    Datetimes as tz-naive wall-clock time at nanosecond resolution.

    ``merge_asof`` needs both keys at the same resolution, and pandas 3 keeps
    whatever the source had: the log's ISO strings parse to microseconds, a
    fetched history may be seconds, and the merge refused to run, so no row was
    ever resolved (README section 77). A timezone is dropped, not converted: a
    daily NSE bar stamped 00:00+05:30 is that session, and converting it to UTC
    would move it to the previous day.
    """
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.astype("datetime64[ns]")


def resolve_outcomes(fetch_fn, now=None, path=None, verbose=True):
    """
    Fill in realised prices for every matured, unresolved prediction.

    Idempotent and additive: a row is touched exactly once, when its
    ``target_datetime`` has passed and a realised price is obtainable. The
    prediction fields themselves are never modified — only ``realised_price`` and
    ``resolved_at_utc`` are written — which is what preserves the log's
    can't-be-retro-fitted property.

    Parameters
    ----------
    fetch_fn : callable
        ``fetch_fn(ticker, interval) -> DataFrame`` with 'datetime' and
        'adj_close'. In the app this is ``get_whole_stock_data`` (optionally
        through ``safe_fetch``); tests inject a local frame.
    now : Timestamp, optional
        Injectable "now"; defaults to the current time.
    path : str, optional
        Log file path.
    verbose : bool, optional
        Print a one-line summary.

    Returns
    -------
    dict
        {'checked', 'resolved', 'still_pending', 'tickers'}.
    """
    path = path or LOG_FILE
    df = read_log(path)
    out = {"checked": len(df), "resolved": 0, "still_pending": 0, "tickers": []}
    if df.empty:
        return out

    now_ts = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    target = pd.to_datetime(df["target_datetime"], errors="coerce")
    pending = df["realised_price"].isna() & target.notna() & (target <= now_ts)
    out["still_pending"] = int((df["realised_price"].isna() & (target > now_ts)).sum())
    if not pending.any():
        if verbose:
            print(f"prediction_log: nothing matured to resolve "
                  f"({out['still_pending']} still pending)")
        return out

    resolved_at = pd.Timestamp.now("UTC").isoformat()
    for (ticker, interval), grp in df[pending].groupby(["ticker", "interval"]):
        try:
            hist = fetch_fn(ticker, interval)
        except Exception as exc:
            print(f"prediction_log: fetch failed for {ticker} ({exc})")
            continue
        if not isinstance(hist, pd.DataFrame) or hist.empty:
            continue
        h = hist.copy()
        h["datetime"] = _wall_clock(h["datetime"]).to_numpy()
        h = h.dropna(subset=["datetime", "adj_close"]).sort_values("datetime")
        if h.empty:
            continue
        out["tickers"].append(ticker)

        # A target is resolvable only once the history runs strictly PAST it.
        # Without this a backward match fills a target whose bar has not printed
        # yet with the latest bar before it: a feed one day behind, or a daily
        # target looked up during its own session (the target is stamped 00:00,
        # so it is "<= now" from midnight), was resolved at a stale or partial
        # price, and resolution never revisits a row. Daily targets need a bar
        # on a later DATE, so the target session has closed; hourly targets a
        # later bar.
        tgt = _wall_clock(grp["target_datetime"]).set_axis(grp.index)
        last_bar = h["datetime"].iloc[-1]
        if str(interval).lower() in ("1d", "daily", "1 day"):
            ready = tgt.dt.normalize() < last_bar.normalize()
        else:
            ready = tgt < last_bar
        out["still_pending"] += int((~ready).sum())
        grp = grp[ready.values]
        if grp.empty:
            continue

        # Backward as-of match: the realised price at (or last before) the target
        # bar. Backward rather than nearest, so a target that has not actually
        # printed yet cannot borrow a later bar's price.
        left = pd.DataFrame({
            "datetime": _wall_clock(grp["target_datetime"]).to_numpy(),
            "_idx": grp.index,
        }).sort_values("datetime")
        merged = pd.merge_asof(
            left, h[["datetime", "adj_close"]], on="datetime", direction="backward")

        # Put the outcome on the ANCHOR's adjustment basis. The anchor is the
        # price at serve time, when the latest bar's adjustment factor was 1;
        # this history was fetched later, and a provider back-adjusts every bar
        # before each new ex-dividend date. A dividend going ex between the
        # target and this fetch scaled the target's adj_close down by about the
        # dividend yield, and a dividend between origin and target left the
        # raw change short of the total return the forecast was made in
        # (Mladjenovic, Ch. 6: the stock "trades at a slightly lower price to
        # reflect the amount of the dividend"). Measuring both ends from this
        # one fetch -- anchor x adj(target) / adj(origin) -- is the total
        # return from origin to target, whatever was adjusted afterwards.
        org = pd.merge_asof(
            pd.DataFrame({"datetime": _wall_clock(grp["origin_datetime"]).to_numpy(),
                          "_idx": grp.index}).sort_values("datetime"),
            h[["datetime", "adj_close"]], on="datetime", direction="backward",
        ).set_index("_idx")["adj_close"]
        for _, m in merged.iterrows():
            if pd.notna(m["adj_close"]):
                i = m["_idx"]
                a0 = org.get(i, np.nan)
                anchor = pd.to_numeric(df.at[i, "anchor_price"], errors="coerce")
                if pd.notna(a0) and a0 > 0 and pd.notna(anchor) and anchor > 0:
                    realised = float(anchor) * float(m["adj_close"]) / float(a0)
                else:
                    realised = float(m["adj_close"])
                df.at[i, "realised_price"] = realised
                df.at[i, "resolved_at_utc"] = resolved_at
                out["resolved"] += 1

    if out["resolved"]:
        # The fetches above take seconds, and another process (the app's Submit,
        # a watchlist run) may have appended forecasts meanwhile. The log is
        # append-only, so those are the rows past the ones read at the start:
        # keep them, or the rewrite would silently drop them. README section 82.
        fresh = read_log(path)
        if len(fresh) > len(df):
            df = pd.concat([df, fresh.iloc[len(df):]], ignore_index=True)
        _rewrite(df, path)
    if verbose:
        print(f"prediction_log: resolved {out['resolved']} prediction(s); "
              f"{out['still_pending']} still pending")
    return out


# ==========================================
# Scoring
# ==========================================
def resolved_frame(path=None):
    """Resolved rows only, with realised/predicted returns and a covered flag."""
    df = read_log(path)
    if df.empty:
        return df
    df = df[df["realised_price"].notna()].copy()
    if df.empty:
        return df
    for c in ("anchor_price", "predicted_price", "realised_price",
              "lower_bound", "upper_bound"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df["anchor_price"] > 0
    df = df[ok & (df["realised_price"] > 0) & (df["predicted_price"] > 0)]
    if df.empty:
        return df
    df["pred_return"] = np.log(df["predicted_price"] / df["anchor_price"])
    df["real_return"] = np.log(df["realised_price"] / df["anchor_price"])
    df["abs_error"] = (df["realised_price"] - df["predicted_price"]).abs()
    band = df["lower_bound"].notna() & df["upper_bound"].notna()
    df["covered"] = np.where(
        band,
        (df["realised_price"] >= df["lower_bound"])
        & (df["realised_price"] <= df["upper_bound"]),
        np.nan)
    df["direction_hit"] = (np.sign(df["pred_return"])
                           == np.sign(df["real_return"]))
    return df


def _binomial_ci(k, n, z=1.96):
    """Wald interval on a proportion, floored/capped to [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    half = z * np.sqrt(max(p * (1 - p), 1e-12) / n)
    return (max(0.0, p - half), min(1.0, p + half))


def drift_report(path=None, recent=60, target=_COVERAGE_TARGET):
    """
    Per-model **live** calibration and skill, with an explicit drift verdict.

    The headline is live interval coverage against the nominal 95%, because that
    is the claim the product actually makes and the one that decays first when a
    regime moves. It is reported with a binomial confidence interval so a small
    sample cannot masquerade as a finding — the single most common way live
    monitoring produces false alarms and gets switched off.

    ``recent`` isolates the most recent N resolved predictions per model, so a
    long-run average cannot hide a recent breakdown: a model that covered
    perfectly for a year and has missed five of its last ten is exactly the case
    worth catching, and the all-time number would bury it.

    Parameters
    ----------
    path : str, optional
        Log file path.
    recent : int, optional
        Size of the recent window per model. Default 60.
    target : float, optional
        Nominal interval coverage. Default 0.95.

    Returns
    -------
    dict
        {'per_model': DataFrame, 'alerts': list of str, 'n_resolved': int}.
    """
    df = resolved_frame(path)
    if df.empty:
        return {"per_model": pd.DataFrame(), "alerts": [], "n_resolved": 0}

    df = df.sort_values("target_datetime")
    rows, alerts = [], []
    for model, g in df.groupby("model"):
        g_all = g
        g_recent = g.tail(recent)
        cov_all = g_all["covered"].dropna()
        cov_rec = g_recent["covered"].dropna()

        rec = {
            "model": model,
            "n_all": int(len(g_all)),
            "n_recent": int(len(g_recent)),
            "coverage_all": (round(float(cov_all.mean()), 3)
                             if len(cov_all) else float("nan")),
            "coverage_recent": (round(float(cov_rec.mean()), 3)
                                if len(cov_rec) else float("nan")),
            "dir_hit_recent": round(float(g_recent["direction_hit"].mean()), 3),
            "mae_recent": round(float(g_recent["abs_error"].mean()), 3),
        }
        if len(cov_rec) >= _MIN_RESOLVED_FOR_DRIFT:
            lo, hi = _binomial_ci(int(cov_rec.sum()), int(len(cov_rec)))
            rec["coverage_ci"] = f"[{lo:.2f}, {hi:.2f}]"
            # Only alert when the CI EXCLUDES the target: below that bar the
            # sample simply cannot distinguish drift from noise.
            if hi < target:
                alerts.append(
                    f"{model}: live coverage {rec['coverage_recent']:.2f} over the "
                    f"last {len(cov_rec)} resolved predictions, 95% CI "
                    f"{rec['coverage_ci']} excludes the {target:.2f} target — the "
                    "cone is UNDER-covering; recalibrate before trusting it.")
            elif lo > target:
                alerts.append(
                    f"{model}: live coverage {rec['coverage_recent']:.2f} "
                    f"(CI {rec['coverage_ci']}) is above the {target:.2f} target — "
                    "intervals are wider than they need to be (safe but costly).")
        else:
            rec["coverage_ci"] = f"n<{_MIN_RESOLVED_FOR_DRIFT}"
        rows.append(rec)

    return {"per_model": pd.DataFrame(rows).sort_values("model")
            .reset_index(drop=True),
            "alerts": alerts,
            "n_resolved": int(len(df))}


def gate_validation(path=None):
    """
    Did the confidence gate actually sort realised skill? (live, not backtested)

    ``calibration.validate_gate`` answers this offline with a nested backtest.
    This answers it from **served** forecasts, which is a stronger claim: the
    bucket was assigned before the outcome existed, so there is no way for the
    labelling to have been informed by what happened next.

    The gate is informative if realised accuracy improves monotonically from
    "No edge" through "High". If it does not, the gate is decoration.

    Returns
    -------
    dict
        {'by_bucket': DataFrame, 'monotonic': bool or None, 'verdict': str}.
    """
    df = resolved_frame(path)
    if df.empty or df["confidence_bucket"].isna().all():
        return {"by_bucket": pd.DataFrame(), "monotonic": None,
                "verdict": "no resolved predictions carry a confidence bucket yet"}

    order = ["No edge", "Low", "Medium", "High"]
    g = (df.dropna(subset=["confidence_bucket"])
           .groupby("confidence_bucket")
           .agg(n=("direction_hit", "size"),
                dir_hit=("direction_hit", "mean"),
                coverage=("covered", "mean"),
                mae=("abs_error", "mean"))
           .reset_index())
    g["_o"] = g["confidence_bucket"].apply(
        lambda b: order.index(b) if b in order else -1)
    g = g.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    for c in ("dir_hit", "coverage", "mae"):
        g[c] = g[c].round(3)

    ranked = [b for b in order if b in set(g["confidence_bucket"])]
    monotonic, spread = None, float("nan")
    if len(ranked) >= 2:
        hits = [float(g.loc[g["confidence_bucket"] == b, "dir_hit"].iloc[0])
                for b in ranked]
        spread = hits[-1] - hits[0]
        # A **strict** spread is required, not merely non-decreasing. Ties satisfy
        # `<=` trivially, so an all-equal set of hit rates would otherwise be
        # reported as "the gate is informative" — a gate that discriminates
        # nothing being declared a success. Demand a real gap between the worst
        # and best bucket before claiming the ordering means anything.
        monotonic = (all(hits[i] <= hits[i + 1] + 1e-9
                         for i in range(len(hits) - 1))
                     and spread > _GATE_MIN_SPREAD)

    if monotonic is None:
        verdict = "too few distinct buckets resolved to judge the gate"
    elif monotonic:
        verdict = (f"the gate IS informative on live data — realised directional "
                   f"accuracy rises with the assigned bucket "
                   f"(+{spread:.3f} from '{ranked[0]}' to '{ranked[-1]}')")
    elif np.isfinite(spread) and abs(spread) <= _GATE_MIN_SPREAD:
        verdict = (f"the gate is FLAT on live data — realised accuracy barely "
                   f"differs across buckets ({spread:+.3f} from '{ranked[0]}' to "
                   f"'{ranked[-1]}'), so the labels are not discriminating yet")
    else:
        verdict = ("the gate did NOT sort realised skill on live data; treat the "
                   "buckets as descriptive, not predictive, until it does")
    return {"by_bucket": g, "monotonic": monotonic, "spread": spread,
            "verdict": verdict}


def paper_portfolio(path=None, model=None, cost_bps=None, mode="sign",
                    interval="daily"):
    """
    Run the **served** signals as fictitious capital (the book's paper portfolio).

    The crucial difference from ``strategy_metrics`` used over backtest origins:
    there is **no origin selection and no re-fitting**. Every forecast the app
    emitted at the final horizon step is a trade, in the order it was emitted.
    That removes the last degree of freedom a backtest retains, which is why the
    book treats the paper portfolio as the step that decides whether test-phase
    assumptions survive contact with live conditions.

    Only the final horizon step is traded, so successive trades do not overlap.

    Parameters
    ----------
    path : str, optional
        Log file path.
    model : str, optional
        Restrict to one model (otherwise every model is reported separately).
    cost_bps : float, optional
        One-way cost; defaults to ``strategy_metrics.DEFAULT_COST_BPS``.
    mode : str, optional
        Position rule. Default 'sign'.
    interval : str, optional
        'daily' or 'hourly', for annualisation.

    Returns
    -------
    dict[str, dict]
        Model name -> ``strategy_metrics.evaluate_strategy`` result (plus 'n').
    """
    import strategy_metrics as sm

    df = resolved_frame(path)
    if df.empty:
        return {}
    df = df[df["step"] == df["horizon"]]          # non-overlapping trades only
    if model:
        df = df[df["model"] == model]
    if df.empty:
        return {}

    cost = sm.DEFAULT_COST_BPS if cost_bps is None else float(cost_bps)
    out = {}
    for name, g in df.groupby("model"):
        g = g.sort_values("target_datetime")
        if len(g) < 3:
            continue
        res = sm.evaluate_strategy(
            g["pred_return"].values, g["real_return"].values,
            interval=interval, cost_bps=cost, mode=mode,
            dates=pd.to_datetime(g["target_datetime"]).values,
            periods_per_bar=int(g["horizon"].iloc[0]))
        res["n"] = int(len(g))
        out[name] = res
    return out


def summary(path=None, recent=60):
    """Render the whole live report as text (drift + gate + paper portfolio)."""
    lines = []
    d = drift_report(path, recent=recent)
    lines.append(f"=== Live prediction log: {d['n_resolved']} resolved "
                 f"predictions ===")
    if d["per_model"].empty:
        lines.append("  (nothing resolved yet — log forecasts, then run "
                     "resolve_outcomes once their horizons mature)")
        return "\n".join(lines)

    lines.append("\n-- Calibration drift (live coverage vs the 95% claim) --")
    lines.append(d["per_model"].to_string(index=False))
    if d["alerts"]:
        lines.append("\n  ALERTS:")
        lines.extend(f"   * {a}" for a in d["alerts"])
    else:
        lines.append("\n  No drift alerts (or too few resolved points to judge).")

    g = gate_validation(path)
    lines.append("\n-- Did the confidence gate sort realised skill? --")
    if g["by_bucket"].empty:
        lines.append(f"  {g['verdict']}")
    else:
        lines.append(g["by_bucket"].to_string(index=False))
        lines.append(f"  Verdict: {g['verdict']}")

    pp = paper_portfolio(path)
    lines.append("\n-- Paper portfolio (served signals, fictitious capital) --")
    if not pp:
        lines.append("  (too few resolved final-step forecasts to trade)")
    else:
        import strategy_metrics as sm
        for name, res in sorted(pp.items()):
            lines.append("")
            lines.append(sm.format_strategy_report(res, name))
    return "\n".join(lines)
```
