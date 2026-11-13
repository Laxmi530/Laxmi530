- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Pure serving-gate logic — the layer the product's *trust* depends on.

A reviewer's point: "your product's trust depends on the gates, not only the
models." So the gate rules live here, free of Streamlit, and are unit-tested
directly (``test_gates.py``). The UI imports these; it does not re-implement them.

The rules encode the project's honest-serving contract:
* a "No edge" confidence bucket withholds the directional call,
* degraded / event-risk data forces every model to "No edge" (cone-only),
* an **off-manifold market regime** (autoencoder novelty) does the same, because
  a backtest-derived confidence bucket is an extrapolation once the present
  state is unlike anything the models were fitted on,
* conviction is the move measured in units of the model's own uncertainty, so a
  wide interval mechanically yields |conviction| < 1 ("inside the band").

The bucket rule itself lives in ``backtest.confidence_bucket`` (shared with the
offline calibration harness); this module is the *serving* layer on top of it.
"""

import numpy as np

from backtest import null_probability_floor

WITHHELD_BUCKET = "No edge"
WITHHELD_DIRECTION = "–"


def gated_direction(change, bucket):
    """
    Resolve (sentiment, direction) for a model row, honouring the gate.

    When the confidence ``bucket`` is "No edge" the directional call is withheld
    — the app never asserts Bullish/Bearish on a forecast its own walk-forward
    test rates no better than a coin flip.

    Parameters
    ----------
    change : float
        Forecast price change (sign gives Bullish/Bearish when not withheld).
    bucket : str
        Confidence bucket ('High'/'Medium'/'Low'/'No edge'/...).

    Returns
    -------
    tuple(str, str)
        (sentiment, direction marker).
    """
    if bucket == WITHHELD_BUCKET:
        return WITHHELD_BUCKET, WITHHELD_DIRECTION
    if change >= 0:
        return "Bullish", "▲"
    return "Bearish", "▼"


def conviction(change, lower, upper):
    """
    Forecast move ÷ half the model's own 95% interval width.

    Same sign as the move; |conviction| >= 1 means the move exceeds the model's
    uncertainty (the current price sits outside the band). A *wide* interval
    therefore drives conviction toward 0 — the band swamps the move — which is
    exactly the "don't over-read a noisy point" behaviour the gate wants.

    Returns
    -------
    float
        Conviction, or ``np.nan`` if no usable interval is available.
    """
    if lower is None or upper is None:
        return float("nan")
    half_band = (float(upper) - float(lower)) / 2.0
    if half_band <= 0:
        return float("nan")
    return float(change) / half_band


def _withhold(entry, note):
    """
    Force ``entry`` to "No edge" and record why, unless a gate already withheld it.

    The UI's confidence entries always carry a ``note`` key (``''`` when the
    assessment had nothing to say), so ``setdefault`` would never write the
    reason, and ``confidence_reason`` would then report "no edge vs random walk"
    for a row that was really withheld for its data or regime. The first gate to
    withhold keeps its reason; an empty or assessment note is replaced.
    """
    entry["bucket"] = WITHHELD_BUCKET
    if not str(entry.get("note") or "").startswith("withheld"):
        entry["note"] = note


def apply_data_quality_gate(conf_map, model_names, degraded):
    """
    On degraded/event-risk data, force every model's bucket to "No edge".

    This reuses the existing abstention machinery (``gated_direction`` keys off
    "No edge") so directional calls vanish consistently when the *input* — not
    the model — is the problem. Returns a NEW dict; inputs are not mutated.

    Parameters
    ----------
    conf_map : dict
        Mapping model name -> confidence dict (may be empty if assessment off).
    model_names : iterable of str
        Models in the current run.
    degraded : bool
        True when the data-quality gate is in 'risk_cone_only' or 'abstain'.

    Returns
    -------
    dict
        Updated confidence map.
    """
    out = {k: dict(v) for k, v in (conf_map or {}).items()}
    if not degraded:
        return out
    for name in model_names:
        entry = dict(out.get(name, {}))
        _withhold(entry, "withheld: data quality")
        out[name] = entry
    return out


def apply_novelty_gate(conf_map, model_names, novelty):
    """
    On an **off-manifold market regime**, force every model's bucket to "No edge".

    Closes a blind spot in the existing gates. ``apply_data_quality_gate`` catches
    a broken *input*; the confidence bucket catches a model with no measured edge
    *in the past*. Neither can see that the present regime is unlike anything the
    models were fitted on — and in that situation a backtest-derived confidence
    bucket is an extrapolation, not evidence. The autoencoder novelty detector
    (``autoencoder_regime.regime_novelty``) supplies the missing signal.

    Deliberately conservative: only an ``'unprecedented'`` verdict (reconstruction
    error above the 99.5th percentile of the held-out reference distribution)
    withholds the call. An ``'unusual'`` verdict is *annotated* but left tradeable,
    because equity data is fat-tailed and a 95th-percentile window is a normal
    monthly occurrence — gating on it would abstain far too often to be useful.

    Reuses the same "No edge" mechanism as the other gates so the directional
    call disappears consistently, whatever the reason. Returns a NEW dict; inputs
    are not mutated.

    Parameters
    ----------
    conf_map : dict
        Mapping model name -> confidence dict.
    model_names : iterable of str
        Models in the current run.
    novelty : dict
        Report from ``autoencoder_regime.regime_novelty``. An unavailable or
        ``'unknown'`` report is a no-op, so the gate never fires on a failed
        novelty check.

    Returns
    -------
    dict
        Updated confidence map.
    """
    out = {k: dict(v) for k, v in (conf_map or {}).items()}
    if not novelty or not novelty.get("available"):
        return out

    verdict = novelty.get("verdict")
    if verdict not in ("unusual", "unprecedented"):
        return out

    pct = novelty.get("percentile", float("nan"))
    pct_s = f"{pct * 100:.1f}th pct" if pct == pct else "n/a"     # NaN check
    for name in model_names:
        entry = dict(out.get(name, {}))
        entry["novelty"] = verdict
        if verdict == "unprecedented":
            _withhold(entry, f"withheld: unseen market regime ({pct_s})")
        elif not entry.get("note"):
            entry["note"] = f"caution: unusual market regime ({pct_s})"
        out[name] = entry
    return out


#: The longest daily horizon (sessions) with a directional call. Every skill
#: measurement in this project is at about five sessions; beyond two weeks the
#: median path is a guess and the 95% range is the forecast (README section 83).
MAX_DIRECTIONAL_SESSIONS = 13

#: Models whose 95% band held under 90% of outcomes at 20-90 sessions on NSE
#: data before the reserved period (run_long_horizon_coverage.py, the rule
#: written down before the run). They are left out of daily runs longer than
#: MAX_DIRECTIONAL_SESSIONS, where the range is the forecast (README section 83).
#: Regime Switching was on it until its range was fixed to use its own fit; it
#: then held 98.4% at 20-90 sessions (run_regime_fix.py, README section 88).
#: LightGBM came off when larger leaves and a shrunk centre held 94.4% on a
#: fresh draw (run_lgbm_band.py, README section 97). Prophet came off when a
#: flat trend with its change shrunk toward no change held 96.9% on a fresh
#: draw (run_prophet_improve.py, README section 116).
LONG_HORIZON_EXCLUDED = ("Hybrid ARIMA+LSTM",)


def long_horizon_models(model_names, horizon, interval="1d",
                        max_sessions=MAX_DIRECTIONAL_SESSIONS, excluded=LONG_HORIZON_EXCLUDED):
    """``(kept, left_out)``: the models to run at this horizon, and the ones dropped.

    Only daily horizons beyond ``max_sessions`` drop anything; order is kept.
    """
    names = list(model_names)
    if str(interval).lower() not in ("1d", "daily", "1 day") or int(horizon) <= max_sessions:
        return names, []
    return ([m for m in names if m not in excluded], [m for m in names if m in excluded])


def apply_horizon_gate(conf_map, model_names, horizon, interval="1d",
                       max_sessions=MAX_DIRECTIONAL_SESSIONS):
    """
    Beyond ``max_sessions`` daily sessions, force every model's bucket to "No edge".

    Same "No edge" mechanism as the other gates, so the directional call goes
    away whatever the reason, and the range is what the page presents. Hourly
    horizons are capped at five sessions by the selector and are left alone.
    Returns a NEW dict; inputs are not mutated.
    """
    out = {k: dict(v) for k, v in (conf_map or {}).items()}
    if str(interval).lower() not in ("1d", "daily", "1 day") or int(horizon) <= max_sessions:
        return out
    for name in model_names:
        entry = dict(out.get(name, {}))
        _withhold(entry, f"withheld: long horizon ({int(horizon)} sessions; the range is the forecast)")
        out[name] = entry
    return out


def confidence_reason(entry):
    """
    Short, auditable reason code for a model's confidence verdict.

    Ties the bucket to the DM p-value / coverage / data-quality so every row says
    *why* it got its label instead of presenting a bare bucket (or an implied
    winner). Reads a conf_map entry produced by the UI's confidence assessment.
    """
    if not entry:
        return "not assessed"
    note = entry.get("note", "")
    if "data quality" in note:
        return "withheld — data quality"
    if "unseen market regime" in note:
        return "withheld — unseen market regime"
    if "long horizon" in note:
        return "withheld — long horizon: the range is the forecast"
    bucket = entry.get("bucket", "—")
    dm_p = entry.get("dm_p", float("nan"))
    cov = entry.get("coverage", float("nan"))
    imp = entry.get("improvement", float("nan"))
    p = f"DM p={dm_p:.2f}" if dm_p == dm_p else "DM n/a"        # NaN check
    cov_s = f", cov {cov*100:.0f}%" if cov == cov else ""
    imp_s = f"{imp*100:+.0f}% vs RW" if imp == imp else ""
    # An 'unusual' (but not unprecedented) regime does not withhold the call; it
    # is surfaced as a caveat on whatever verdict the skill test produced.
    suffix = " · unusual regime" if entry.get("novelty") == "unusual" else ""
    # A p-value is not the probability of no edge. Where the label claims an
    # edge, print the least that probability can be (backtest.null_probability_floor),
    # so "DM p=0.08" cannot be read as "92% sure".
    floor = null_probability_floor(dm_p) if dm_p == dm_p else float("nan")
    floor_s = f"; P(no edge) ≥ {floor*100:.0f}%" if floor == floor else ""

    if bucket == "High":
        return f"beats random walk ({p}{cov_s}{floor_s}){suffix}"
    if bucket == "Medium":
        return f"weak edge ({p}{floor_s}){suffix}"
    if bucket == "Low":
        return f"within noise ({imp_s}, {p}){suffix}"
    if bucket == WITHHELD_BUCKET:
        return f"no edge vs random walk ({p})"
    if bucket == "N/A":
        return note or "assessment failed"
    return "—"
```
