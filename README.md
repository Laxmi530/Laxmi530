- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Kalman filter (damped local linear trend) — the one model class this project lacked.

Why this model is different from the other nineteen
---------------------------------------------------
Every other model here fits **fixed parameters** over a training window and then
freezes them to forecast. A Kalman filter carries a *state* that is re-estimated
at every bar, so the level and slope it extrapolates from are the ones implied
by the most recent data, continuously reweighted against their own uncertainty.
That is a genuinely different answer to non-stationarity than refitting on a
rolling window.

Klaas, *Machine Learning for Finance*, Ch. 4:

    "Most classic probabilistic modeling techniques, such as Kalman filters, can
     give confidence intervals for predictions, whereas regular deep learning
     cannot do this."

That second property is the interesting one here, because this project's
deliverable is calibrated uncertainty rather than a point forecast. Every
existing interval is **conformal** — distribution-free, calibrated on held-out
residuals, and agnostic about the model. A Kalman interval is the opposite: it
falls straight out of the state covariance, and it is only as good as the model
assumptions behind it. Whether that native interval is as trustworthy as the
conformal one is a measurable question, and ``run_kalman_eval.py`` measures it
rather than assuming either way.

The model
---------
Damped local linear trend, on **log prices**::

    level_t  = level_{t-1} + phi * slope_{t-1} + w_level
    slope_t  = phi * slope_{t-1}               + w_slope
    y_t      = level_t                         + v

``phi`` in [0, 1] damps the trend, so the h-step forecast adds
``slope * (phi + phi^2 + ... + phi^h)`` — a geometric sum that converges instead
of extrapolating a straight line to infinity. Damped trend is the choice that
wins forecasting competitions, and it matters more than usual here: an
undamped slope fitted to a recent run-up would extrapolate it indefinitely,
which is exactly the failure mode this project's whole evaluation layer exists
to catch.

**The model is allowed to conclude it is a random walk.** If the data prefers
``q_slope -> 0`` or ``phi -> 0``, the fit degenerates to a local level model,
whose forecast is flat — indistinguishable from the naive baseline. That is not
a failure of the implementation; it is the honest answer for a near-efficient
price series, and the fitted parameters are reported so it is visible when it
happens.

Estimation
----------
Parameters are fitted by maximum likelihood on the training window, via a coarse
grid over ``(q_level/r, q_slope/r, phi)`` followed by a local refinement. Grid
search rather than an optimiser because the likelihood surface for state-space
models is famously multi-modal and flat in places, and a 3-parameter grid is
cheap. ``r`` is fixed at 1 and the noise **scale** is profiled out analytically (the
concentrated likelihood), because only the *ratios* drive the filter's
behaviour. Scoring the raw likelihood at a fixed ``r`` instead is a trap that
this implementation fell into first: on log-price data of scale ~0.02 it rewards
whichever ratio pushes the innovation variance towards 1, so it drives
``q -> 0`` no matter what the truth is. No SciPy, consistent with the rest of this
project.
"""

import numpy as np
import pandas as pd

_EPS = 1e-12
_Z95 = 1.959963984540054


# ==========================================
# 1. The filter
# ==========================================
def kalman_filter(y, q_level, q_slope, r, phi=1.0, return_states=False):
    """
    Run the damped local-linear-trend filter over ``y`` (log prices).

    Returns
    -------
    dict
        ``loglik`` (diffuse: the first two observations initialise the state and
        are excluded from the likelihood, so different parameterisations are
        compared on the same terms), ``level``, ``slope``, ``P`` (final 2x2 state
        covariance), ``innovations``, and the filtered paths when requested.

    Implementation note: the 2x2 algebra is written out rather than using
    ``numpy.linalg``. At this size the explicit form is faster, and — more
    usefully — it makes the update visibly a scalar Kalman gain, which is what
    the reader needs to check.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return {"loglik": -np.inf, "level": np.nan, "slope": 0.0,
                "P": np.eye(2) * 1e6, "innovations": np.array([]),
                "levels": np.array([]), "slopes": np.array([])}

    # Diffuse-ish initialisation from the first two points: level at y[0], slope
    # from the first difference, with a wide prior so the data dominates fast.
    level, slope = float(y[0]), float(y[1] - y[0])
    p11, p12, p22 = 1e4, 0.0, 1e4

    sum_log_s = 0.0
    sum_e2_s = 0.0
    n_eff = 0
    innov = np.empty(n)
    innov[:] = np.nan
    levels = np.empty(n)
    slopes = np.empty(n)

    for t in range(n):
        # --- predict -------------------------------------------------------
        level_p = level + phi * slope
        slope_p = phi * slope
        # F P F' + Q, with F = [[1, phi], [0, phi]]
        p11_p = p11 + 2.0 * phi * p12 + phi * phi * p22 + q_level
        p12_p = phi * p12 + phi * phi * p22
        p22_p = phi * phi * p22 + q_slope

        # --- update --------------------------------------------------------
        s = p11_p + r                      # innovation variance (scalar)
        if not np.isfinite(s) or s <= _EPS:
            return {"loglik": -np.inf, "level": np.nan, "slope": 0.0,
                    "P": np.eye(2) * 1e6, "innovations": innov,
                    "levels": levels, "slopes": slopes}
        e = float(y[t]) - level_p
        k1 = p11_p / s
        k2 = p12_p / s

        level = level_p + k1 * e
        slope = slope_p + k2 * e
        p11 = p11_p * (1.0 - k1)
        p12 = p12_p * (1.0 - k1)
        p22 = p22_p - k2 * p12_p

        innov[t] = e
        levels[t] = level
        slopes[t] = slope
        # Skip the first two: they are initialising, not being predicted.
        if t >= 2:
            sum_log_s += np.log(s)
            sum_e2_s += e * e / s
            n_eff += 1

    # CONCENTRATED (profile) log-likelihood. The filter runs with r = 1, so every
    # variance is known only up to a common factor sigma^2. Scoring the raw
    # Gaussian likelihood at that arbitrary scale does not compare parameter
    # RATIOS - it just rewards whichever ratio makes the innovation variance
    # closest to 1, which on log-price data (scale ~0.02) means driving q to
    # zero regardless of the truth. Profiling sigma^2 out analytically removes
    # the scale direction entirely and leaves a likelihood in the ratios alone.
    if n_eff < 3 or sum_e2_s <= _EPS:
        loglik = -np.inf
    else:
        sigma2 = sum_e2_s / n_eff
        loglik = -0.5 * (sum_log_s + n_eff * np.log(sigma2))

    out = {"loglik": float(loglik), "level": float(level), "slope": float(slope),
           "P": np.array([[p11, p12], [p12, p22]]), "innovations": innov,
           "sigma2": float(sum_e2_s / n_eff) if n_eff else float("nan")}
    if return_states:
        out.update({"levels": levels, "slopes": slopes})
    return out


# ==========================================
# 2. Fitting
# ==========================================
# Ratios relative to r = 1. The grid spans "almost no state noise" (the series is
# essentially a constant plus noise) to "state noise dominates" (a pure random
# walk), so the fit can land anywhere on that spectrum rather than being nudged.
_Q_LEVEL_GRID = np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 20.0])
_Q_SLOPE_GRID = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
_PHI_GRID = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 0.98])


def fit_kalman(log_prices, refine=True):
    """
    Maximum-likelihood fit of ``(q_level, q_slope, phi)`` with ``r`` profiled out.

    Only the noise *ratios* drive the filter's behaviour, so the search fixes
    ``r = 1`` and rescales afterwards using the mean standardised innovation —
    which turns a 4-parameter search into a 3-parameter one and removes the
    scale direction the likelihood is flattest along.

    Returns
    -------
    dict
        Fitted parameters, the log-likelihood, and the final state. Includes
        ``degenerate_to_random_walk``, which is True when the fitted slope
        variance or damping is effectively zero — the honest outcome for a
        near-efficient price series, and worth surfacing rather than hiding.
    """
    y = np.asarray(log_prices, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 30:
        return {"ok": False, "note": "insufficient history for a state-space fit"}

    best = {"loglik": -np.inf}
    for ql in _Q_LEVEL_GRID:
        for qs in _Q_SLOPE_GRID:
            for phi in _PHI_GRID:
                res = kalman_filter(y, ql, qs, 1.0, phi)
                if res["loglik"] > best["loglik"]:
                    best = {"loglik": res["loglik"], "q_level": float(ql),
                            "q_slope": float(qs), "phi": float(phi)}

    if not np.isfinite(best.get("loglik", -np.inf)):
        return {"ok": False, "note": "likelihood did not evaluate"}

    # Local refinement around the grid optimum: the grid is coarse on purpose
    # (to survive the multi-modality), so a short multiplicative walk recovers
    # the resolution the grid gave up.
    if refine:
        for _ in range(3):
            improved = False
            for factor in (0.5, 2.0):
                for key in ("q_level", "q_slope"):
                    cand = dict(best)
                    cand[key] = best[key] * factor
                    res = kalman_filter(y, cand["q_level"], cand["q_slope"],
                                        1.0, cand["phi"])
                    if res["loglik"] > best["loglik"] + 1e-9:
                        best = dict(cand, loglik=res["loglik"])
                        improved = True
            if not improved:
                break

    final = kalman_filter(y, best["q_level"], best["q_slope"], 1.0,
                          best["phi"], return_states=True)

    # Recover the absolute scale. sigma2 is exactly the factor profiled out of
    # the likelihood - the mean squared STANDARDISED innovation - so multiplying
    # every variance by it restores the fitted model to the data's units.
    mult = float(final.get("sigma2", 1.0))
    if not np.isfinite(mult) or mult <= _EPS:
        mult = 1.0

    return {
        "ok": True,
        "q_level": best["q_level"] * mult,
        "q_slope": best["q_slope"] * mult,
        "r": 1.0 * mult,
        "phi": best["phi"],
        "loglik": best["loglik"],
        "level": final["level"],
        "slope": final["slope"],
        "P": final["P"] * mult,
        "n_obs": int(len(y)),
        "degenerate_to_random_walk": bool(
            best["q_slope"] <= 1e-6 or best["phi"] <= 1e-9),
    }


# ==========================================
# 3. Forecasting
# ==========================================
def forecast_kalman(fit, horizon):
    """
    h-step-ahead means and variances from a fitted state.

    The damped trend contributes ``slope * sum(phi^i for i in 1..h)`` — a
    geometric sum, so the extrapolation converges rather than running away. The
    forecast variance grows with horizon through the accumulated state noise,
    which is what makes the cone widen without any conformal step.

    Returns
    -------
    (mean, var) : tuple of numpy arrays, both length ``horizon``, in log-price
    units.
    """
    if not fit.get("ok"):
        return np.array([]), np.array([])

    level, slope = fit["level"], fit["slope"]
    phi, ql, qs, r = fit["phi"], fit["q_level"], fit["q_slope"], fit["r"]
    p11, p12, p22 = fit["P"][0, 0], fit["P"][0, 1], fit["P"][1, 1]

    means = np.empty(horizon)
    variances = np.empty(horizon)
    lvl, slp = level, slope
    for h in range(horizon):
        # Propagate the state one step, exactly as the filter's predict step.
        lvl_next = lvl + phi * slp
        slp_next = phi * slp
        p11_n = p11 + 2.0 * phi * p12 + phi * phi * p22 + ql
        p12_n = phi * p12 + phi * phi * p22
        p22_n = phi * phi * p22 + qs

        lvl, slp = lvl_next, slp_next
        p11, p12, p22 = p11_n, p12_n, p22_n
        means[h] = lvl
        variances[h] = p11 + r        # observation variance, not just state
    return means, variances


# ==========================================
# 4. Project-facing entry point
# ==========================================
#: The configuration before section 115, which every run keeps unless a switch
#: is given: the static 95% quantile of the one-step residuals, times sqrt(h).
KALMAN_LEGACY = dict(aci=False, hstep=False)


def resolve_switches(interval, horizon, aci=None, hstep=None):
    """
    The switches a run uses. A switch left as None takes KALMAN_LEGACY's value;
    an explicit value is kept. ``interval`` and ``horizon`` are taken so a
    horizon-dependent default can be added without changing callers.
    """
    given = dict(aci=aci, hstep=hstep)
    return {k: KALMAN_LEGACY[k] if v is None else bool(v) for k, v in given.items()}


def hstep_errors(log_px, levels, slopes, fit_end, phi, horizon):
    """
    The h-step errors on the calibration tail, h = 1..horizon: from the
    filtered state at each origin t >= fit_end - 1, the forecast
    ``level_t + slope_t * (phi + ... + phi^h)`` against ``log_px[t + h]``. At
    h = 1 these are exactly the one-step innovations of the tail.
    """
    n = len(log_px)
    out = []
    for h in range(1, horizon + 1):
        s_h = float(sum(phi ** i for i in range(1, h + 1)))
        t = np.arange(fit_end - 1, n - h)
        e = log_px[t + h] - (levels[t] + slopes[t] * s_h) if len(t) else np.array([])
        out.append(e[np.isfinite(e)])
    return out


def _halfwidths(resid, errors, horizon, aci, hstep, min_n=20):
    """Conformal half-widths per step (README section 115).

    Static (shipped): the 95% quantile of |one-step residual| times sqrt(h).
    ``aci``: that quantile's level adapted along the residual stream (Gibbs and
    Candes, Adaptive Conformal Inference, 2021; ``model_utils.aci_quantile``).
    ``hstep``: step h takes the quantile of its own h-step errors (calibration
    per horizon, as Stankeviciute, Alaa and van der Schaar 2021), falling back
    to the one-step rule when fewer than ``min_n`` are available."""
    from model_utils import aci_quantile

    def q(x):
        x = np.abs(np.asarray(x, dtype=float))
        return float(aci_quantile(x)) if aci else float(np.quantile(x, 0.95))

    q1 = q(resid)
    half = np.array([q1 * np.sqrt(h + 1) for h in range(horizon)])
    if hstep:
        for h in range(horizon):
            if len(errors[h]) >= min_n:
                half[h] = q(errors[h])
    return half


def kalman_paths(df, interval, horizon, use_conformal=True, variants=((False, False),)):
    """
    Forecast paths for several (aci, hstep) settings from one fit:
    {(aci, hstep): DataFrame}. The switches change only the band; the fit and
    the centre are the same for every setting.
    """
    from model_utils import normalize_interval, generate_future_dates

    interval = normalize_interval(interval)
    variants = tuple((bool(a), bool(b)) for a, b in variants)

    work = df.copy()
    price_col = next((c for c in ("adj_close", "close", "price")
                      if c in work.columns), None)
    if price_col is None:
        raise ValueError("no price column found for the Kalman model")

    prices = pd.to_numeric(work[price_col], errors="coerce").dropna().values
    if len(prices) < 60:
        raise ValueError("insufficient history for the Kalman model")

    max_obs = 1000 if interval == "daily" else 750
    if len(prices) > max_obs:
        prices = prices[-max_obs:]
        work = work.tail(max_obs)

    log_px = np.log(np.maximum(prices, _EPS))

    # Fit on all but a calibration tail, so the conformal residuals are genuinely
    # out-of-sample with respect to the parameter estimates.
    cal_n = max(20, int(0.2 * len(log_px)))
    fit_end = len(log_px) - cal_n
    fit = fit_kalman(log_px[:fit_end])
    if not fit.get("ok"):
        raise ValueError(fit.get("note", "Kalman fit failed"))

    # One-step conformal residuals on the held-out tail, produced by running the
    # SAME fitted parameters forward — never refitting on the calibration block.
    cal_res = kalman_filter(log_px, fit["q_level"], fit["q_slope"], fit["r"],
                            fit["phi"], return_states=True)
    resid = cal_res["innovations"][fit_end:]
    resid = resid[np.isfinite(resid)]
    errors = None
    if any(b for _, b in variants):
        errors = hstep_errors(log_px, cal_res["levels"], cal_res["slopes"],
                              fit_end, fit["phi"], horizon)

    full = fit_kalman(log_px)
    if not full.get("ok"):
        full = fit
    means, variances = forecast_kalman(full, horizon)
    native_sd = np.sqrt(np.maximum(variances, _EPS))

    last_price = float(prices[-1])
    last_date = work["datetime"].iloc[-1] if "datetime" in work.columns else None
    future = generate_future_dates(last_date, horizon, interval)

    outs = {}
    for aci, hstep in variants:
        # Conformal half-width per step. One-step residuals scaled by sqrt(h) is
        # the standard random-walk widening; it makes no Gaussian claim, only
        # that the residual distribution is stationary over the calibration block.
        if use_conformal and len(resid) >= 20:
            half = _halfwidths(resid, errors, horizon, aci, hstep)
            band_mode = "conformal" + ("+aci" if aci else "") + ("+hstep" if hstep else "")
        else:
            half = _Z95 * native_sd
            band_mode = "native-gaussian"

        rows = []
        for h in range(horizon):
            centre = float(np.exp(means[h]))
            rows.append({
                "datetime": future[h],
                "predicted_price": centre,
                "lower_bound": float(np.exp(means[h] - half[h])),
                "upper_bound": float(np.exp(means[h] + half[h])),
                "volatility": float(centre * native_sd[h]),
            })
        out = pd.DataFrame(rows)

        out.attrs["kalman"] = {
            "q_level": full["q_level"], "q_slope": full["q_slope"],
            "r": full["r"], "phi": full["phi"], "loglik": full["loglik"],
            "slope": full["slope"], "n_obs": full["n_obs"],
            "degenerate_to_random_walk": full["degenerate_to_random_walk"],
            "band_mode": band_mode,
            "native_halfwidth": [float(v) for v in (_Z95 * native_sd)],
            "used_halfwidth": [float(v) for v in half],
            "n_calibration": int(len(resid)),
        }
        outs[(aci, hstep)] = out
    if full["degenerate_to_random_walk"]:
        print("Kalman: fitted slope variance ~ 0 - the series prefers a LOCAL "
              "LEVEL (random-walk) model, so the forecast is flat by choice.")
    return outs, full


def run_kalman_prediction(df, interval, horizon, use_conformal=True, aci=None,
                          hstep=None):
    """
    Forecast with a damped local linear trend, returning the project's schema.

    Parameters
    ----------
    use_conformal : bool
        When True (default) the interval half-widths are **conformalised** on
        held-out one-step residuals, exactly like every other model here, and
        the Kalman-native Gaussian width is reported alongside in ``attrs`` for
        comparison. When False the native width is used directly.

        Default True on purpose. The native interval is the book's selling point
        and it is genuinely elegant, but it is only as good as its Gaussian
        assumption, and this project has measured repeatedly that equity
        residuals are fat-tailed. Shipping the assumption-light version and
        *reporting* the assumption-heavy one is the honest arrangement;
        ``run_kalman_eval.py`` measures which is better calibrated.
    aci, hstep : bool or None
        None (default) takes KALMAN_LEGACY (README section 115). ``aci``: the
        residual quantile's level adapts along the calibration stream
        (adaptive conformal inference). ``hstep``: each step's half-width is the
        quantile of the tail's own h-step errors rather than the one-step
        quantile times sqrt(h). Both apply only to the conformal band.
    """
    from model_utils import normalize_interval

    interval = normalize_interval(interval)
    print(f"--- Starting Kalman (local trend) Prediction ({interval}) ---")
    sw = resolve_switches(interval, horizon, aci, hstep)
    v = (sw["aci"], sw["hstep"])
    outs, full = kalman_paths(df, interval, horizon, use_conformal, variants=(v,))
    out = outs[v]
    print(f"Kalman fit (phi={full['phi']:.2f}, q_level={full['q_level']:.2e}, "
          f"q_slope={full['q_slope']:.2e}); band={out.attrs['kalman']['band_mode']}")
    print("Kalman Prediction Complete.")
    return out
```
