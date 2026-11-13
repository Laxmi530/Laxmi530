- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
HAR-RV: Heterogeneous AutoRegressive model of Realized Volatility (Corsi, 2009).

The backtest established the uncomfortable but honest truth of this domain: at
single-name daily/hourly horizons the **return** (first moment) is essentially a
random walk — no feature set reliably beat it. But the **second moment** is a
different story: volatility clusters and is genuinely forecastable (it is exactly
why the GARCH layer works). HAR-RV is the cheap, hard-to-beat workhorse for that
job, and it is the natural home for "robustness" once you accept that the price
line itself carries little signal.

This model therefore makes **no directional claim**. Its point forecast is the
random walk (last price held flat), and its real output is a **calibrated
uncertainty cone** driven by a forecast of realized variance:

1. Build a realized-variance series from the bar returns (``RV_t = r_t^2`` — the
   standard proxy when sub-bar data is unavailable; with hourly bars each value
   is already an intraday variance contribution).
2. Fit the HAR regression on ``log RV``: today's (log) variance is explained by
   its **daily**, **weekly** and **monthly** lagged averages — three horizons of
   memory that capture volatility's long-range persistence with just four
   parameters and ordinary least squares.
3. Convert the log forecast back to a variance level with a **data-estimated
   offset** (:func:`har_level_offset`). This matters: the earlier implementation
   used the log-normal Jensen correction ``exp(mu + resid_var/2)``, which assumes
   a Gaussian residual — but the residual of a regression on ``log(r^2)`` is
   log-chi-squared, and the mismatch over-stated volatility by a measured ~1.9x.
   The conformal layer in step 4 hid the error (an inflated sigma is cancelled by
   a deflated multiplier, so coverage stayed correct), but the *reported*
   per-step volatility was wrong. See the README section "Volatility model
   comparison — and a latent bias it exposed in HAR-RV".
4. Roll the fitted model forward ``horizon`` steps to get per-step variance
   forecasts, sum them (variance is additive across independent steps) to get the
   cumulative variance to each horizon, and wrap the flat price path in a
   ``last_price * exp(±k_h * sqrt(cum_var))`` band whose multiplier ``k_h`` is
   conformally calibrated per horizon.

The result is a model whose point RMSE is, by construction, that of the random
walk (it *is* the random walk) but whose **interval coverage** is its reason to
exist — the deliverable when the point forecast is within noise. It plugs into
the same ``run_*_prediction(df, interval, horizon)`` contract as every other
model and feeds the UI's risk/probability panel.
"""

import numpy as np
import pandas as pd

from model_utils import (
    normalize_interval,
    coerce_numeric_columns,
    generate_future_dates,
    widen_to_monotone,
)

try:
    import iv_data                     # India-VIX forward-vol tilt (optional)
except Exception:                      # never let an import issue break the model
    iv_data = None

_Z95 = 1.959963984540054   # 97.5th percentile of the standard normal
_Z05 = 1.6448536269514722  # 95th percentile (one-sided), for VaR-style reads
_EPS = 1e-12


def _har_lags(interval):
    """
    HAR memory horizons (short / medium / long), in bars, for the interval.

    Daily uses the canonical day/week/month (1, 5, 22). Hourly uses
    bar / ~day / ~week of ~6-7-hour sessions (1, 7, 35).
    """
    return (1, 5, 22) if interval == 'daily' else (1, 7, 35)


def _har_regressors(log_rv, i, lags):
    """
    Build the three HAR regressors at index ``i`` from PAST log-RV only.

    Returns (x_short, x_med, x_long) = the most recent value, the mean over the
    medium window, and the mean over the long window — all strictly before ``i``.
    """
    s, m, l = lags
    x_s = log_rv[i - 1]
    x_m = np.mean(log_rv[max(0, i - m):i])
    x_l = np.mean(log_rv[max(0, i - l):i])
    return x_s, x_m, x_l


def har_level_offset(log_rv):
    """
    Offset ``c`` that makes ``exp(mu + c)`` an **unbiased** variance estimate.

    The HAR regression predicts the mean of ``log(r^2)``, and exponentiating that
    mean does not give the conditional variance. Writing ``r = sigma * z``,
    ``log(r^2) = log(sigma^2) + log(z^2)``, and ``E[log z^2] = -1.27`` for a
    Gaussian ``z`` (the mean of a log-chi-squared(1)) — so ``exp(prediction)``
    understates ``sigma^2`` by roughly ``e^-1.27``.

    The obvious repair, the log-normal Jensen correction ``exp(mu +
    resid_var/2)``, is **wrong here** and was the source of a measured ~1.9x
    over-statement of volatility in this model: it assumes a Gaussian residual,
    but the residual of a regression on ``log(r^2)`` is log-chi-squared, whose
    variance is ``pi^2/2 ~ 4.93``. That inflates the variance by ``e^2.46 ~ 11.7``
    where the correct factor is ``e^1.27 ~ 3.6``.

    So the offset is estimated from the data instead: the constant that makes the
    level match the realised mean variance,
    ``c = log(mean(RV)) - mean(log RV)``. On Gaussian returns this reproduces the
    theoretical 1.27, and it self-adjusts for the percentile clipping applied to
    ``log_rv`` and for genuinely fat-tailed returns. Because it is computed from
    whatever slice of history the caller passes, using it inside the
    walk-forward calibration loop stays leak-free.

    Parameters
    ----------
    log_rv : array-like
        Log realized-variance series (already clipped, as the caller does).

    Returns
    -------
    float
        The additive offset in log-variance space.
    """
    lr = np.asarray(log_rv, dtype=np.float64)
    if len(lr) == 0:
        return 0.0
    mean_rv = float(np.mean(np.exp(lr)))
    return float(np.log(max(mean_rv, _EPS)) - float(np.mean(lr)))


def fit_har(log_rv, lags):
    """
    Fit the HAR OLS model ``logRV_t ~ 1 + short + medium + long``.

    Parameters
    ----------
    log_rv : np.ndarray
        Log realized-variance series.
    lags : tuple of int
        (short, medium, long) memory windows in bars.

    Returns
    -------
    tuple(np.ndarray, float)
        (coefficients [b0, b_s, b_m, b_l], residual variance). The residual
        variance is returned as a **fit diagnostic** — it is deliberately *not*
        used to transform the log forecast back to a level; see
        :func:`har_level_offset` for why the Jensen correction is the wrong tool
        for a log-chi-squared residual.
    """
    long_lag = lags[2]
    rows, targets = [], []
    for i in range(long_lag, len(log_rv)):
        x_s, x_m, x_l = _har_regressors(log_rv, i, lags)
        rows.append([1.0, x_s, x_m, x_l])
        targets.append(log_rv[i])
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    resid_var = float(np.var(resid)) if len(resid) > 1 else 0.0
    return coef, resid_var


def forecast_har_variance(log_rv, coef, lags, horizon, level_offset=None):
    """
    Recursively forecast per-step realized variance for ``horizon`` steps.

    Each step's log-RV is predicted from the daily/weekly/monthly averages of the
    observed-then-forecasted history, then converted back to a variance level with
    the **data-estimated offset** from :func:`har_level_offset`, so the result is
    an unbiased variance rather than a ~1.9x over-statement.

    Parameters
    ----------
    log_rv : array-like
        Observed (clipped) log realized-variance history up to the origin.
    coef : np.ndarray
        HAR coefficients ``[b0, b_s, b_m, b_l]`` from :func:`fit_har`.
    lags : tuple of int
        (short, medium, long) memory windows in bars.
    horizon : int
        Number of steps to forecast.
    level_offset : float, optional
        Log-to-level offset. Defaults to :func:`har_level_offset` computed on the
        ``log_rv`` slice that was passed — so inside the walk-forward calibration
        loop each origin derives its own offset from past bars only, and the
        serving call uses the full observed history.

    Returns
    -------
    np.ndarray
        ``horizon`` per-step variance forecasts (in price-return^2 units).
    """
    if level_offset is None:
        level_offset = har_level_offset(log_rv)
    buf = list(log_rv)
    out = []
    for _ in range(horizon):
        i = len(buf)
        x_s, x_m, x_l = _har_regressors(np.asarray(buf), i, lags)
        mu = coef[0] + coef[1] * x_s + coef[2] * x_m + coef[3] * x_l
        buf.append(mu)
        out.append(np.exp(mu + level_offset))
    return np.asarray(out, dtype=np.float64)


def decay_weights(ages, halflife):
    """
    Weights ``0.5 ** (age / halflife)`` for calibration scores ``age``
    calibration points old: the newest has age 0 (README section 93). This is
    Barber et al.'s ``rho ** (n + 1 - i)``, counted in points rather than
    sessions, so the weight available to reach the 95% level does not depend
    on how far apart the calibration origins are (about 6 sessions on a few
    years of history, about 27 on decades). ``halflife=None`` gives equal
    weights, the shipped unweighted quantile.
    """
    ages = np.asarray(ages, dtype=np.float64)
    if halflife is None:
        return np.ones_like(ages)
    return np.power(0.5, ages / float(halflife))


def weighted_quantile(scores, weights, level):
    """
    Nonexchangeable split-conformal quantile (Barber, Candes, Ramdas and
    Tibshirani 2023): the ``level`` quantile of the scores with ``weights`` plus
    a point at +infinity of weight 1 for the test point. With equal weights it
    is the usual finite-sample-corrected conformal quantile. When the finite
    weights cannot reach ``level`` the quantile would be infinite; the largest
    score is returned instead, so a band can widen but never become infinite.
    """
    z = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(z) == 0:
        return float("nan")
    order = np.argsort(z)
    z, w = z[order], w[order]
    cum = np.cumsum(w) / (w.sum() + 1.0)
    hit = np.searchsorted(cum, min(max(level, 0.0), 1.0) - 1e-12)
    return float(z[min(hit, len(z) - 1)])


#: DtACI's grid of learning rates (Gibbs and Candes, arXiv 2208.08401, section 4).
DTACI_GAMMAS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)


def _dtaci_alpha(pool, quantile, base_alpha=0.05, gammas=DTACI_GAMMAS,
                 alpha_lo=0.005, alpha_hi=0.30):
    """
    Final miscoverage level from DtACI (Gibbs and Candes, Algorithm 2) run over
    the time-ordered score stream ``pool``: one ACI expert per learning rate,
    weighted by exponentiated pinball loss and mixed with the uniform weights.
    ``quantile(levels)`` maps an array of levels to score thresholds. eta and sigma follow
    the paper's recommendations with the stream length as the interval.
    """
    pool = np.asarray(pool, dtype=np.float64)
    n, k = len(pool), len(gammas)
    a = base_alpha
    if n == 0:
        return a
    sigma = 1.0 / (2 * n)
    eta = np.sqrt(3.0 / n) * np.sqrt((np.log(k * n) + 2) / ((1 - a) ** 2 * a ** 2))
    alphas = np.full(k, a)
    w = np.ones(k)
    srt = np.sort(pool)
    for s_t in pool:
        beta = 1.0 - np.searchsorted(srt, s_t, side="left") / n    # level at which s_t is just covered
        d = beta - alphas
        loss = a * d - np.minimum(0.0, d)                           # pinball loss
        wb = w * np.exp(-eta * (loss - loss.min()))
        w = (1 - sigma) * wb + wb.sum() * sigma / k
        err = (s_t > np.asarray(quantile(1.0 - np.clip(alphas, alpha_lo, alpha_hi)))).astype(float)
        alphas = np.clip(alphas + np.asarray(gammas) * (a - err), alpha_lo, alpha_hi)
    return float(np.clip(np.dot(w / w.sum(), alphas), alpha_lo, alpha_hi))


def _conformal_band_multipliers(prices, log_rv, coef, lags, horizon,
                                alpha=0.05, cal_frac=0.30, min_cal=20,
                                max_origins=80, fit_window=None, halflife=None,
                                return_ages=False):
    """
    Calibrate a **per-horizon** cone multiplier from cumulative standardized
    residuals (split-conformal calibration of the HAR variance cone).

    A fixed Gaussian multiplier (1.96) mis-covers because (a) returns are
    fat-tailed and (b) the additive ``sqrt(sum of step variances)`` cone does not
    exactly match the realised h-step dispersion. So for each calibration origin
    we roll the HAR variance forecast out ``horizon`` steps, form the cumulative
    forecast sigma to each horizon ``h``, and record the standardized cumulative
    residual ``z_{t,h} = |log(P[t+h]/P[t])| / sigma_cum_h``. The (1-α) empirical
    quantile of ``z_{·,h}`` (with the finite-sample correction) is the multiplier
    ``k_h`` for horizon ``h`` — exactly the level that makes the cone cover the
    *realised h-step move* at the target rate, distribution-free and per horizon.

    All residuals use only bars strictly before each origin (the variance
    forecast is genuinely out-of-sample), so inside the walk-forward backtest
    this never peeks at the evaluation fold.

    Returns
    -------
    tuple(np.ndarray or None, dict)
        ``(mult, signed)`` where ``mult`` is the length-``horizon`` multipliers
        ``[k_1, …, k_horizon]`` (None if too few calibration origins), and
        ``signed`` maps each horizon ``h`` to the array of **signed** standardized
        cumulative residuals ``log(P[t+h]/P[t]) / sigma_cum_h``. The signed
        residuals are the empirical, fat-tailed return distribution (in units of
        forecast sigma) that the risk panel reads probabilities off — so the
        downstream P(up)/P(move)/VaR keep the fat tails instead of re-imposing a
        Gaussian.

    ``fit_window`` (sessions) computes each origin's level offset from that many
    bars before it instead of all of them, matching a fit on recent history
    (README section 93). ``halflife`` weights the scores by age with
    :func:`decay_weights` when taking the quantile. ``return_ages=True`` adds a
    third element: each calibration origin's age in calibration points (the
    newest is 0), for weighting.
    """
    n_prices = len(prices)
    long_lag = lags[2]
    last_p = n_prices - 1 - horizon
    first_p = max(long_lag + 1,
                  last_p - max(min_cal, int(round(len(log_rv) * cal_frac))))
    origins = list(range(first_p, last_p + 1))
    if len(origins) < min_cal:
        return None, {}
    if len(origins) > max_origins:
        stride = int(np.ceil(len(origins) / max_origins))
        origins = origins[::stride]

    resid = {h: [] for h in range(1, horizon + 1)}        # |z| for the multiplier
    signed = {h: [] for h in range(1, horizon + 1)}       # z for the distribution
    for p in origins:
        # No level_offset passed: each origin derives its own from log_rv[:p],
        # i.e. from bars strictly before the origin (leak-free).
        lo_p = 0 if fit_window is None else max(0, p - int(fit_window))
        sv = forecast_har_variance(log_rv[lo_p:p], coef, lags, horizon)
        cum_sigma = np.sqrt(np.cumsum(sv))
        for h in range(1, horizon + 1):
            if cum_sigma[h - 1] > 0:
                cr = np.log(prices[p + h] / prices[p])
                resid[h].append(abs(cr) / cum_sigma[h - 1])
                signed[h].append(cr / cum_sigma[h - 1])

    ages = np.arange(len(origins), dtype=np.float64)[::-1]   # newest origin: age 0
    mult = np.empty(horizon)
    for h in range(1, horizon + 1):
        z = resid[h]
        m = len(z)
        if m >= min_cal:
            if halflife is None:
                q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
                mult[h - 1] = float(np.quantile(z, q_level))
            else:
                mult[h - 1] = weighted_quantile(z, decay_weights(ages[:m], halflife), 1 - alpha)
        else:
            mult[h - 1] = _Z95
    signed = {h: np.asarray(v, dtype=np.float64) for h, v in signed.items()}
    if return_ages:
        return mult, signed, ages
    return mult, signed


def _aci_multipliers(signed, horizon, base_alpha=0.05, gamma=0.05,
                     alpha_lo=0.005, alpha_hi=0.30, weights=None, method="aci"):
    """
    Adaptive Conformal Inference (Gibbs & Candès, 2021) refinement of the
    per-horizon cone multiplier.

    The static multiplier ``k_h`` is the fixed ``(1-α)`` quantile of the whole
    calibration pool — it cannot tell that the *recent* stretch has been
    systematically under-covering (a volatility regime shift), which is exactly
    the failure mode the backtest exposed. ACI fixes that by adapting the
    miscoverage level online: walking the calibration residuals in time order it
    updates ``α_{t+1} = α_t + γ (α_target - err_t)`` where ``err_t = 1`` if the
    realised standardised residual fell outside the current band. A run of misses
    (high-vol regime) drives ``α`` down → a higher quantile → a wider cone; a run
    of easy covers nudges it back up → tighter. Because ``α`` carries memory, the
    value at the end of the stream is recency-weighted, so the live multiplier
    ``k_h = quantile(|z|, 1 - α_final)`` reflects the *current* regime rather than
    the long-run average.

    The score quantile is taken over the full pool (stable), and only ``α`` is
    adapted — the standard ACI move — with ``α`` clipped to ``[alpha_lo,
    alpha_hi]`` so the cone can neither collapse nor explode. Falls back to the
    static ``(1-base_alpha)`` quantile for any horizon with too few residuals.

    Parameters
    ----------
    signed : dict[int, np.ndarray]
        Per-horizon signed standardized residuals (time-ordered by origin), as
        returned by :func:`_conformal_band_multipliers`.
    horizon : int
        Number of steps.
    base_alpha : float, optional
        Target miscoverage (0.05 -> 95% cone). Default 0.05.
    gamma : float, optional
        ACI learning rate. Default 0.05 (conservative).
    alpha_lo, alpha_hi : float, optional
        Clipping bounds for the adapted ``α``.
    weights : array-like, optional
        Weights of the scores in time order (:func:`decay_weights`); the
        quantile is then :func:`weighted_quantile`. None: the plain quantile.
    method : str, optional
        ``'aci'`` (shipped) or ``'dtaci'``, DtACI's mixture of learning rates
        (:func:`_dtaci_alpha`, README section 93).

    Returns
    -------
    np.ndarray
        Length-``horizon`` ACI-adapted multipliers.
    """
    mult = np.full(horizon, _Z95)
    for h in range(1, horizon + 1):
        z = signed.get(h)
        if z is None or len(z) < 20:
            continue
        pool = np.abs(np.asarray(z, dtype=np.float64))   # |z| score distribution
        if weights is None:
            def q(level, pool=pool):
                return np.quantile(pool, level) if np.ndim(level) else float(np.quantile(pool, level))
        else:
            w = np.asarray(weights, dtype=np.float64)[:len(pool)]
            order = np.argsort(pool)
            srt, cum = pool[order], np.cumsum(w[order]) / (w.sum() + 1.0)

            def q(level, srt=srt, cum=cum):     # weighted_quantile, sorted once
                i = np.minimum(np.searchsorted(cum, np.clip(level, 0.0, 1.0) - 1e-12), len(srt) - 1)
                return srt[i] if np.ndim(level) else float(srt[i])
        if method == "dtaci":
            mult[h - 1] = q(1.0 - _dtaci_alpha(pool, q, base_alpha, alpha_lo=alpha_lo,
                                                alpha_hi=alpha_hi))
            continue
        a = base_alpha
        for s in pool:                                   # stream in time order
            k_t = q(1.0 - min(max(a, alpha_lo), alpha_hi))
            err = 1.0 if s > k_t else 0.0
            a = a + gamma * (base_alpha - err)
            a = min(max(a, alpha_lo), alpha_hi)
        mult[h - 1] = q(1.0 - a)
    return mult


def _realized_semivariance(returns):
    """
    Split realized variance into **downside** and **upside** semivariance.

    Barndorff-Nielsen, Kinnebrock & Shephard (2010): ``RS^- = sum r^2 1{r<0}``
    and ``RS^+ = sum r^2 1{r>0}``, with ``RS^- + RS^+ = RV``. Patton & Sheppard
    (2015, "Good Volatility, Bad Volatility") show the *downside* semivariance
    carries most of the predictive content for future volatility — the empirical
    basis for treating the left tail specially.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Per-bar ``(rs_down, rs_up)`` contributions (each ``r^2`` gated by sign).
    """
    r = np.asarray(returns, dtype=np.float64)
    rs_down = np.where(r < 0, r * r, 0.0)
    rs_up = np.where(r > 0, r * r, 0.0)
    return rs_down, rs_up


def _aci_one_sided(scores, base_alpha, gamma=0.05, alpha_lo=0.0025, alpha_hi=0.20):
    """
    One-sided Adaptive Conformal Inference multiplier for a single tail.

    Same online ``α`` update as :func:`_aci_multipliers` but over a one-sided
    score stream (the signed standardized residuals of *one* tail), targeting a
    one-sided miscoverage ``base_alpha`` (= half the two-sided level). Used to
    build the asymmetric cone's lower and upper edges independently so each tail
    is calibrated to its own — possibly fatter — side of the distribution.
    """
    pool = np.asarray(scores, dtype=np.float64)
    if len(pool) < 20:
        return _Z95
    a = base_alpha
    for s in pool:
        k_t = float(np.quantile(pool, 1.0 - min(max(a, alpha_lo), alpha_hi)))
        err = 1.0 if s > k_t else 0.0
        a = a + gamma * (base_alpha - err)
        a = min(max(a, alpha_lo), alpha_hi)
    return float(np.quantile(pool, 1.0 - min(max(a, alpha_lo), alpha_hi)))


def _asymmetric_multipliers(signed, horizon, adaptive, base_alpha=0.05,
                            gamma=0.05):
    """
    Per-horizon **lower** and **upper** cone multipliers from signed residuals.

    A symmetric ``exp(±k·sigma)`` band assumes the up- and down-moves are equally
    dispersed; for equities they are not (the downside is typically fatter). Here
    each side gets its own multiplier from the empirical signed standardized
    residuals — the right-tail quantile for the upper edge and the left-tail
    magnitude for the lower edge — each at the ``base_alpha/2`` one-sided level so
    the two-sided coverage target is preserved while the *shape* is asymmetric.
    Static uses the fixed empirical quantile; adaptive runs one-sided ACI per tail
    (:func:`_aci_one_sided`).

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        ``(k_lower, k_upper)`` length-``horizon`` multiplier arrays.
    """
    half = base_alpha / 2.0
    k_lo = np.full(horizon, _Z95)
    k_up = np.full(horizon, _Z95)
    for h in range(1, horizon + 1):
        z = signed.get(h)
        if z is None or len(z) < 20:
            continue
        z = np.asarray(z, dtype=np.float64)
        if adaptive:
            k_up[h - 1] = _aci_one_sided(z, half, gamma=gamma)
            k_lo[h - 1] = _aci_one_sided(-z, half, gamma=gamma)
        else:
            q = min(1.0, np.ceil((len(z) + 1) * (1 - half)) / len(z))
            k_up[h - 1] = float(np.quantile(z, q))
            k_lo[h - 1] = float(np.quantile(-z, q))
    # Guard against degenerate (non-positive) multipliers from tiny samples.
    k_lo = np.where(k_lo > 0, k_lo, _Z95)
    k_up = np.where(k_up > 0, k_up, _Z95)
    return k_lo, k_up


def _fallback_variance(returns, horizon):
    """
    EWMA (RiskMetrics, lambda=0.94) per-step variance when data is too short
    to fit HAR. Returns a constant per-step variance repeated over the horizon.
    """
    lam = 0.94
    var = float(np.var(returns)) if len(returns) > 1 else _EPS
    for r in returns:
        var = lam * var + (1 - lam) * r * r
    return np.full(horizon, max(var, _EPS))


def default_aci_method(interval, horizon):
    """
    HAR-RV's conformal method for a run: ``'dtaci'`` on daily runs of at most
    ``gating.MAX_DIRECTIONAL_SESSIONS`` (13) sessions, ``'aci'`` otherwise.

    DtACI passed a test written down in advance on a third set of 59 stocks,
    2019-2024: the five-session interval score fell from 37.28% to 35.98%
    (bootstrap interval of the gain [+0.73, +1.83]) with coverage of 94.5% at
    one session, 94.0% at five and 93.5% at thirteen (run_har_dtaci_short.py,
    README section 100). Beyond 13 sessions it twice covered 89.9% against a
    90% bar (sections 93 and 99), and hourly runs were not tested, so both keep
    ACI.
    """
    from gating import MAX_DIRECTIONAL_SESSIONS
    short_daily = (normalize_interval(interval) == "daily"
                   and int(horizon) <= MAX_DIRECTIONAL_SESSIONS)
    return "dtaci" if short_daily else "aci"


def run_har_rv_prediction(df, interval, horizon, adaptive=True, aci_gamma=0.05,
                          iv_adjust=False, asymmetric=False,
                          rv_estimator="squared", fit_window=None,
                          cal_halflife=None, aci_alpha_lo=0.005,
                          aci_method=None):
    """
    Forecast a volatility cone with HAR-RV; point path is the random walk.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close'.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to forecast.
    adaptive : bool, optional
        When True (default), refine the per-horizon conformal multiplier with
        **Adaptive Conformal Inference** (:func:`_aci_multipliers`) so the cone
        responds to a recent volatility regime shift rather than only the
        long-run calibration average. Set False for the plain static-conformal
        cone (used by the calibration harness to A/B the two).
    aci_gamma : float, optional
        ACI learning rate when ``adaptive`` is True. Default 0.05.
    iv_adjust : bool, optional
        When True, multiply the cone by a **forward-looking India-VIX factor**
        (:mod:`iv_data`) so it widens/tightens with the market's *implied* vol
        relative to recently realized vol, scaled by the stock's market linkage.
        Leak-free (the factor is taken as of the last bar). Off by default;
        A/B-gated via the calibration harness. No-op if VIX data is unavailable.
    rv_estimator : str, optional
        Which realized-variance proxy feeds the HAR regression
        (:mod:`range_volatility`). ``'squared'`` (default) is ``r_t**2``, the
        incumbent. ``'parkinson'``, ``'garman_klass'`` and ``'rogers_satchell'``
        use the **high/low** the app already fetches and are 1.7-2.4x more
        precise on real NSE bars (measured; the textbook figure of ~5x assumes
        a continuous price path). Range estimators are intraday-only, so the
        series is rescaled to close-to-close level via
        :func:`range_volatility.scale_to_total` before it reaches the
        regression. Default stays ``'squared'`` so no published figure moves
        until the A/B justifies a change.
    asymmetric : bool, optional
        When True, build an **asymmetric cone** with separate lower/upper
        multipliers from the empirical signed residuals
        (:func:`_asymmetric_multipliers`), so the downside edge can be wider than
        the upside when the loss tail is fatter — the "how bad could this get"
        refinement. Off by default; A/B-gated on coverage *and tail-miss balance*.
    fit_window : int, optional
        Fit the HAR regression and its level offset on the last ``fit_window``
        bars only (and each calibration origin's offset likewise). None: all
        history, as shipped. README section 93.
    cal_halflife : float, optional
        Weight calibration scores by age with this half-life, counted in
        calibration points (nonexchangeable conformal, Barber et al. 2023;
        :func:`decay_weights`). None: equal weights.
    aci_alpha_lo : float, optional
        The floor of ACI's adapted miscoverage level; 0.005 as shipped.
    aci_method : str, optional
        ``'aci'`` or ``'dtaci'`` (Gibbs and Candes' mixture of learning rates).
        None (default) takes :func:`default_aci_method`: DtACI on daily runs of
        13 sessions or fewer (README section 100), ACI otherwise.

    Returns
    -------
    pd.DataFrame
        Columns ['datetime', 'predicted_price', 'lower_bound', 'upper_bound',
        'volatility']. The predicted price is flat at the last observed value
        (no directional claim); the band is the HAR-RV uncertainty cone.
    """
    interval = normalize_interval(interval)
    if aci_method is None:
        aci_method = default_aci_method(interval, horizon)
    print(f"--- Starting HAR-RV (volatility) Prediction ({interval}) ---")

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    coerce_numeric_columns(df)

    prices = df['adj_close'].dropna().values.astype(np.float64)
    prices = prices[prices > 0]
    last_price = float(prices[-1])
    last_date = df.index[-1]
    future_dates = generate_future_dates(last_date, horizon, interval)

    returns = np.diff(np.log(prices))
    lags = _har_lags(interval)

    # Realized-variance proxy per bar; clip extreme log-RV for OLS stability.
    if rv_estimator == "squared":
        rv = returns ** 2
    else:
        # Range estimators need OHLC. They are computed on RAW open/high/low/
        # close on purpose: every one of them is a ratio WITHIN a bar, so they
        # are invariant to split/dividend adjustment (unlike adj_close, which is
        # what `returns` above is built from). Falling back to squared returns
        # when the columns are absent keeps a caller that asked for Parkinson
        # from silently getting the incumbent -- so it raises instead.
        import range_volatility as _rvm
        # Select the SAME bars `prices` was built from, so the range series and
        # the return series cannot drift out of alignment.
        _ok = df["adj_close"].notna() & (df["adj_close"] > 0)
        _bars = df.loc[_ok, [c for c in ("open", "high", "low", "close")
                             if c in df.columns]].reset_index(drop=True)
        rv_full = _rvm.estimate(_bars, rv_estimator)          # one per bar
        # Intraday-only estimators miss the overnight gap, so rescale to
        # close-to-close level using the ADJUSTED closes. Pairing a split-immune
        # range term with an unadjusted return series is a known trap: on real
        # data it inflated one name's factor from 1.26 to 8.32.
        _scaled, _scale_info = _rvm.scale_to_total(rv_full, prices)
        # `returns` has one fewer element than the bars; drop the first bar.
        rv = _scaled[1:] if len(_scaled) == len(returns) + 1 else _scaled[-len(returns):]
        rv = np.where(np.isfinite(rv), rv, returns ** 2)
    if len(rv) >= lags[2] + 15:
        log_rv = np.log(rv + _EPS)
        lo, hi = np.percentile(log_rv, [1, 99])
        log_rv = np.clip(log_rv, lo, hi)
        # The fit residual variance is a diagnostic only — it is deliberately not
        # used for the log->level transform (see `har_level_offset`).
        fit_rv = log_rv if fit_window is None else log_rv[-int(fit_window):]
        coef, _resid_var = fit_har(fit_rv, lags)
        offset = har_level_offset(fit_rv)
        step_var = forecast_har_variance(
            fit_rv, coef, lags, horizon, level_offset=offset)
        # Conformalize the cone per horizon: data-driven, fat-tail-aware. The
        # signed standardized residuals are the empirical return distribution the
        # risk panel reads probabilities off (fat tails preserved).
        mult, signed_resid, ages = _conformal_band_multipliers(
            prices, log_rv, coef, lags, horizon, fit_window=fit_window,
            halflife=cal_halflife, return_ages=True)
        if mult is None:
            mult = np.full(horizon, _Z95)
        mode = "static"
        if adaptive and signed_resid:
            aci_mult = _aci_multipliers(
                signed_resid, horizon, gamma=aci_gamma, alpha_lo=aci_alpha_lo,
                weights=None if cal_halflife is None else decay_weights(ages, cal_halflife),
                method=aci_method)
            mult = aci_mult
            mode = "ACI-adaptive" if aci_method == "aci" else "DtACI-adaptive"
        print(f"HAR-RV fit (lags={lags}); next-step vol "
              f"{np.sqrt(step_var[0]):.4%} (level offset {offset:.3f}); "
              f"{mode} conformal k(h) {np.round(mult, 2)} (normal=1.96)")
    else:
        step_var = _fallback_variance(returns, horizon)
        mult = np.full(horizon, _Z95)
        signed_resid = {}
        print("HAR-RV: insufficient data — using EWMA volatility fallback.")

    # Asymmetric cone (optional): separate lower/upper multipliers from the
    # signed residuals, so a fatter loss tail gives a wider downside edge.
    k_lo = k_up = None
    if asymmetric and signed_resid:
        k_lo, k_up = _asymmetric_multipliers(
            signed_resid, horizon, adaptive=adaptive, gamma=aci_gamma)

    # Variance is additive across (assumed-uncorrelated) steps.
    cum_sigma = np.sqrt(np.cumsum(step_var))

    # Optional forward-looking tilt: scale the whole cone by the India-VIX
    # implied/realized factor. Applied to cum_sigma so the band AND the empirical
    # risk distribution (which uses cum_sigma below) stay mutually consistent.
    iv_info = {"multiplier": 1.0, "reason": "disabled"}
    if iv_adjust and iv_data is not None:
        iv_frame = pd.DataFrame(
            {"datetime": df.index, "adj_close": df["adj_close"].values})
        iv_mult, iv_info = iv_data.forward_vol_factor(iv_frame, interval)
        cum_sigma = cum_sigma * float(iv_mult)
        print(f"HAR-RV IV tilt: x{iv_mult:.3f} "
              f"(VIX={iv_info.get('vix')}, F={iv_info.get('F')}, "
              f"R2={iv_info.get('linkage_r2')})")

    rows = []
    for i in range(horizon):
        sig = float(cum_sigma[i])
        if k_lo is not None:                       # asymmetric cone
            kl, ku = float(k_lo[i]), float(k_up[i])
        else:                                      # symmetric cone
            kl = ku = float(mult[i])
        rows.append({
            'datetime': future_dates[i],
            'predicted_price': last_price,                       # random walk
            'lower_bound': float(last_price * np.exp(-kl * sig)),
            'upper_bound': float(last_price * np.exp(ku * sig)),
            'volatility': float(np.sqrt(step_var[i])),           # per-step vol
        })

    out = pd.DataFrame(rows)
    # The per-step multipliers are calibrated separately and can dip, which
    # made the 3-day range narrower than the 2-day one; never let it narrow.
    widen_to_monotone(out)

    # Expose the empirical return distribution for the risk panel: the signed
    # standardized residuals at the final horizon, scaled by the horizon's
    # cumulative sigma, give a fat-tailed sample of horizon-end log-returns
    # (point forecast is the random walk, so these are centred at 0). The risk
    # panel reads P(up)/P(move>X)/VaR straight off this sample — no Gaussian.
    horizon_resid = signed_resid.get(horizon)
    if horizon_resid is not None and len(horizon_resid) >= 20:
        out.attrs['risk_log_returns'] = (
            np.asarray(horizon_resid, dtype=np.float64) * float(cum_sigma[-1]))
        out.attrs['risk_source'] = 'HAR-RV conformal residuals (empirical)'

    if iv_adjust:
        out.attrs['iv_adjust'] = iv_info

    # Downside-semivariance transparency: what share of recent realized variance
    # came from down-moves (Patton-Sheppard's "bad vol"). > 0.5 => the downside
    # dominates the stock's recent variance.
    rs_down, rs_up = _realized_semivariance(returns[-min(len(returns), 60):])
    denom = float(rs_down.sum() + rs_up.sum())
    if denom > 0:
        out.attrs['downside_share'] = round(float(rs_down.sum()) / denom, 3)
    if asymmetric and k_lo is not None:
        out.attrs['cone'] = 'asymmetric'
        out.attrs['cone_k'] = {'lower': float(k_lo[-1]), 'upper': float(k_up[-1])}

    print("HAR-RV Prediction Complete.")
    return out
```
