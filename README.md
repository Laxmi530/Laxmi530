- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Value-at-Risk estimation and **VaR backtesting**.

Motivated by *Hands-On Deep Learning for Finance* (Troiano et al.), Chapter 9,
"Risk Measurement Using GAN" — whose opening sections are "Estimating value at
risk" and "Computing methods and drawbacks", and whose closing section is
"Benchmarking results". This module implements that arc: the classical VaR
estimators, their documented drawbacks, a robust alternative, a generative
sampler in the spirit of the chapter, and — critically — the **statistical tests
that decide whether a VaR number was actually right**.

The gap this fills
------------------
The app already *reports* risk: ``risk_metrics.empirical_risk`` reads a 5% VaR
and expected shortfall off HAR-RV's conformal residuals. But nothing in the
project ever **validated a VaR number**. Interval coverage is checked for the
two-sided 95% cone, which is a different question from "does the 5% left tail
break exactly 5% of the time, and are the breaks independent?" A VaR model can
have perfect unconditional coverage and still be useless because all its
exceptions arrive in one cluster — precisely the failure that matters, since
clustered breaches are what actually bankrupt a position.

So the deliverable here is two-sided:

* **Estimators** — parametric (variance-covariance), historical simulation,
  Cornish-Fisher (skew/kurtosis-adjusted), Filtered Historical Simulation, and
  an optional generative sampler.
* **Backtests** — Kupiec's proportion-of-failures test (unconditional coverage),
  Christoffersen's independence test (are breaches clustered?), and their joint
  conditional-coverage test, plus a rolling out-of-sample harness and a
  method comparison table.

Sign convention
---------------
Every VaR function returns ``var_return``: the **return threshold**, so it is
**negative** for a loss (e.g. ``-0.031`` = "we expect to lose more than 3.1% on
5% of days"). This matches ``risk_metrics.var5_pct``, which is also a negative
fraction, so the two are directly comparable. ``var_loss`` is the same figure as
a positive magnitude for display.

Dependencies: numpy / pandas only. The optional generative sampler imports
TensorFlow lazily and degrades to a block bootstrap when it is unavailable or
fails to train — it can never break a caller.
"""

import math

import numpy as np
import pandas as pd

# RiskMetrics decay for the EWMA variance filter. 0.94 is the standard daily
# value; it puts a ~1-month effective half-life on the variance estimate.
EWMA_LAMBDA = 0.94
_N_SIMS = 20000
_EPS = 1e-12


# ==========================================
# Small distribution helpers (no SciPy)
# ==========================================
def _norm_ppf(p):
    """
    Standard-normal inverse CDF via ``statistics.NormalDist``.

    Kept local so this module has no SciPy dependency (``backtest.py`` already
    treats SciPy as optional, and the rest of the project follows suit).
    """
    from statistics import NormalDist
    return float(NormalDist().inv_cdf(min(max(p, 1e-12), 1 - 1e-12)))


def _norm_pdf(x):
    """Standard-normal density."""
    return float(math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi))


def _norm_cdf(x):
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _tail_es(log_samples, q):
    """
    Expected shortfall as a SIMPLE return, from log-return samples.

    ``es_return`` is reported on the same scale as ``var_return`` and as the
    realised returns the ES backtest scores it against: simple returns. For
    the VaR that is free, because a quantile commutes with ``expm1``. A MEAN
    does not. By Jensen's inequality ``expm1(mean(x)) <= mean(expm1(x))``, so
    converting the mean log tail overstates the average loss, more so the more
    dispersed the tail is (Sadr, *Mathematical Techniques in Finance*, A.2.4).
    The average of the tail's simple returns is the quantity the label claims.
    """
    tail = log_samples[log_samples <= q]
    return float(np.mean(np.expm1(tail))) if len(tail) else float(math.expm1(q))


def _gaussian_es_gross(mu, sigma, alpha):
    """E[exp(X) | X <= q_alpha] for X ~ N(mu, sigma^2): the lognormal tail mean."""
    z = _norm_ppf(alpha)
    return math.exp(mu + 0.5 * sigma * sigma) * _norm_cdf(z - sigma) / alpha


def _chi2_sf(x, df):
    """
    Upper-tail probability of a chi-squared variate, for df in {1, 2}.

    Closed forms only — those are the two degrees of freedom the Kupiec and
    Christoffersen tests need, so a general incomplete-gamma implementation
    would be dead weight.

    * df = 1: ``P(X > x) = erfc(sqrt(x / 2))``
    * df = 2: ``P(X > x) = exp(-x / 2)``
    """
    if not np.isfinite(x) or x < 0:
        return float('nan')
    if df == 1:
        return float(math.erfc(math.sqrt(x / 2.0)))
    if df == 2:
        return float(math.exp(-x / 2.0))
    raise ValueError("only df=1 and df=2 are supported")


# ==========================================
# Volatility filter
# ==========================================
def ewma_volatility(returns, lam=EWMA_LAMBDA, warmup=20):
    """
    One-step-ahead EWMA conditional volatility for each bar.

    ``sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t-1]^2``, so ``sigma[t]``
    uses only bars strictly before ``t`` — it is a genuine forecast, not a
    contemporaneous fit, which is what makes the standardized residuals below
    usable for out-of-sample work.

    Parameters
    ----------
    returns : array-like
        Per-bar log returns.
    lam : float, optional
        Decay factor. Default 0.94 (RiskMetrics daily).
    warmup : int, optional
        Bars used to seed the variance from a sample estimate. Default 20.

    Returns
    -------
    np.ndarray
        Conditional sigma per bar, same length as ``returns``, plus the
        next-bar forecast appended is *not* included (use
        :func:`ewma_next_sigma` for that).
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return np.array([])
    seed = float(np.var(r[:min(warmup, n)])) if n > 1 else float(r[0] ** 2)
    sigma2 = np.empty(n, dtype=np.float64)
    v = max(seed, _EPS)
    for t in range(n):
        sigma2[t] = v
        v = lam * v + (1.0 - lam) * r[t] * r[t]
    return np.sqrt(np.maximum(sigma2, _EPS))


def ewma_next_sigma(returns, lam=EWMA_LAMBDA, warmup=20):
    """Conditional sigma forecast for the bar *after* the last observation."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return float('nan')
    sig = ewma_volatility(r, lam=lam, warmup=warmup)
    v = sig[-1] ** 2
    v = lam * v + (1.0 - lam) * r[-1] ** 2
    return float(math.sqrt(max(v, _EPS)))


# ==========================================
# 1. Variance-covariance (parametric normal)
# ==========================================
def var_variance_covariance(returns, alpha=0.05, horizon=1, ewma=True):
    """
    Parametric VaR under a normal return distribution.

    The book's first method and the industry's oldest: estimate mu and sigma,
    then read the alpha-quantile off a Gaussian. Its **documented drawback** is
    the whole reason the later methods exist — equity returns are leptokurtic
    and left-skewed, so a Gaussian systematically *understates* the tail. At
    alpha = 1% the understatement is typically severe.

    Multi-step scaling uses the square-root-of-time rule
    (``mu*h``, ``sigma*sqrt(h)``), which is exact only for i.i.d. returns — a
    second drawback, since volatility clusters.

    Parameters
    ----------
    returns : array-like
        Per-bar log returns.
    alpha : float, optional
        Tail probability. Default 0.05.
    horizon : int, optional
        Steps ahead. Default 1.
    ewma : bool, optional
        Use the EWMA conditional sigma (responsive to the current regime)
        instead of the unconditional sample sigma. Default True.

    Returns
    -------
    dict
        {'method', 'var_return', 'var_loss', 'es_return', 'sigma', 'mu'}.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return {'method': 'variance-covariance', 'var_return': float('nan'),
                'var_loss': float('nan'), 'es_return': float('nan'),
                'sigma': float('nan'), 'mu': float('nan')}

    mu = float(np.mean(r))
    sigma = ewma_next_sigma(r) if ewma else float(np.std(r, ddof=1))

    mu_h = mu * horizon
    sigma_h = sigma * math.sqrt(horizon)
    z = _norm_ppf(alpha)

    var_log = mu_h + z * sigma_h
    # Gaussian ES of the LOG return is mu - sigma * phi(z)/alpha, but es_return
    # is a simple return, so take the lognormal tail mean instead (see _tail_es).
    es_simple = _gaussian_es_gross(mu_h, sigma_h, alpha) - 1.0

    return {
        'method': 'variance-covariance',
        'var_return': float(math.expm1(var_log)),
        'var_loss': float(-math.expm1(var_log)),
        'es_return': float(es_simple),
        'sigma': float(sigma_h),
        'mu': float(mu_h),
    }


# ==========================================
# 2. Historical simulation
# ==========================================
def var_historical_simulation(returns, alpha=0.05, horizon=1, overlapping=True):
    """
    Non-parametric VaR: the empirical alpha-quantile of realised returns.

    The book's second method. It makes no distributional assumption, so it keeps
    the fat tails — but its **drawbacks** are equally well known:

    * It is **backward-looking and unconditional**: every observation in the
      window gets the same weight, so a calm estimate persists into a crisis and
      a crisis estimate persists into the recovery. It cannot respond to today's
      volatility at all.
    * The tail is estimated from very few points (5% of 250 bars = 12
      observations), so the number is noisy and moves in jumps as extremes enter
      and leave the window.
    * Multi-step estimates need overlapping windows, which makes the effective
      sample far smaller than it looks.

    Filtered Historical Simulation (below) fixes the first drawback while
    keeping the empirical tail.

    Parameters
    ----------
    returns : array-like
        Per-bar log returns.
    alpha : float, optional
        Tail probability. Default 0.05.
    horizon : int, optional
        Steps ahead; > 1 aggregates with a rolling sum. Default 1.
    overlapping : bool, optional
        Use overlapping h-bar windows (more points, serially dependent) rather
        than disjoint blocks. Default True.

    Returns
    -------
    dict
        {'method', 'var_return', 'var_loss', 'es_return', 'n_obs'}.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    nan = {'method': 'historical-simulation', 'var_return': float('nan'),
           'var_loss': float('nan'), 'es_return': float('nan'), 'n_obs': 0}
    if len(r) < 10:
        return nan

    if horizon <= 1:
        agg = r
    elif overlapping:
        agg = np.convolve(r, np.ones(horizon), mode='valid')
    else:
        k = len(r) // horizon
        if k < 5:
            return nan
        agg = r[-k * horizon:].reshape(k, horizon).sum(axis=1)

    if len(agg) < 10:
        return nan

    q = float(np.quantile(agg, alpha))

    return {
        'method': 'historical-simulation',
        'var_return': float(math.expm1(q)),
        'var_loss': float(-math.expm1(q)),
        'es_return': _tail_es(agg, q),
        'n_obs': int(len(agg)),
    }


# ==========================================
# 3. Cornish-Fisher (skew/kurtosis-adjusted parametric)
# ==========================================
def var_cornish_fisher(returns, alpha=0.05, horizon=1, ewma=True):
    """
    Parametric VaR with a Cornish-Fisher expansion for skew and excess kurtosis.

    The cheapest honest repair of the variance-covariance method: keep the
    conditional sigma, but shift the Gaussian quantile using the sample third
    and fourth moments::

        z_cf = z + (z^2 - 1) S / 6 + (z^3 - 3z) K / 24
               - (2 z^3 - 5 z) S^2 / 36

    where ``S`` is skewness and ``K`` excess kurtosis.

    **Read the sign of each term before trusting this.** The kurtosis term is
    proportional to ``z^3 - 3z``, which is *positive* for ``|z| < sqrt(3) ~ 1.732``
    — and the 5% quantile sits at ``z = -1.645``, inside that range. So at
    ``alpha = 0.05`` a large excess kurtosis makes the adjusted quantile **less**
    negative, i.e. it *reduces* the reserve, which is the opposite of what "fat
    tails" intuitively suggests. The fat-tail correction only widens the estimate
    further out in the tail (``alpha <~ 0.025``, where ``z^3 - 3z`` turns
    negative). On a long, highly-leptokurtic history this is not a subtlety: it
    can drive the 5% exception rate to double its nominal level.

    That is why the expansion is included as the documented "one-line improvement
    over normal" **reference point and cautionary case**, not as a recommended
    estimator. It is asymptotic in the moments, so it degrades exactly where a
    risk model needs to be reliable; ``var_filtered_historical_simulation`` is the
    method to reach for instead. ``backtest_var`` will show this directly — see
    the Cornish-Fisher row of ``compare_var_methods``.

    Returns
    -------
    dict
        {'method', 'var_return', 'var_loss', 'es_return', 'skew', 'excess_kurtosis'}.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return {'method': 'cornish-fisher', 'var_return': float('nan'),
                'var_loss': float('nan'), 'es_return': float('nan'),
                'skew': float('nan'), 'excess_kurtosis': float('nan')}

    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= _EPS:
        return {'method': 'cornish-fisher', 'var_return': 0.0, 'var_loss': 0.0,
                'es_return': 0.0, 'skew': 0.0, 'excess_kurtosis': 0.0}
    z_std = (r - mu) / sd
    S = float(np.mean(z_std ** 3))
    K = float(np.mean(z_std ** 4) - 3.0)

    z = _norm_ppf(alpha)
    z_cf = (z
            + (z ** 2 - 1.0) * S / 6.0
            + (z ** 3 - 3.0 * z) * K / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * S ** 2 / 36.0)

    sigma = ewma_next_sigma(r) if ewma else sd
    mu_h = mu * horizon
    sigma_h = sigma * math.sqrt(horizon)
    var_log = mu_h + z_cf * sigma_h
    # No closed-form ES under Cornish-Fisher; use the Gaussian ES shape scaled by
    # the ratio of the adjusted to the Gaussian quantile (a documented approximation).
    ratio = z_cf / z if abs(z) > _EPS else 1.0
    es_log = mu_h - sigma_h * _norm_pdf(z) / alpha * ratio
    # The same log-to-simple (Jensen) correction the Gaussian of this sigma
    # needs, applied to the scaled shape; see _tail_es.
    gauss_log = mu_h - sigma_h * _norm_pdf(z) / alpha
    jensen = _gaussian_es_gross(mu_h, sigma_h, alpha) / math.exp(gauss_log)

    return {
        'method': 'cornish-fisher',
        'var_return': float(math.expm1(var_log)),
        'var_loss': float(-math.expm1(var_log)),
        'es_return': float(math.exp(es_log) * jensen - 1.0),
        'skew': S,
        'excess_kurtosis': K,
    }


# ==========================================
# 4. Filtered Historical Simulation
# ==========================================
def var_filtered_historical_simulation(returns, alpha=0.05, horizon=1,
                                       lam=EWMA_LAMBDA, n_sims=_N_SIMS,
                                       seed=42):
    """
    Filtered Historical Simulation (Barone-Adesi, Giannopoulos & Vosper).

    The estimator that fixes historical simulation's fatal flaw while keeping its
    virtue. Three steps:

    1. **Filter.** Divide each return by its one-step-ahead EWMA conditional
       volatility, giving standardized residuals ``z_t = r_t / sigma_t``. These
       are approximately i.i.d. — the volatility clustering has been removed —
       but they retain the *shape* of the return distribution: the skew and the
       fat tails.
    2. **Resample.** Bootstrap from the ``z`` pool and rescale by the *current*
       volatility forecast, propagating the EWMA recursion forward so simulated
       shocks feed the next step's variance. This is what makes the estimate
       **conditional**: after a volatile week the whole simulated distribution
       widens, immediately, without waiting for the historical window to turn
       over.
    3. **Read the quantile** off the simulated ``horizon``-step cumulative
       returns.

    The result is a conditional, distribution-free, path-dependent VaR — the
    honest workhorse the book's GAN chapter is reaching for, achieved with a
    filter and a bootstrap instead of an adversarial network that needs far more
    data than a single stock's history provides.

    Parameters
    ----------
    returns : array-like
        Per-bar log returns.
    alpha : float, optional
        Tail probability. Default 0.05.
    horizon : int, optional
        Steps ahead. Default 1.
    lam : float, optional
        EWMA decay. Default 0.94.
    n_sims : int, optional
        Simulated paths. Default 20000.
    seed : int, optional
        RNG seed for reproducibility. Default 42.

    Returns
    -------
    dict
        {'method', 'var_return', 'var_loss', 'es_return', 'sigma_forecast',
         'samples'} — ``samples`` is the simulated cumulative log-return array,
        reusable for any other tail statistic.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    nan = {'method': 'filtered-historical-simulation', 'var_return': float('nan'),
           'var_loss': float('nan'), 'es_return': float('nan'),
           'sigma_forecast': float('nan'), 'samples': np.array([])}
    if len(r) < 60:
        return nan

    sig = ewma_volatility(r, lam=lam)
    z = r / np.maximum(sig, _EPS)
    z = z[np.isfinite(z)]
    if len(z) < 40:
        return nan
    # Rescale the pool to unit variance so the filter's seeding bias does not
    # inflate or deflate the simulated dispersion.
    z = z / max(float(np.std(z, ddof=1)), _EPS)

    sigma0 = ewma_next_sigma(r, lam=lam)
    rng = np.random.default_rng(seed)

    cum = np.zeros(n_sims, dtype=np.float64)
    var_state = np.full(n_sims, sigma0 ** 2, dtype=np.float64)
    for _ in range(int(horizon)):
        draws = rng.choice(z, size=n_sims, replace=True)
        step = np.sqrt(var_state) * draws
        cum += step
        var_state = lam * var_state + (1.0 - lam) * step ** 2

    q = float(np.quantile(cum, alpha))

    return {
        'method': 'filtered-historical-simulation',
        'var_return': float(math.expm1(q)),
        'var_loss': float(-math.expm1(q)),
        'es_return': _tail_es(cum, q),
        'sigma_forecast': float(sigma0),
        'samples': cum,
    }


# ==========================================
# 5. Generative sampler (the book's Ch. 9 technique)
# ==========================================
def generative_return_samples(returns, horizon=1, n_sims=_N_SIMS, seed=42,
                              block_mean=5, prefer_gan=False, gan_epochs=400):
    """
    Draw simulated ``horizon``-step cumulative returns from a generative model.

    Chapter 9 trains a **GAN** to learn the return distribution and reads VaR off
    generated samples. That idea is sound but, applied to a single stock, it runs
    into a hard data limit that is worth stating plainly rather than papering
    over: a GAN on ~1000 daily returns is fitting a distribution from a sample
    that is small even for a histogram. In practice it mode-collapses or memorises,
    and its tail — the only region VaR cares about — is where it is least
    trustworthy. That is a bad trade for a risk number.

    So the default generator here is a **stationary block bootstrap** (Politis &
    Romano): resample contiguous blocks of geometric length so each draw keeps
    the time structure *inside* a block, not just the marginal distribution.
    It needs no training, cannot mode-collapse, and is reproducible.

    How much of the structure it keeps was measured later (README §57,
    ``synthetic_check``): at the default mean block of 5 it keeps about a
    fifth of NSE's volatility clustering and less of the fat tails of 20-day
    sums (median ratios 0.20 and 0.12 over 5 seeded paths per name); a mean
    block of 40 keeps ~0.6 and ~0.85. The default is left at 5 because the test that
    matters for this function, multi-day VaR coverage, did not separate the
    two (1% 5-day exception rate 1.59% vs 1.36%, identical at 10 days and at
    5%, on 10 names). The optional GAN keeps none of it: it maps independent
    noise to returns one day at a time.

    ``prefer_gan=True`` opts into a compact 1-D GAN over standardized windows
    for comparison. It imports TensorFlow lazily and silently falls back to the
    block bootstrap on any failure, so enabling it can never break a caller —
    but treat its tail with suspicion and check it with
    :func:`backtest_var` before relying on it.

    Parameters
    ----------
    returns : array-like
        Per-bar log returns.
    horizon : int, optional
        Steps to aggregate per sample. Default 1.
    n_sims : int, optional
        Number of samples to draw. Default 20000.
    seed : int, optional
        RNG seed. Default 42.
    block_mean : int, optional
        Mean block length for the stationary bootstrap. Default 5.
    prefer_gan : bool, optional
        Try the GAN generator first. Default False.
    gan_epochs : int, optional
        Training epochs for the GAN attempt. Default 400.

    Returns
    -------
    (np.ndarray, str)
        ``(samples, generator_used)`` where ``generator_used`` is
        ``'gan'`` or ``'block-bootstrap'``.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 60:
        return np.array([]), 'insufficient-data'

    if prefer_gan:
        try:
            samples = _gan_samples(r, horizon, n_sims, seed, gan_epochs)
            if samples is not None and len(samples) == n_sims:
                return samples, 'gan'
        except Exception as exc:
            print(f"var_models: GAN generator failed ({exc}); "
                  "falling back to the block bootstrap.")

    rng = np.random.default_rng(seed)
    n = len(r)
    p = 1.0 / max(block_mean, 1)

    # Stationary bootstrap: start at a random index, continue to the next index
    # with probability 1-p, otherwise jump to a fresh random index. Wrapping
    # keeps the series stationary (no edge under-sampling).
    out = np.empty(n_sims, dtype=np.float64)
    idx = rng.integers(0, n, size=n_sims)
    total = np.zeros(n_sims, dtype=np.float64)
    for _ in range(int(horizon)):
        total += r[idx]
        jump = rng.random(n_sims) < p
        idx = np.where(jump, rng.integers(0, n, size=n_sims), (idx + 1) % n)
    out[:] = total
    return out, 'block-bootstrap'


def _gan_samples(returns, horizon, n_sims, seed, epochs):
    """
    Train a small 1-D GAN on standardized returns and sample from it.

    Architecture is intentionally tiny (two 32-unit hidden layers each side) with
    label smoothing on the discriminator — the standard minimum needed to keep a
    GAN from collapsing on a few hundred samples. Returns None on any problem so
    the caller can fall back.
    """
    try:
        import tensorflow as tf
    except Exception:
        return None

    from reproducibility import set_all_seeds
    set_all_seeds(seed)

    r = np.asarray(returns, dtype=np.float64)
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    if sd <= _EPS:
        return None
    real = ((r - mu) / sd).reshape(-1, 1).astype('float32')

    latent = 8
    gen = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    disc = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    bce = tf.keras.losses.BinaryCrossentropy()
    g_opt = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
    batch = min(128, len(real))

    for _ in range(int(epochs)):
        real_batch = real[np.random.randint(0, len(real), batch)]
        noise = tf.random.normal([batch, latent])
        fake = gen(noise, training=False)
        with tf.GradientTape() as tape:
            d_loss = (bce(tf.ones([batch, 1]) * 0.9, disc(real_batch, training=True))
                      + bce(tf.zeros([batch, 1]), disc(fake, training=True)))
        d_opt.apply_gradients(zip(tape.gradient(d_loss, disc.trainable_variables),
                                  disc.trainable_variables))
        noise = tf.random.normal([batch, latent])
        with tf.GradientTape() as tape:
            g_loss = bce(tf.ones([batch, 1]),
                         disc(gen(noise, training=True), training=False))
        g_opt.apply_gradients(zip(tape.gradient(g_loss, gen.trainable_variables),
                                  gen.trainable_variables))

    noise = tf.random.normal([n_sims * int(horizon), latent])
    gen_std = np.asarray(gen(noise, training=False)).reshape(n_sims, int(horizon))
    return (gen_std * sd + mu).sum(axis=1).astype(np.float64)


def var_generative(returns, alpha=0.05, horizon=1, n_sims=_N_SIMS, seed=42,
                   prefer_gan=False):
    """
    VaR / ES read off :func:`generative_return_samples`.

    Returns the same dict shape as the other estimators, plus ``generator`` so a
    comparison table records which sampler actually ran.
    """
    samples, gen = generative_return_samples(
        returns, horizon=horizon, n_sims=n_sims, seed=seed, prefer_gan=prefer_gan)
    if len(samples) == 0:
        return {'method': 'generative', 'generator': gen,
                'var_return': float('nan'), 'var_loss': float('nan'),
                'es_return': float('nan'), 'samples': samples}
    q = float(np.quantile(samples, alpha))
    return {
        'method': f'generative ({gen})',
        'generator': gen,
        'var_return': float(math.expm1(q)),
        'var_loss': float(-math.expm1(q)),
        'es_return': _tail_es(samples, q),
        'samples': samples,
    }


# ==========================================
# 6. VaR backtests (book: "Benchmarking results")
# ==========================================
def kupiec_pof_test(exceptions, alpha=0.05):
    """
    Kupiec proportion-of-failures test of **unconditional** VaR coverage.

    Asks the first question any VaR number must answer: did the loss exceed the
    threshold about ``alpha`` of the time? Under H0 the exception count is
    Binomial(n, alpha), and the likelihood ratio::

        LR_pof = -2 ln[ (1-alpha)^(n-x) alpha^x
                        / ((1 - x/n)^(n-x) (x/n)^x ) ]

    is asymptotically chi-squared with 1 degree of freedom. A small p-value means
    the VaR is mis-calibrated: too many exceptions (understating risk — the
    dangerous direction) or too few (overstating it, which is merely expensive).

    Parameters
    ----------
    exceptions : array-like of bool/int
        1 where the realised loss breached the VaR threshold, else 0.
    alpha : float, optional
        The VaR's nominal tail probability. Default 0.05.

    Returns
    -------
    dict
        {'n', 'exceptions', 'observed_rate', 'expected_rate', 'lr_stat',
         'p_value', 'reject_5pct', 'direction'} — ``direction`` is
        'understates risk' / 'overstates risk' / 'ok'.
    """
    e = np.asarray(exceptions, dtype=np.float64)
    e = e[np.isfinite(e)]
    n = len(e)
    x = int(np.sum(e > 0))
    out = {'n': n, 'exceptions': x, 'observed_rate': float('nan'),
           'expected_rate': float(alpha), 'lr_stat': float('nan'),
           'p_value': float('nan'), 'reject_5pct': False, 'direction': 'n/a'}
    if n < 10:
        return out

    rate = x / n
    out['observed_rate'] = float(rate)
    out['direction'] = ('understates risk' if rate > alpha
                        else 'overstates risk' if rate < alpha else 'ok')

    # Degenerate corners: the log-likelihood of the observed rate is 0 there.
    if x == 0:
        lr = -2.0 * (n * math.log(1.0 - alpha))
    elif x == n:
        lr = -2.0 * (n * math.log(alpha))
    else:
        ll_null = (n - x) * math.log(1.0 - alpha) + x * math.log(alpha)
        ll_obs = (n - x) * math.log(1.0 - rate) + x * math.log(rate)
        lr = -2.0 * (ll_null - ll_obs)

    out['lr_stat'] = float(lr)
    out['p_value'] = _chi2_sf(lr, 1)
    out['reject_5pct'] = bool(out['p_value'] < 0.05)
    return out


def christoffersen_independence_test(exceptions):
    """
    Christoffersen independence test: are VaR breaches **clustered**?

    The question Kupiec cannot answer. A model can breach exactly 5% of the time
    and still be worthless if all its breaches arrive consecutively, because
    that is the pattern that wipes out a position. Under H0 the exception
    indicator is a first-order Markov chain with ``pi_01 = pi_11`` (the
    probability of a breach does not depend on whether yesterday breached), and::

        LR_ind = -2 ln[ (1-pi)^(n00+n10) pi^(n01+n11)
                        / ((1-pi01)^n00 pi01^n01 (1-pi11)^n10 pi11^n11) ]

    is asymptotically chi-squared with 1 df. Rejection means the breaches are
    serially dependent — the signature of a VaR model that ignores volatility
    clustering (exactly the historical-simulation drawback).

    Returns
    -------
    dict
        {'n00', 'n01', 'n10', 'n11', 'lr_stat', 'p_value', 'reject_5pct',
         'clustering'} — ``clustering`` is True when ``pi_11 > pi_01``.
    """
    e = (np.asarray(exceptions, dtype=np.float64) > 0).astype(int)
    out = {'n00': 0, 'n01': 0, 'n10': 0, 'n11': 0, 'lr_stat': float('nan'),
           'p_value': float('nan'), 'reject_5pct': False, 'clustering': False}
    if len(e) < 12:
        return out

    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    out.update({'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11})

    total = n00 + n01 + n10 + n11
    if total == 0:
        return out
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    out['clustering'] = bool(pi11 > pi01)

    # A degenerate transition table (no breaches, or no 1->* transitions) leaves
    # the test undefined; report the counts and stop rather than emit a fake p.
    if pi <= 0 or pi >= 1 or (n10 + n11) == 0:
        return out

    def _ll(k0, k1, p):
        if p <= 0 or p >= 1:
            return 0.0 if (k1 == 0 or p > 0) and (k0 == 0 or p < 1) else float('-inf')
        return k0 * math.log(1.0 - p) + k1 * math.log(p)

    ll_null = _ll(n00 + n10, n01 + n11, pi)
    ll_alt = _ll(n00, n01, pi01) + _ll(n10, n11, pi11)
    if not (np.isfinite(ll_null) and np.isfinite(ll_alt)):
        return out

    lr = -2.0 * (ll_null - ll_alt)
    out['lr_stat'] = float(max(lr, 0.0))
    out['p_value'] = _chi2_sf(out['lr_stat'], 1)
    out['reject_5pct'] = bool(out['p_value'] < 0.05)
    return out


def christoffersen_conditional_coverage(exceptions, alpha=0.05):
    """
    Joint conditional-coverage test: correct *rate* **and** independence.

    ``LR_cc = LR_pof + LR_ind`` is asymptotically chi-squared with 2 df. This is
    the summary verdict — a VaR model passes only if it breaches at the right
    frequency *and* its breaches are not clustered.

    Returns
    -------
    dict
        {'lr_stat', 'p_value', 'reject_5pct', 'pof', 'independence'} where the
        last two are the component test results.
    """
    pof = kupiec_pof_test(exceptions, alpha=alpha)
    ind = christoffersen_independence_test(exceptions)
    lr_p, lr_i = pof.get('lr_stat'), ind.get('lr_stat')
    if not (np.isfinite(lr_p or np.nan) and np.isfinite(lr_i or np.nan)):
        return {'lr_stat': float('nan'), 'p_value': float('nan'),
                'reject_5pct': False, 'pof': pof, 'independence': ind}
    lr = float(lr_p + lr_i)
    return {'lr_stat': lr, 'p_value': _chi2_sf(lr, 2),
            'reject_5pct': bool(_chi2_sf(lr, 2) < 0.05),
            'pof': pof, 'independence': ind}


# ==========================================
# 7. Rolling out-of-sample harness + comparison
# ==========================================
#: Estimators available to :func:`backtest_var` / :func:`compare_var_methods`.
VAR_ESTIMATORS = {
    'variance-covariance': lambda r, a, h: var_variance_covariance(r, a, h),
    'historical': lambda r, a, h: var_historical_simulation(r, a, h),
    'cornish-fisher': lambda r, a, h: var_cornish_fisher(r, a, h),
    'fhs': lambda r, a, h: var_filtered_historical_simulation(r, a, h, n_sims=4000),
    'generative': lambda r, a, h: var_generative(r, a, h, n_sims=4000),
}


def returns_from_prices(prices):
    """Log returns from a positive price array (non-positive values dropped)."""
    p = np.asarray(prices, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    return np.diff(np.log(p)) if len(p) > 1 else np.array([])


def backtest_var(returns, method='fhs', alpha=0.05, horizon=1, window=250,
                 step=1, estimator=None):
    """
    Rolling out-of-sample VaR backtest with the full test battery.

    At each step the estimator sees only ``returns[:t]`` (a genuine expanding-
    then-rolling window), forecasts the ``horizon``-step VaR, and is scored
    against the realised ``horizon``-step return. Exceptions are then run through
    Kupiec, Christoffersen and their joint test.

    Two reported extras matter as much as the p-values:

    * **average VaR magnitude** — a model can pass coverage by simply being
      enormous; the width tells you what that safety cost.
    * **mean exceedance beyond VaR** (a positive magnitude) — how far past the
      threshold the loss went when it breached, which is the empirical analogue
      of expected shortfall and distinguishes "a few small breaks" from "one
      catastrophic one".

    A note on the independence test's power: with thousands of evaluations it will
    reject on essentially any real equity series spanning multiple crises, because
    volatility clustering is real and the test is consistent. On large samples,
    read the *relative* LR statistics across methods rather than the pass/fail
    label — a conditional method should have a materially smaller LR than an
    unconditional one even when both formally reject.

    Parameters
    ----------
    returns : array-like
        Full per-bar log-return history.
    method : str, optional
        Key into :data:`VAR_ESTIMATORS`. Default 'fhs'.
    alpha : float, optional
        Tail probability. Default 0.05.
    horizon : int, optional
        VaR horizon in bars. Default 1.
    window : int, optional
        Trailing window handed to the estimator. Default 250.
    step : int, optional
        Bars between evaluations. Use ``step >= horizon`` to keep the exception
        sequence non-overlapping — required for the independence test to mean
        what it says. Default 1.
    estimator : callable, optional
        Custom ``fn(returns, alpha, horizon) -> dict`` overriding ``method``.

    Returns
    -------
    dict
        {'method', 'alpha', 'horizon', 'n_eval', 'exception_rate',
         'avg_var_loss', 'mean_exceedance', 'kupiec', 'independence',
         'conditional_coverage', 'records'} where ``records`` is a DataFrame of
         per-step (var_return, realised, exception).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    fn = estimator or VAR_ESTIMATORS.get(method)
    if fn is None:
        raise ValueError(f"unknown VaR method '{method}'; "
                         f"choose from {sorted(VAR_ESTIMATORS)}")

    rows = []
    t = window
    while t + horizon <= len(r):
        est = fn(r[max(0, t - window):t], alpha, horizon)
        var_ret = est.get('var_return', float('nan'))
        if np.isfinite(var_ret):
            realised = float(math.expm1(np.sum(r[t:t + horizon])))
            rows.append({'t': int(t), 'var_return': float(var_ret),
                         'realised_return': realised,
                         'exception': int(realised < var_ret)})
        t += max(int(step), 1)

    records = pd.DataFrame(rows)
    if records.empty:
        return {'method': method, 'alpha': alpha, 'horizon': horizon,
                'n_eval': 0, 'exception_rate': float('nan'),
                'avg_var_loss': float('nan'), 'mean_exceedance': float('nan'),
                'kupiec': {}, 'independence': {}, 'conditional_coverage': {},
                'records': records}

    exc = records['exception'].values
    breached = records[records['exception'] == 1]
    # Reported as a POSITIVE magnitude: how much further than the threshold the
    # loss went. A breach means realised < var_return (both negative), so the
    # gap is `var_return - realised`.
    mean_exceed = (float(np.mean(breached['var_return'].values
                                 - breached['realised_return'].values))
                   if len(breached) else 0.0)

    return {
        'method': method,
        'alpha': float(alpha),
        'horizon': int(horizon),
        'n_eval': int(len(records)),
        'exception_rate': float(np.mean(exc)),
        'avg_var_loss': float(np.mean(-records['var_return'].values)),
        'mean_exceedance': mean_exceed,
        'kupiec': kupiec_pof_test(exc, alpha),
        'independence': christoffersen_independence_test(exc),
        'conditional_coverage': christoffersen_conditional_coverage(exc, alpha),
        'records': records,
    }


def compare_var_methods(returns, methods=None, alpha=0.05, horizon=1,
                        window=250, step=None):
    """
    Benchmark every VaR estimator on the same history (the book's Ch. 9 finale).

    Runs :func:`backtest_var` per method and returns a single ranked table, so
    the classical methods' drawbacks show up as numbers rather than as claims.
    The expected pattern on equity data: variance-covariance under-reserves
    (exception rate well above ``alpha``, Kupiec rejects), historical simulation
    gets the rate roughly right but **fails independence** (clustered breaches,
    because it is unconditional), and FHS passes both while being no wider than
    it needs to be.

    ``step`` defaults to ``horizon`` so the exception sequence is
    non-overlapping — without that, multi-step breaches are mechanically
    autocorrelated and the independence test rejects for the wrong reason.

    Returns
    -------
    pd.DataFrame
        One row per method, sorted so the best-calibrated (highest
        conditional-coverage p-value) comes first.
    """
    methods = list(VAR_ESTIMATORS) if methods is None else list(methods)
    step = horizon if step is None else step

    rows = []
    for m in methods:
        try:
            res = backtest_var(returns, method=m, alpha=alpha, horizon=horizon,
                               window=window, step=step)
        except Exception as exc:
            print(f"compare_var_methods: '{m}' failed ({exc})")
            continue
        if not res['n_eval']:
            continue
        rows.append({
            'method': m,
            'n_eval': res['n_eval'],
            'exception_rate': round(res['exception_rate'], 4),
            'expected_rate': alpha,
            'avg_var_loss': round(res['avg_var_loss'], 4),
            'mean_exceedance': round(res['mean_exceedance'], 4),
            'kupiec_p': round(res['kupiec'].get('p_value', float('nan')), 4),
            'independence_p': round(
                res['independence'].get('p_value', float('nan')), 4),
            'cc_p': round(
                res['conditional_coverage'].get('p_value', float('nan')), 4),
            'clustered': res['independence'].get('clustering'),
            'verdict': res['kupiec'].get('direction'),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values('cc_p', ascending=False,
                              na_position='last').reset_index(drop=True)
    return out


# ==========================================
# 8. Expected-Shortfall backtesting (Acerbi-Szekely)
# ==========================================
def acerbi_szekely_z2(records_realised, records_var, records_es, alpha=0.05):
    """
    Acerbi-Szekely **Z2** test of Expected Shortfall, conditional on exceedances.

    Why this exists: every estimator in this module reports ``es_return``
    alongside ``var_return``, and until now nothing validated it. That is exactly
    the criticism this module was written to answer for VaR — reporting a risk
    number without ever checking it — so leaving ES unchecked would have repeated
    the mistake one level down.

    ES is famously **not elicitable**, so it has no scoring function that can be
    minimised the way a quantile loss pins down VaR. Acerbi & Szekely (2014)
    sidestep that with a test conditional on the breaches actually observed::

        Z2 = 1 - sum_t ( X_t * 1{X_t < VaR_t} ) / (N * alpha * mean ES)

    Under a correctly specified model ``E[Z2] = 0``, and the sign carries the
    whole message:

    * ``Z2 < 0`` -> realised tail losses were **worse** than the ES predicted, so
      the model **under-estimates severity**. This is the dangerous direction.
    * ``Z2 > 0`` -> the tail was milder than forecast; the model is conservative.

    Significance comes from a **bootstrap under the null**: the observed
    exceedance severities are resampled with a Binomial(n, alpha) breach count to
    build the distribution of Z2 a correctly-calibrated model would produce, so
    the p-value needs no distributional assumption about returns. It is
    one-sided — only under-estimation is a risk failure.

    Parameters
    ----------
    records_realised, records_var, records_es : array-like
        Aligned per-evaluation realised returns, VaR thresholds and ES estimates,
        all as returns (losses negative — the sign convention used throughout
        this module).
    alpha : float, optional
        Tail probability the VaR/ES were computed at. Default 0.05.

    Returns
    -------
    dict
        {'z2', 'n', 'n_exceptions', 'mean_realised_tail', 'mean_predicted_es',
        'severity_ratio', 'p_value', 'reject_5pct', 'direction'} where
        ``severity_ratio`` is realised tail loss / predicted ES (> 1 means the
        model under-reserved).
    """
    x = np.asarray(records_realised, dtype=np.float64)
    v = np.asarray(records_var, dtype=np.float64)
    e = np.asarray(records_es, dtype=np.float64)
    n = min(len(x), len(v), len(e))
    out = {'z2': float('nan'), 'n': int(n), 'n_exceptions': 0,
           'mean_realised_tail': float('nan'), 'mean_predicted_es': float('nan'),
           'severity_ratio': float('nan'), 'p_value': float('nan'),
           'reject_5pct': False, 'direction': 'n/a'}
    if n < 20:
        return out

    x, v, e = x[:n], v[:n], e[:n]
    ok = np.isfinite(x) & np.isfinite(v) & np.isfinite(e) & (e < 0)
    if int(ok.sum()) < 20:
        return out
    x, v, e = x[ok], v[ok], e[ok]
    n = len(x)
    out['n'] = int(n)

    breach = x < v
    out['n_exceptions'] = int(breach.sum())
    if out['n_exceptions'] == 0:
        # No breaches at all means ES is untestable here, not perfect. Say so
        # rather than returning a flattering zero.
        out['direction'] = 'no exceptions - ES untestable'
        return out

    ratios = x[breach] / e[breach]
    z2 = float(1.0 - float(np.sum(ratios)) / (n * alpha))
    out['z2'] = z2
    out['mean_realised_tail'] = float(np.mean(x[breach]))
    out['mean_predicted_es'] = float(np.mean(e[breach]))
    denom = float(np.mean(e[breach]))
    out['severity_ratio'] = (float(np.mean(x[breach]) / denom)
                             if denom != 0 else float('nan'))
    out['direction'] = ('under-estimates severity' if z2 < 0
                        else 'conservative' if z2 > 0 else 'ok')

    # Bootstrap under H0. The subtle part: the null must describe a CORRECTLY
    # calibrated model, so the resampled severities have to have mean 1
    # (realised tail loss = predicted ES). Drawing them straight from the
    # observed pool would bake the model's own mis-calibration into H0 — the
    # null then centres on the observed failure and the test can never reject on
    # severity, only on breach frequency, and with the sign backwards. So the
    # pool is re-centred to unit mean, which imposes H0 while preserving the
    # observed dispersion and shape. Breach count is Binomial(n, alpha), the
    # correct null frequency, so the test still detects too-many/too-few breaches.
    rng = np.random.default_rng(42)
    mean_ratio = float(np.mean(ratios))
    pool_h0 = ratios / mean_ratio if mean_ratio > 0 else ratios
    sims = np.empty(2000, dtype=np.float64)
    for i in range(len(sims)):
        k = int(rng.binomial(n, alpha))
        if k == 0:
            sims[i] = 1.0
            continue
        draw = rng.choice(pool_h0, size=k, replace=True)
        sims[i] = 1.0 - float(np.sum(draw)) / (n * alpha)
    # One-sided: only a more negative Z2 (worse-than-promised tail) is a failure.
    out['p_value'] = float(np.mean(sims <= z2))
    out['reject_5pct'] = bool(out['p_value'] < 0.05)
    return out


def backtest_es(returns, method='fhs', alpha=0.05, horizon=1, window=250,
                step=None, estimator=None):
    """
    Rolling out-of-sample **Expected Shortfall** backtest.

    Mirrors :func:`backtest_var` — each estimate sees only past bars — but keeps
    the per-step ES alongside the VaR and scores the pair with
    :func:`acerbi_szekely_z2`. Run it whenever an ES number is going to be shown
    to anyone: passing the VaR test says the breach *frequency* is right and says
    nothing about whether the breaches were as bad as promised.

    ``step`` defaults to ``horizon`` so the evaluations are non-overlapping.

    Returns
    -------
    dict
        The :func:`acerbi_szekely_z2` result plus 'method', 'alpha', 'horizon'
        and the per-step 'records' DataFrame.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    fn = estimator or VAR_ESTIMATORS.get(method)
    if fn is None:
        raise ValueError("unknown VaR method '%s'; choose from %s"
                         % (method, sorted(VAR_ESTIMATORS)))
    step = horizon if step is None else step

    rows = []
    t = window
    while t + horizon <= len(r):
        est = fn(r[max(0, t - window):t], alpha, horizon)
        vr = est.get('var_return', float('nan'))
        es = est.get('es_return', float('nan'))
        if np.isfinite(vr) and np.isfinite(es):
            realised = float(math.expm1(float(np.sum(r[t:t + horizon]))))
            rows.append({'t': int(t), 'var_return': float(vr),
                         'es_return': float(es), 'realised_return': realised,
                         'exception': int(realised < vr)})
        t += max(int(step), 1)

    records = pd.DataFrame(rows)
    if records.empty:
        return {'method': method, 'alpha': float(alpha), 'horizon': int(horizon),
                'n': 0, 'records': records}

    res = acerbi_szekely_z2(records['realised_return'].values,
                            records['var_return'].values,
                            records['es_return'].values, alpha=alpha)
    res.update({'method': method, 'alpha': float(alpha),
                'horizon': int(horizon), 'records': records})
    return res


def compare_es_methods(returns, methods=None, alpha=0.05, horizon=1,
                       window=250, step=None):
    """
    Benchmark every estimator's **Expected Shortfall**, ranked by |Z2|.

    The ES counterpart of :func:`compare_var_methods`. Read the two tables
    together: a method can pass Kupiec (right breach *frequency*) while failing
    here (breaches far *worse* than promised), and that combination is precisely
    the one that quietly under-reserves a portfolio.

    Returns
    -------
    pd.DataFrame
        One row per method, best-calibrated (smallest |Z2|) first.
    """
    methods = list(VAR_ESTIMATORS) if methods is None else list(methods)
    rows = []
    for m in methods:
        try:
            res = backtest_es(returns, method=m, alpha=alpha, horizon=horizon,
                              window=window, step=step)
        except Exception as exc:
            print("compare_es_methods: '%s' failed (%s)" % (m, exc))
            continue
        if not res.get('n') or not res.get('n_exceptions'):
            continue
        rows.append({
            'method': m,
            'n_eval': res['n'],
            'exceptions': res['n_exceptions'],
            'realised_tail_%': round(res['mean_realised_tail'] * 100, 3),
            'predicted_ES_%': round(res['mean_predicted_es'] * 100, 3),
            'severity_ratio': round(res['severity_ratio'], 3),
            'z2': round(res['z2'], 4),
            'p_value': round(res['p_value'], 4),
            'verdict': res['direction'],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out['z2'].abs().sort_values().index)
        out = out.reset_index(drop=True)
    return out
```
