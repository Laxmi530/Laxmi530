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
from sklearn.metrics import mean_absolute_error
import warnings

from model_utils import (
    normalize_interval, wilder_rsi, generate_future_dates, NSE_BARS_PER_SESSION,
    lgb_threads, xgb_threads,
)
from direct_forecast import (
    horizon_feature_cols, make_horizon_targets, training_frame,
    compute_origin_row, conformal_halfwidth, conformal_calibration,
    conformal_quantile,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

_DAILY_WINDOWS = [5, 10, 20, 50]
_HOURLY_WINDOWS = [6, 12, 24, 48]
_RSI_PERIOD = 14
_BUFFER_SIZE = 80

# Intraday (hourly-only) features capturing NSE session structure. NSE sessions
# are 09:15-15:30, which yfinance returns as 7 contiguous hourly bars (hours
# 9..15), so "N bars back = same bar of an earlier session" is exact regardless
# of holidays.
#
# The default and its rationale live in config.py with every other flag; it is
# re-bound here (rather than read through the module) so the existing A/B tests
# can still monkeypatch ML_model.USE_INTRADAY_FEATURES, and so the functions
# below keep reading a plain module global. Set STOCKAPP_USE_INTRADAY_FEATURES=0
# to A/B it for a run without editing source.
from config import USE_INTRADAY_FEATURES
_NSE_OPEN_HOUR = 9


def _add_intraday_features_vec(df):
    """
    Add leak-free NSE intraday features to a datetime-indexed hourly frame.

    All features use only the current bar and the past: bar position in the
    session, first/last-bar flags, return since the session open, the overnight
    gap (today's open vs the *previous* session's close), and the return vs the
    same bar one session ago. Modifies ``df`` in place.
    """
    hour = df.index.hour
    bar = np.clip(hour - _NSE_OPEN_HOUR, 0, NSE_BARS_PER_SESSION - 1)
    df['bar_in_session'] = bar
    df['is_first_bar'] = (bar == 0).astype(int)
    df['is_last_bar'] = (bar == NSE_BARS_PER_SESSION - 1).astype(int)

    sess = df.index.normalize()
    sess_series = pd.Series(sess, index=df.index)
    # Return since the session's opening bar (0 at the open; past-only otherwise).
    session_open = df.groupby(sess)['adj_close'].transform('first')
    df['ret_since_open'] = np.log(df['adj_close'] / session_open)
    # Overnight gap: today's open vs the PREVIOUS session's close (both known at
    # the open, so it is constant across the session and never peeks ahead).
    daily_first = df.groupby(sess)['adj_close'].first()
    daily_last = df.groupby(sess)['adj_close'].last()
    gap = np.log(daily_first / daily_last.shift(1))
    df['overnight_gap'] = sess_series.map(gap).values
    # Same bar, one session ago (7 contiguous bars back).
    df['same_hour_prev_day_ret'] = np.log(
        df['adj_close'] / df['adj_close'].shift(NSE_BARS_PER_SESSION)
    )


# ==========================================
# 1. Vectorised Feature Engineering (Training)
# ==========================================
def create_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Build scale-invariant, return-based features plus technical indicators.

    Features include multi-horizon log returns, MA ratios, rolling
    volatility at multiple windows, RSI (Wilder smoothing),
    normalised MACD line and histogram, ATR (normalised),
    price range ratio, price z-score (mean-reversion signal), volume
    ratio and change, and calendar variables.  Every feature can also
    be computed from a price/volume buffer in ``_features_from_buffers``,
    keeping training and prediction consistent.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with datetime, open, high, low, close,
        adj_close, volume columns.
    interval : str
        'daily' or 'hourly'; determines rolling window sizes and
        whether the 'hour' calendar feature is included.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with all engineered features and
        a 'target_log_return' column (next-step log return).
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['adj_close'])

    df['target_log_return'] = np.log(
        df['adj_close'] / df['adj_close'].shift(1)
    ).shift(-1)

    for lag in [1, 2, 3, 5, 10]:
        df[f'log_return_{lag}'] = np.log(
            df['adj_close'] / df['adj_close'].shift(lag)
        )

    windows = _DAILY_WINDOWS if interval == 'daily' else _HOURLY_WINDOWS
    ret_1 = np.log(df['adj_close'] / df['adj_close'].shift(1))

    for w in windows:
        ma = df['adj_close'].rolling(w).mean()
        df[f'ma_ratio_{w}'] = df['adj_close'] / ma
        df[f'volatility_{w}'] = ret_1.rolling(w).std()

    df['rsi_14'] = wilder_rsi(df['adj_close'], period=_RSI_PERIOD)

    # No Bollinger-band position: any Bollinger position is a linear function
    # of price_zscore below (the old bb_position was exactly price_zscore / 2,
    # correlation 1.0), so it added no information while doubling that
    # signal's chance of being sampled into a tree's feature subset and
    # splitting its attribution across two names (README §55).

    ema_12 = df['adj_close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['adj_close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_norm'] = macd_line / df['adj_close']
    df['macd_hist_norm'] = (macd_line - macd_signal) / df['adj_close']

    if all(c in df.columns for c in ['high', 'low', 'close']):
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df['atr_norm'] = tr.ewm(span=_RSI_PERIOD, adjust=False).mean() / (
            df['adj_close'].replace(0, 1e-10)
        )
        df['range_pct'] = (df['high'] - df['low']) / (
            df['adj_close'].replace(0, 1e-10)
        )
    else:
        abs_ret = ret_1.abs()
        df['atr_norm'] = abs_ret.rolling(_RSI_PERIOD).mean()
        df['range_pct'] = abs_ret.rolling(5).mean() * 2

    ma_20 = df['adj_close'].rolling(20).mean()
    std_20 = df['adj_close'].rolling(20).std()
    df['price_zscore'] = (df['adj_close'] - ma_20) / (
        std_20.replace(0, 1e-10)
    )

    df['return_accel'] = ret_1.diff()

    vol_ma = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / vol_ma.replace(0, 1)
    df['volume_change'] = df['volume'].pct_change()

    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    if interval == 'hourly':
        df['hour'] = df.index.hour
        if USE_INTRADAY_FEATURES:
            _add_intraday_features_vec(df)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df


# ==========================================
# 2. Single-row feature computation (Prediction)
# ==========================================
def _features_from_buffers(prices, volumes, dt, interval, highs=None, lows=None,
                           closes=None):
    """
    Compute a single feature row from raw price/volume buffers.

    Mirror of ``create_features`` but operates on numpy arrays and
    returns a dict for one timestep.  Uses Wilder's exponential
    smoothing for RSI and pandas EWM for MACD, ensuring consistency
    with the training-time feature definitions.

    ATR and range_pct use the bar's high/low/close when ``highs``, ``lows``
    and ``closes`` are supplied, exactly as ``create_features`` does. Without
    them they fall back to close-to-close proxies, which is only right when
    training had no OHLC either. The forecast origin used to take that
    fallback while the models trained on true range: measured on NSE daily
    bars, the served ``atr_norm`` was ~46% below the trained one and
    ``range_pct`` ~20% above it (README §55), so ``compute_origin_row`` now
    passes the bars.

    Parameters
    ----------
    prices : array-like
        Historical price buffer (most recent at the end, length
        >= ``_BUFFER_SIZE``).
    volumes : array-like
        Historical volume buffer (same length as prices).
    dt : datetime-like
        Timestamp for the current prediction step (used for calendar
        features).
    interval : str
        'daily' or 'hourly'; determines window sizes and whether
        the 'hour' feature is included.
    highs, lows, closes : array-like, optional
        Raw bar high, low and close, aligned with ``prices``. When all three
        are given, ATR and range_pct are computed from true range.

    Returns
    -------
    dict
        Feature name -> value mapping for a single observation.
    """
    features = {}
    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    for lag in [1, 2, 3, 5, 10]:
        if len(prices) > lag and prices[-1 - lag] > 0:
            features[f'log_return_{lag}'] = np.log(
                prices[-1] / prices[-1 - lag]
            )
        else:
            features[f'log_return_{lag}'] = 0.0

    windows = _DAILY_WINDOWS if interval == 'daily' else _HOURLY_WINDOWS
    log_rets = np.diff(np.log(np.maximum(prices, 1e-10)))

    for w in windows:
        if len(prices) >= w:
            features[f'ma_ratio_{w}'] = prices[-1] / np.mean(prices[-w:])
        else:
            features[f'ma_ratio_{w}'] = 1.0

        if len(log_rets) >= w:
            features[f'volatility_{w}'] = float(
                np.std(log_rets[-w:], ddof=1)
            )
        elif len(log_rets) > 1:
            features[f'volatility_{w}'] = float(np.std(log_rets, ddof=1))
        else:
            features[f'volatility_{w}'] = 0.0

    if len(prices) > _RSI_PERIOD:
        s = pd.Series(prices)
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(
            alpha=1.0 / _RSI_PERIOD, min_periods=_RSI_PERIOD, adjust=False,
        ).mean()
        avg_loss = loss.ewm(
            alpha=1.0 / _RSI_PERIOD, min_periods=_RSI_PERIOD, adjust=False,
        ).mean()
        rs_val = avg_gain.iloc[-1] / (
            avg_loss.iloc[-1] if avg_loss.iloc[-1] > 0 else 1e-10
        )
        features['rsi_14'] = 100.0 - (100.0 / (1.0 + rs_val))
    else:
        features['rsi_14'] = 50.0

    if len(prices) >= 26:
        ps = pd.Series(prices)
        ema_12 = ps.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = ps.ewm(span=26, adjust=False).mean().iloc[-1]
        macd_line = ema_12 - ema_26
        macd_signal = (
            ps.ewm(span=12, adjust=False).mean()
            - ps.ewm(span=26, adjust=False).mean()
        ).ewm(span=9, adjust=False).mean().iloc[-1]
        features['macd_norm'] = macd_line / prices[-1] if prices[-1] > 0 else 0.0
        features['macd_hist_norm'] = (
            (macd_line - macd_signal) / prices[-1] if prices[-1] > 0 else 0.0
        )
    else:
        features['macd_norm'] = 0.0
        features['macd_hist_norm'] = 0.0

    have_ohlc = (highs is not None and lows is not None and closes is not None
                 and len(highs) == len(prices) == len(lows) == len(closes)
                 and len(prices) >= 2 and prices[-1] > 0)
    if have_ohlc:
        # Mirror create_features: true range on the raw bars, EWM over the
        # buffer, normalised by the adjusted close.
        h = pd.Series(np.asarray(highs, dtype=np.float64))
        lo = pd.Series(np.asarray(lows, dtype=np.float64))
        prev_c = pd.Series(np.asarray(closes, dtype=np.float64)).shift(1)
        tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()],
                       axis=1).max(axis=1)
        atr = float(tr.ewm(span=_RSI_PERIOD, adjust=False).mean().iloc[-1] / prices[-1])
        rng_pct = float((h.iloc[-1] - lo.iloc[-1]) / prices[-1])
        # A missing high or low on the origin bar would otherwise put a NaN
        # into the one row every forecast is made from; fall back instead.
        if np.isfinite(atr):
            features['atr_norm'] = atr
        if np.isfinite(rng_pct):
            features['range_pct'] = rng_pct

    abs_rets = np.abs(log_rets) if len(log_rets) > 0 else np.array([0.0])
    if 'atr_norm' not in features:
        # Close-to-close proxies: the same fallback create_features uses when
        # the frame has no high/low/close.
        if len(abs_rets) >= _RSI_PERIOD:
            features['atr_norm'] = float(np.mean(abs_rets[-_RSI_PERIOD:]))
        elif len(abs_rets) > 0:
            features['atr_norm'] = float(np.mean(abs_rets))
        else:
            features['atr_norm'] = 0.0

    if 'range_pct' not in features:
        if len(abs_rets) >= 5:
            features['range_pct'] = float(np.mean(abs_rets[-5:]) * 2)
        elif len(abs_rets) > 0:
            features['range_pct'] = float(np.mean(abs_rets) * 2)
        else:
            features['range_pct'] = 0.0

    if len(prices) >= 20:
        ma_20 = np.mean(prices[-20:])
        std_20 = float(np.std(prices[-20:], ddof=1))
        features['price_zscore'] = (
            (prices[-1] - ma_20) / std_20 if std_20 > 0 else 0.0
        )
    else:
        features['price_zscore'] = 0.0

    if len(log_rets) >= 2:
        features['return_accel'] = float(log_rets[-1] - log_rets[-2])
    else:
        features['return_accel'] = 0.0

    if len(volumes) >= 20:
        vol_ma = np.mean(volumes[-20:])
        features['volume_ratio'] = (
            volumes[-1] / vol_ma if vol_ma > 0 else 1.0
        )
    else:
        features['volume_ratio'] = 1.0

    if len(volumes) >= 2 and volumes[-2] > 0:
        features['volume_change'] = volumes[-1] / volumes[-2] - 1
    else:
        features['volume_change'] = 0.0

    features['month'] = dt.month
    features['dayofweek'] = dt.dayofweek
    if interval == 'hourly':
        features['hour'] = dt.hour
        if USE_INTRADAY_FEATURES:
            # Scalar mirror of _add_intraday_features_vec for the origin bar.
            # Contiguous 7-bar sessions => the session open is `bar` bars back and
            # the same bar one session ago is NSE_BARS_PER_SESSION bars back.
            n = len(prices)
            bar = int(min(max(dt.hour - _NSE_OPEN_HOUR, 0),
                          NSE_BARS_PER_SESSION - 1))
            features['bar_in_session'] = bar
            features['is_first_bar'] = int(bar == 0)
            features['is_last_bar'] = int(bar == NSE_BARS_PER_SESSION - 1)
            features['ret_since_open'] = (
                float(np.log(prices[-1] / prices[-1 - bar]))
                if n > bar and prices[-1 - bar] > 0 else 0.0
            )
            features['overnight_gap'] = (
                float(np.log(prices[-1 - bar] / prices[-2 - bar]))
                if n > bar + 1 and prices[-2 - bar] > 0 else 0.0
            )
            features['same_hour_prev_day_ret'] = (
                float(np.log(prices[-1] / prices[-1 - NSE_BARS_PER_SESSION]))
                if n > NSE_BARS_PER_SESSION
                and prices[-1 - NSE_BARS_PER_SESSION] > 0 else 0.0
            )

    return features


# ==========================================
# 3. Model Factory
# ==========================================
def get_model_and_params(model_type: str, min_child_samples=None):
    """
    Create and return a configured gradient-boosting regressor.

    Uses Huber loss (pseudo-Huber for XGBoost, Huber for LightGBM)
    for robustness to fat-tailed return outliers, which are common
    in stock data and can dominate squared-error training.

    Parameters
    ----------
    model_type : str
        'xgboost' or 'lightgbm'.
    min_child_samples : int, optional
        None (shipped): 20 rows per LightGBM leaf, a minimum child weight of 5
        for XGBoost. An int sets both (README section 97).

    Returns
    -------
    tuple of (estimator, str)
        model : configured XGBRegressor or LGBMRegressor instance.
        engine : string identifier ('xgboost' or 'lightgbm').

    Raises
    ------
    ValueError
        If model_type is not 'xgboost' or 'lightgbm'.
    """
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            objective='reg:pseudohubererror',
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5 if min_child_samples is None else min_child_samples,
            random_state=42,
            early_stopping_rounds=50,
            **xgb_threads(),
        )
        return model, 'xgboost'

    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(
            objective='huber',
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20 if min_child_samples is None else min_child_samples,
            random_state=42,
            verbose=-1,
            **lgb_threads(),
        )
        return model, 'lightgbm'

    else:
        raise ValueError("model_type must be 'xgboost' or 'lightgbm'")


# ==========================================
# 4. Unified Training Function
# ==========================================
def train_generic_model(features_df: pd.DataFrame, target_col: str,
                        model_type: str, test_size=0.1, purge=0,
                        min_child_samples=None):
    """
    Train an XGBoost or LightGBM model with temporal train/test split.

    Automatically selects feature columns (excluding raw OHLCV and the
    target), trains with early stopping on a held-out temporal
    validation set, and reports test MAE.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with engineered feature columns and the target column.
    target_col : str
        Name of the target column (e.g., 'target_log_return').
    model_type : str
        'xgboost' or 'lightgbm'.
    test_size : float, optional
        Fraction of data reserved for temporal validation. Default 0.1.
    purge : int, optional
        Rows dropped from the end of the training block, before the
        early-stopping block. A cumulative h-step target overlaps the next h
        rows, so ``purge=h`` keeps the stopping check off labels the trees were
        fit on (purged validation, Lopez de Prado, Advances in Financial
        Machine Learning, 2018, ch. 7). Default 0 (shipped).
    min_child_samples : int, optional
        Passed to :func:`get_model_and_params`.

    Returns
    -------
    tuple of (estimator, list of str)
        model : trained model instance.
        feature_cols : list of feature column names used during training.
    """
    raw_cols = {'open', 'high', 'low', 'close', 'adj_close', 'volume'}
    feature_cols = [
        c for c in features_df.columns
        if c != target_col and c not in raw_cols
    ]

    X = features_df[feature_cols]
    y = features_df[target_col]

    split_idx = int(len(X) * (1 - test_size))
    cut = max(1, split_idx - max(0, int(purge)))
    X_train, X_test = X.iloc[:cut], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:cut], y.iloc[split_idx:]

    print(
        f"Training {model_type.upper()}... "
        f"(Train: {len(X_train)}, Test: {len(X_test)}, "
        f"Features: {len(feature_cols)})"
    )

    model, engine = get_model_and_params(model_type, min_child_samples)

    if engine == 'xgboost':
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ]
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=callbacks,
        )

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"{model_type.upper()} Test MAE: {mae:.6f}")

    return model, feature_cols


# ==========================================
# 5. Main Orchestrator (Direct Multi-Horizon)
# ==========================================
#: LightGBM and XGBoost on daily bars beyond gating.MAX_DIRECTIONAL_SESSIONS:
#: larger leaves and a centre shrunk toward no change. Chosen on 32 forecasts and
#: confirmed on 72 fresh ones before the reserved period, for LightGBM
#: (run_lgbm_band.py, README section 97) and then for XGBoost on its own run of
#: the same test (run_xgb_band.py, README section 101).
LGBM_LONG = dict(leaf_scale=True, shrink=True)
#: The engines that take LGBM_LONG beyond the up/down horizon.
LONG_SWITCH_MODELS = ('lightgbm', 'xgboost')
#: The configuration before section 97, which every other run keeps.
LGBM_LEGACY = dict(purge=False, shrink=False, leaf_scale=False)


def resolve_switches(model_type, interval, horizon, purge=None, shrink=None,
                     leaf_scale=None):
    """
    The switches a run uses. A switch left as None takes the default: LGBM_LONG
    for LightGBM and XGBoost (LONG_SWITCH_MODELS) on daily bars beyond
    gating.MAX_DIRECTIONAL_SESSIONS, otherwise LGBM_LEGACY. An explicit True or
    False is kept.
    """
    from gating import MAX_DIRECTIONAL_SESSIONS
    long = (model_type in LONG_SWITCH_MODELS and normalize_interval(interval) == 'daily'
            and int(horizon) > MAX_DIRECTIONAL_SESSIONS)
    base = dict(LGBM_LEGACY, **(LGBM_LONG if long else {}))
    given = dict(purge=purge, shrink=shrink, leaf_scale=leaf_scale)
    return {k: base[k] if v is None else bool(v) for k, v in given.items()}


def run_ML_prediction(df: pd.DataFrame, interval: str, horizon: int,
                      model_type: str = 'xgboost', purge=None, shrink=None,
                      leaf_scale=None):
    """
    Main function to run XGBoost or LightGBM prediction.

    Uses **direct multi-horizon** forecasting: a separate model is trained for
    each step ``h`` to predict the cumulative log return ``log(P[t+h]/P[t])``,
    and every step is forecast directly from the observed features at the last
    bar. This replaces the previous recursive scheme, which compounded errors
    across the horizon (and started one bar early), and was the weakest
    performer in the walk-forward backtest.

    Orchestrates feature engineering (Wilder RSI, MACD, price z-score,
    multi-scale momentum/volatility), per-horizon model training with Huber
    loss and early stopping, and direct forecasting from the origin row.
    Distribution-free **conformal** prediction intervals are added per horizon
    via a temporal split-conformal calibration, so the forecast now carries
    calibrated 95% bounds whose width grows with horizon. Training data is
    limited to recent observations to focus on the current market regime.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns datetime, open, high, low, close,
        adj_close, volume.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    model_type : str, optional
        'xgboost' (default) or 'lightgbm'.
    purge, shrink, leaf_scale : bool or None, optional
        None (default) takes :func:`resolve_switches`'s default: shrink and
        leaf_scale on for LightGBM on daily bars beyond 13 sessions (README
        section 97), everything off otherwise.
    purge : bool, optional
        True: purge ``h`` rows before each step's early-stopping block
        (see :func:`train_generic_model`).
    shrink : bool, optional
        True: shrink each step's predicted return toward no change by
        beta = clip(sum(p*y)/sum(p*p), 0, 1), the least-squares slope through
        the origin of realised on predicted returns in the calibration block,
        and take the band from the residuals of the shrunk forecast (in the
        spirit of Campbell and Thompson, Review of Financial Studies 2008).
    leaf_scale : bool, optional
        True: at least min(20 h, a fifth of the training rows) rows per leaf,
        because overlapping h-step targets hold about n/h independent outcomes
        (Boudoukh, Israel and Richardson, Financial Analysts Journal 2019).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'predicted_price',
        'lower_bound', 'upper_bound', 'volatility'].
    """
    interval = normalize_interval(interval)
    sw = resolve_switches(model_type, interval, horizon, purge, shrink, leaf_scale)
    purge, shrink, leaf_scale = sw['purge'], sw['shrink'], sw['leaf_scale']

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

    print(f"Training {horizon} direct {model_type.upper()} horizon models...")

    rows = []
    for i, h in enumerate(range(1, horizon + 1)):
        target_name = f'target_h{h}'
        train_df = training_frame(
            features_df, feature_cols, targets[h], target_name,
        )
        fit_kw = {}
        if purge:
            fit_kw['purge'] = h
        if leaf_scale:
            fit_kw['min_child_samples'] = int(max(20, min(20 * h, len(train_df) // 5)))
        model, _ = train_generic_model(train_df, target_name, model_type, **fit_kw)
        cum_return = float(model.predict(origin_X)[0])

        # Conformal half-width (log-return space) from a temporal calibration split.
        def _fit_predict(proper_df, X_cal, _tn=target_name, _mt=model_type, _kw=fit_kw):
            cal_model, _ = train_generic_model(proper_df, _tn, _mt, **_kw)
            return cal_model.predict(X_cal)

        if shrink:
            half = None
            parts = conformal_calibration(
                train_df, feature_cols, target_name, _fit_predict, embargo=h,
            )
            if parts is not None:
                y_cal, p_cal = parts
                den = float(np.sum(p_cal * p_cal))
                beta = float(np.clip(np.sum(p_cal * y_cal) / den, 0.0, 1.0)) if den > 0 else 0.0
                cum_return *= beta
                half = conformal_quantile(y_cal - beta * p_cal)
        else:
            half = conformal_halfwidth(
                train_df, feature_cols, target_name, _fit_predict, embargo=h,
            )
        if half is None:
            half = 1.96 * float(np.std(train_df[target_name].values))

        median = last_price * np.exp(cum_return)
        lower = last_price * np.exp(cum_return - half)
        upper = last_price * np.exp(cum_return + half)
        rows.append({
            'datetime': future_dates[i],
            'predicted_price': float(median),
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'volatility': float((upper - lower) / (2 * 1.96)),
        })

    return pd.DataFrame(rows)
```
