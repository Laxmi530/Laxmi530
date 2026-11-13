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
import pmdarima as pm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

from model_utils import (
    normalize_interval,
    prepare_log_returns,
    reset_keras_seeds,
    temporal_train_val_split,
    default_keras_callbacks,
    generate_future_dates,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

_LOOK_BACK = 30


# ==========================================
# 1. ARIMA Component (Linear)
# ==========================================
def _fit_arima(returns):
    """
    Fit Auto-ARIMA on log returns to capture the linear component.

    Uses BIC for model selection, d=0 (returns are stationary), and
    caps the total ARMA order at 10 to prevent overfitting.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Log return series.

    Returns
    -------
    pmdarima.ARIMA
        Fitted ARIMA model.
    """
    print("Fitting ARIMA (linear component)...")
    arima_model = pm.auto_arima(
        returns,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        max_order=10,
        d=0,
        seasonal=False,
        information_criterion='bic',
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False,
    )
    print(f"Best ARIMA Order: {arima_model.order}")
    return arima_model


# ==========================================
# 3. LSTM Component (Non-linear Residuals)
# ==========================================
def _build_residual_sequences(residuals, look_back=_LOOK_BACK):
    """
    Create scaled look-back sequences from ARIMA residuals for the LSTM.

    The LSTM learns the non-linear patterns that ARIMA cannot capture.
    Residuals are scaled to [0, 1] before sequencing.

    Parameters
    ----------
    residuals : np.ndarray
        1-D array of ARIMA residuals (observed - ARIMA fitted).
    look_back : int, optional
        Number of past residual values per input sequence. Default 30.

    Returns
    -------
    X : np.ndarray
        (n_samples, look_back, 1) input sequences.
    y : np.ndarray
        (n_samples,) target values (next residual, scaled).
    scaler : MinMaxScaler
        Fitted scaler for inverse transforms.
    """
    resid = residuals.reshape(-1, 1).astype(np.float64)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(resid).flatten()

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i])
        y.append(scaled[i])

    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)

    return X, y, scaler


def _train_residual_lstm(X, y, output_dim=1):
    """
    Train a small LSTM to model the ARIMA residual series.

    Architecture: two LSTM layers (32, 16 units) with dropout,
    trained with Huber loss, Adam with gradient clipping, early
    stopping, and LR scheduling on a temporal validation split.

    Parameters
    ----------
    X : np.ndarray
        (n_samples, look_back, 1) input sequences.
    y : np.ndarray
        (n_samples,) target values, or (n_samples, output_dim) for the direct
        strategy.
    output_dim : int, optional
        Outputs of the last layer: 1 (shipped, one step) or the horizon.

    Returns
    -------
    tf.keras.Model
        Trained LSTM model for residual prediction.
    """
    reset_keras_seeds()

    X_train, X_val, y_train, y_val = temporal_train_val_split(X, y)

    model = Sequential([
        LSTM(32, return_sequences=True, recurrent_dropout=0.1,
             input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(16, return_sequences=False, recurrent_dropout=0.1),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(output_dim),
    ])

    model.compile(
        loss='huber',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    )

    print("Training LSTM on ARIMA residuals...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=default_keras_callbacks(),
        shuffle=True,
        verbose=0,
    )
    return model


# ==========================================
# 4. Recursive Hybrid Prediction
# ==========================================
def _hybrid_paths(arima_preds, residual_histories, last_prices, lstm_model,
                  resid_scaler):
    """
    Recursive hybrid price paths from several origins at once.

    At each step h:
    1. ARIMA gives the next log return (linear component). Fed its own
       predictions with fixed parameters, that is exactly the ARIMA's
       multi-step forecast, so ``arima_preds`` holds it for every step.
    2. The LSTM predicts the next residual correction (non-linear) from the
       last ``_LOOK_BACK`` residuals; its own correction extends the buffer.
    3. The combined return = ARIMA mean + LSTM residual correction, and the
       price is compounded.

    Every origin moves one step per LSTM call, so calibration costs one
    batched call per step instead of one per origin and step. The same
    function makes the calibration paths and the live forecast (one origin),
    so the band measures the procedure that produces the forecast. (It used
    to call pmdarima's ``update`` with each prediction, which re-estimates the
    parameters on the model's own output: 75x slower, and a drift of about 1%
    of a day's move on an AR(2). README section 69.)

    Parameters
    ----------
    arima_preds : array (n_origins, horizon)
        Each origin's ARIMA multi-step forecast of log returns.
    residual_histories : list of arrays
        Each origin's residuals up to the origin (at least ``_LOOK_BACK``).
    last_prices : array (n_origins,)
        The price at each origin.

    Returns
    -------
    np.ndarray (n_origins, horizon)
        Predicted prices.
    """
    arima_preds = np.atleast_2d(np.asarray(arima_preds, dtype=float))
    n_orig, horizon = arima_preds.shape
    buf = np.stack([np.asarray(r, dtype=float)[-_LOOK_BACK:] for r in residual_histories])
    price = np.asarray(last_prices, dtype=float).copy()
    out = np.empty((n_orig, horizon))
    for h in range(horizon):
        scaled = resid_scaler.transform(buf.reshape(-1, 1)).reshape(n_orig, _LOOK_BACK, 1)
        lstm_scaled = lstm_model.predict(scaled.astype(np.float32), verbose=0,
                                         batch_size=max(32, n_orig)).reshape(-1)
        correction = resid_scaler.inverse_transform(lstm_scaled.reshape(-1, 1)).reshape(-1)
        price = price * np.exp(arima_preds[:, h] + correction)
        out[:, h] = price
        buf = np.concatenate([buf[:, 1:], correction[:, None]], axis=1)
    return out


def _build_direct_sequences(residuals, horizon, look_back=_LOOK_BACK):
    """
    Look-back windows of residuals and, as targets, the cumulative residual
    over each of the next ``horizon`` steps, for the direct (multiple-output)
    strategy (Ben Taieb, Bontempi, Atiya and Sorjamaa, Expert Systems with
    Applications 2012). Inputs are scaled to [0, 1] as for the recursive LSTM;
    targets are divided by the residuals' standard deviation, returned as
    ``scale``.
    """
    resid = np.asarray(residuals, dtype=np.float64)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(resid.reshape(-1, 1))
    scaled = scaler.transform(resid.reshape(-1, 1)).ravel()
    scale = float(np.std(resid)) or 1.0
    X, Y = [], []
    for i in range(look_back, len(resid) - horizon + 1):
        X.append(scaled[i - look_back:i])
        Y.append(np.cumsum(resid[i:i + horizon]) / scale)
    return np.array(X).reshape(-1, look_back, 1), np.array(Y), scaler, scale


def _direct_paths(arima_preds, residual_histories, last_prices, lstm_model,
                  resid_scaler, scale):
    """
    Direct hybrid price paths: the ARIMA's cumulative multi-step forecast plus
    the network's cumulative residual correction for every step at once, from
    one call. Nothing the network predicts is fed back to it.
    """
    arima_preds = np.atleast_2d(np.asarray(arima_preds, dtype=float))
    n_orig = arima_preds.shape[0]
    buf = np.stack([np.asarray(r, dtype=float)[-_LOOK_BACK:] for r in residual_histories])
    x = resid_scaler.transform(buf.reshape(-1, 1)).reshape(n_orig, _LOOK_BACK, 1)
    corr = np.asarray(lstm_model.predict(x.astype(np.float32), verbose=0,
                                         batch_size=max(32, n_orig))) * scale
    cum = np.cumsum(arima_preds, axis=1) + corr.reshape(n_orig, -1)
    return np.asarray(last_prices, dtype=float)[:, None] * np.exp(cum)


def _vol_at(returns, t, window=60):
    """Standard deviation of the ``window`` returns before position ``t``."""
    r = np.asarray(returns, dtype=float)[max(0, t - window):t]
    return float(np.std(r, ddof=1)) if len(r) > 2 else float("nan")


def _conditioned(arima_res, new_returns):
    """The ARIMA fitted on the training block, run on later returns with its parameters fixed."""
    if not len(new_returns):
        return arima_res
    return arima_res.append(np.asarray(new_returns, dtype=float), refit=False)


def _residual_history(arima_res, returns_seen):
    """Residuals (observed - one-step fitted) over every return the ARIMA has seen."""
    fitted = np.asarray(arima_res.fittedvalues, dtype=float)
    return np.asarray(returns_seen, dtype=float)[-len(fitted):] - fitted


# ==========================================
# 5. Main Orchestrator
# ==========================================
def run_hybrid_prediction(df, interval, horizon, alpha=0.05, cal_frac=0.2,
                          max_cal_origins=80,
                          strategy="recursive", band="absolute",
                          lstm=True, min_cal_origins=None):
    """
    Main function to run the ARIMA + LSTM hybrid prediction, with
    split-conformal 95% bands.

    Decomposes the forecasting problem into a linear component
    (ARIMA on log returns) and a non-linear component (LSTM on
    ARIMA residuals). This hybrid approach is well-documented in
    the literature as outperforming either model alone, because
    ARIMA captures the autocorrelation structure while the LSTM
    learns complex non-linear residual patterns that ARIMA misses.

    The band is horizon-wise split conformal, as in
    ``model_utils.forecast_nn_conformal`` for the other recursive models
    (README section 69):

    1. ARIMA and the residual LSTM are fitted on the earlier ``1 - cal_frac``
       of the returns only.
    2. From up to ``max_cal_origins`` origins in the recent block, the ARIMA is
       updated with the actual returns up to the origin and the hybrid's own
       recursive path is run; the absolute log error at each step is recorded.
    3. The finite-sample ``1 - alpha`` quantile of those errors is the
       half-width at that step, so the band widens with the horizon and
       carries the model's real multi-step error.
    4. The forecast comes from the same models updated with every return, and
       each step is wrapped in ``price * exp(+-half_width)``.

    As in textbook split conformal (and the other recursive models), the
    networks are trained on about 80% of the window; this costs one fit.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock DataFrame with 'datetime' and 'adj_close' columns.
    interval : str
        '1d' / 'daily' / '1 Day' or '1h' / 'hourly' / '1 Hour'.
    horizon : int
        Number of future steps to predict.
    alpha : float, optional
        Miscoverage; 0.05 gives 95% bands.
    cal_frac : float, optional
        Share of the most recent returns held out to calibrate the band.
    max_cal_origins : int, optional
        Cap on calibration origins (subsampled with a stride) to bound cost.
    strategy : str, optional
        ``'recursive'`` (shipped): one-step residual LSTM fed its own
        corrections. ``'direct'``: one network with ``horizon`` outputs
        predicting the cumulative residual correction at every step.
    band : str, optional
        ``'rw'``: no calibration quantile; the reference's band,
        ``exp(+-1.96 sd60 sqrt(step))`` around the hybrid's own centre
        (README section 96).
        ``'absolute'`` (shipped): the conformal quantile of absolute log
        errors. ``'vol_scaled'``: each error divided by the 60-session return
        volatility at its origin times sqrt(step), the quantile multiplied back
        by today's (normalized conformal scores, Papadopoulos, Gammerman and
        Vovk 2008; Lei et al., JASA 2018).
    lstm : bool, optional
        False: the ARIMA alone, a diagnostic for what the LSTM adds; never
        served.
    min_cal_origins : int, optional
        Lengthen the calibration block so it holds at least this many origins
        (at 90 sessions the shipped 20% of the window holds about 60, whose
        paths overlap almost entirely). The proper-training block shrinks by
        the same amount. None: as shipped. README section 96.

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'predicted_price', 'lower_bound', 'upper_bound',
        'volatility']. With too little data to calibrate, only
        ['datetime', 'predicted_price'].
    """
    interval = normalize_interval(interval)
    print(f"--- Starting Hybrid ARIMA+LSTM Prediction ({interval}) ---")
    if strategy not in ("recursive", "direct"):
        raise ValueError(f"strategy must be 'recursive' or 'direct', not {strategy!r}")
    if band not in ("absolute", "vol_scaled", "rw"):
        raise ValueError(f"band must be 'absolute', 'vol_scaled' or 'rw', not {band!r}")

    max_obs = 750 if interval == 'daily' else 500
    clean_df = prepare_log_returns(df, max_obs=max_obs)

    returns = clean_df['log_return'].values.astype(float)
    prices = clean_df['adj_close'].values.astype(float)
    n = len(returns)
    last_price = float(prices[-1])
    last_date = clean_df.index[-1]
    future_dates = generate_future_dates(last_date, horizon, interval)

    # Temporal split: proper-train block vs the recent calibration block.
    cal_size = max(_LOOK_BACK + horizon + 5, int(round(n * cal_frac)))
    if min_cal_origins:
        cal_size = max(cal_size, horizon + int(min_cal_origins) - 1)
    train_end = n - cal_size
    calibrate = train_end > 2 * _LOOK_BACK + 20
    if not calibrate:
        train_end = n          # too little data: fit on all, no band

    arima_model = _fit_arima(returns[:train_end])
    arima_res = arima_model.arima_res_          # statsmodels results, for fixed-parameter paths
    residuals = _residual_history(arima_res, returns[:train_end])

    if len(residuals) <= _LOOK_BACK + 10:
        print("Insufficient data for LSTM residual model, using ARIMA only")
        arima_fcast = arima_model.predict(n_periods=horizon)
        curr_price, future_data = last_price, []
        for i in range(horizon):
            curr_price *= np.exp(arima_fcast[i])
            future_data.append({'datetime': future_dates[i],
                                'predicted_price': float(curr_price)})
        return pd.DataFrame(future_data)

    if not lstm:
        def paths(preds, hists, lasts):
            return np.asarray(lasts, dtype=float)[:, None] * np.exp(
                np.cumsum(np.atleast_2d(preds), axis=1))
    elif strategy == "direct" and len(residuals) - _LOOK_BACK - horizon + 1 >= 40:
        X_res, y_res, resid_scaler, scale = _build_direct_sequences(residuals, horizon)
        lstm_model = _train_residual_lstm(X_res, y_res, output_dim=horizon)

        def paths(preds, hists, lasts):
            return _direct_paths(preds, hists, lasts, lstm_model, resid_scaler, scale)
    else:
        if strategy == "direct":
            print("Too little history for the direct strategy; using the recursive one")
        X_res, y_res, resid_scaler = _build_residual_sequences(residuals)
        lstm_model = _train_residual_lstm(X_res, y_res)

        def paths(preds, hists, lasts):
            return _hybrid_paths(preds, hists, lasts, lstm_model, resid_scaler)

    half = [None] * horizon
    if calibrate:
        origins = list(range(train_end, n - horizon + 1))
        if len(origins) > max_cal_origins:
            stride = int(np.ceil(len(origins) / max_cal_origins))
            origins = origins[::stride]
        print(f"Calibrating the hybrid's 95% band on {len(origins)} recent origins...")
        preds, hists = [], []
        for t in origins:
            # The forecast made after observing return t-1 targets prices t..t+h-1.
            cond = _conditioned(arima_res, returns[train_end:t])
            preds.append(np.asarray(cond.forecast(horizon), dtype=float))
            hists.append(_residual_history(cond, returns[:t]))
        cal_paths = paths(np.vstack(preds), hists, prices[np.array(origins) - 1])
        actual = np.stack([prices[t:t + horizon] for t in origins])
        errors = np.abs(np.log(actual / cal_paths))
        if band == "vol_scaled":
            root = np.sqrt(np.arange(1, horizon + 1))
            sd_o = np.array([_vol_at(returns, t) for t in origins])
            errors = errors / (sd_o[:, None] * root[None, :])
        for h in range(horizon):
            r = errors[:, h]
            m = len(r)
            if m:
                q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
                half[h] = float(np.quantile(r, q_level))

    print(f"Generating {horizon} hybrid (ARIMA+LSTM) forecasts...")
    live = _conditioned(arima_res, returns[train_end:])
    point = paths(np.asarray(live.forecast(horizon), dtype=float)[None, :],
                  [_residual_history(live, returns)], [last_price])[0]
    if band == "vol_scaled" and calibrate:
        sd_now = _vol_at(returns, n)
        half = [None if q is None else q * sd_now * np.sqrt(i + 1) for i, q in enumerate(half)]
    elif band == "rw":
        sd_now = _vol_at(returns, n)
        half = [1.959963984540054 * sd_now * np.sqrt(i + 1) for i in range(horizon)]

    rows = []
    for i in range(horizon):
        p, q = float(point[i]), half[i]
        if q is None:
            rows.append({'datetime': future_dates[i], 'predicted_price': p})
        else:
            lower, upper = float(p * np.exp(-q)), float(p * np.exp(q))
            rows.append({'datetime': future_dates[i], 'predicted_price': p,
                         'lower_bound': lower, 'upper_bound': upper,
                         'volatility': float((upper - lower) / (2 * 1.96))})

    print("Hybrid ARIMA+LSTM Prediction Complete.")
    return pd.DataFrame(rows)
```
