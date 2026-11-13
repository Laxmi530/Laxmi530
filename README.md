- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
The app's forecast charts, as plain functions of DataFrames (no Streamlit), so
they can be tested directly (README section 70).

Two charts:

* :func:`forecast_chart` - one model: recent history (a line, or candles
  adjusted to the same basis as the forecast), a dashed "last close" marker,
  the forecast's median path, its 95% range drawn as a cone that fans out from
  the last close, and the random-walk reference.
* :func:`comparison_chart` - every model at once: each one's median path, the
  median of their paths, and the median of their 95% ranges, so agreement or
  disagreement is visible at a glance.

Both put trading sessions side by side: weekends, NSE holidays and (on hourly
charts) the overnight gap are removed from the axis, so five forecast sessions
sit evenly rather than bunched around a weekend. Daily charts hide the missing
days; hourly charts give every bar its own slot on a category axis (README
section 80).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HISTORY = "#1f77b4"
FORECAST = "#d62728"
BAND = "rgba(214, 39, 40, 0.15)"
NAIVE = "#7f7f7f"
CONSENSUS = "#111111"
#: Distinct colours for up to 20 models' lines; the history blue and the
#: random-walk grey are left out so no model can be mistaken for either.
MODEL_COLOURS = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
                 "#d62728", "#ffbb78", "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2", "#9edae5",
                 "#dbdb8d", "#393b79", "#637939", "#843c39", "#7b4173", "#e7ba52"]

DATE_COLS = ("date", "datetime", "time", "ds")
PRICE_COLS = ("adj close", "adj_close", "close", "price", "y")
PRED_COLS = ("predicted_price", "prediction", "forecast", "yhat", "pred", "close")


def col(df, candidates):
    """The first of ``candidates`` that is a column of ``df`` (case-insensitive)."""
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def is_hourly(interval):
    s = str(interval).lower()
    return "h" in s and "day" not in s


def trading_gaps(dates):
    """Calendar days with no bar between the first and last of ``dates`` (weekends, holidays)."""
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).normalize().unique()
    if len(d) < 2:
        return []
    full = pd.date_range(d.min(), d.max(), freq="D")
    return [x.strftime("%Y-%m-%d") for x in full.difference(d)]


def _trading_axis(fig, dates, interval):
    """Remove non-trading time from the x-axis so sessions sit side by side."""
    if is_hourly(interval):
        _bar_axis(fig)
        return
    gaps = trading_gaps(dates)
    fig.update_xaxes(rangebreaks=[dict(values=gaps)] if gaps else [],
                     rangeslider_visible=False, automargin=True, tickformat="%d-%b-%y")
    fig.update_yaxes(automargin=True)


def _bar_axis(fig, max_ticks=8):
    """
    Hourly charts: one evenly spaced slot per bar, labelled by date and time.

    The axis used to be a date axis with Plotly rangebreaks hiding the night
    (an hour-pattern break, 15:30-09:15) and the weekends (day values). With
    both, Plotly mapped whole sessions to the wrong place: the history line ran
    back across the chart and dates bunched together (README section 80). A
    category axis has no hidden time to get wrong: every bar takes one slot,
    in order, so sessions sit side by side by construction.
    """
    def ts(x):
        return pd.Timestamp(x)

    xs = set()
    for t in fig.data:
        xs.update(ts(x) for x in (t.x if t.x is not None else []))
    for s in fig.layout.shapes:
        xs.update((ts(s.x0), ts(s.x1)))
    for a in fig.layout.annotations:
        if a.x is not None:
            xs.add(ts(a.x))
    xs = sorted(xs)
    fmt = "%d-%b %H:%M"
    if len({x.strftime(fmt) for x in xs}) < len(xs):
        fmt = "%d-%b-%y %H:%M"            # the window crosses a year
    label = {x: x.strftime(fmt) for x in xs}
    for t in fig.data:
        if t.x is not None:
            t.x = [label[ts(x)] for x in t.x]
    for s in fig.layout.shapes:
        s.x0, s.x1 = label[ts(s.x0)], label[ts(s.x1)]
    for a in fig.layout.annotations:
        if a.x is not None:
            a.x = label[ts(a.x)]
    # A tick at the first bar of each session, thinned to about max_ticks.
    starts = [x for i, x in enumerate(xs) if i == 0 or x.normalize() != xs[i - 1].normalize()]
    step = max(1, -(-len(starts) // max_ticks))
    ticks = starts[::step]
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=[label[x] for x in xs],
                     tickmode="array", tickvals=[label[x] for x in ticks],
                     ticktext=[x.strftime("%d-%b %H:%M") for x in ticks],
                     rangeslider_visible=False, automargin=True)
    fig.update_yaxes(automargin=True)


def adjusted_ohlc(hist):
    """
    Open/high/low/close scaled by ``adj_close / close``, so candles sit on the
    same dividend- and split-adjusted basis as the forecast. None when the
    frame has no OHLC.
    """
    o, h, l, c = (col(hist, [k]) for k in ("open", "high", "low", "close"))
    adj = col(hist, ["adj close", "adj_close"])
    if not all((o, h, l, c)):
        return None
    close = pd.to_numeric(hist[c], errors="coerce")
    ratio = (pd.to_numeric(hist[adj], errors="coerce") / close) if adj else 1.0
    ratio = ratio.where(np.isfinite(ratio) & (ratio > 0), 1.0) if adj else 1.0
    return pd.DataFrame({k: pd.to_numeric(hist[v], errors="coerce") * ratio
                         for k, v in (("open", o), ("high", h), ("low", l), ("close", c))})


def _history_trace(sub, date_col, price_col, style, name="History"):
    if style == "candles":
        ohlc = adjusted_ohlc(sub)
        if ohlc is not None and ohlc.notna().all(axis=1).any():
            return go.Candlestick(x=sub[date_col], open=ohlc["open"], high=ohlc["high"],
                                  low=ohlc["low"], close=ohlc["close"], name=name,
                                  increasing_line_color="#2ca02c", decreasing_line_color="#d62728",
                                  showlegend=True)
    return go.Scatter(x=sub[date_col], y=sub[price_col], mode="lines", name=name,
                      line=dict(color=HISTORY, width=2),
                      hovertemplate="Close ₹%{y:,.2f}<extra></extra>")


def _last_close_marker(fig, x):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                  line=dict(color="#9a9a9a", width=1, dash="dash"))
    fig.add_annotation(x=x, y=1, yref="paper", text="last close", showarrow=False,
                       yanchor="bottom", font=dict(size=10, color="#6b6b6b"))


def _layout(fig, title, legend_rows=1):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title="Price (₹)",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        height=460 + 22 * max(0, legend_rows - 1),
        # Legend in a row BELOW the plot: overlaid, it covered the most recent
        # prices on a narrow screen (README section 66).
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.14, yanchor="top"),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
    )


def _history(original_df, lookback):
    d, p = col(original_df, DATE_COLS), col(original_df, PRICE_COLS)
    if not (d and p):
        raise ValueError(f"no date/price columns in the history: {list(original_df.columns)}")
    sub = original_df.tail(min(lookback, len(original_df))).copy()
    sub[d] = pd.to_datetime(sub[d])
    return sub, d, p


def forecast_chart(original_df, prediction_df, company_name, interval, lookback=50, style="line"):
    """
    One model's forecast: history, last-close marker, median path, the 95%
    range as a cone from the last close, and the random-walk reference.

    Raises ValueError when the date or price columns cannot be found.
    """
    sub, dh, ph = _history(original_df, lookback)
    dp, pp = col(prediction_df, DATE_COLS), col(prediction_df, PRED_COLS)
    if not (dp and pp):
        raise ValueError(f"no date/forecast columns in the prediction: {list(prediction_df.columns)}")
    last_x, last_y = sub[dh].iloc[-1], float(sub[ph].iloc[-1])
    fx = [last_x] + list(pd.to_datetime(prediction_df[dp]))
    fy = [last_y] + [float(v) for v in prediction_df[pp]]

    fig = go.Figure()
    fig.add_trace(_history_trace(sub, dh, ph, style))
    lo, hi = col(prediction_df, ["lower_bound", "lower"]), col(prediction_df, ["upper_bound", "upper"])
    if lo and hi:
        # The cone starts at the last close with zero width and fans out: the
        # widening IS the message. It used to start at full width on day one.
        upper = [last_y] + [float(v) for v in prediction_df[hi]]
        lower = [last_y] + [float(v) for v in prediction_df[lo]]
        fig.add_trace(go.Scatter(x=fx, y=upper, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fx, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=BAND, name="95% range", customdata=upper,
                                 hovertemplate="95% range ₹%{y:,.2f} – ₹%{customdata:,.2f}<extra></extra>"))
    # The random walk: the price staying where it is, the no-skill reference.
    fig.add_trace(go.Scatter(x=fx, y=[last_y] * len(fx), mode="lines", name="Random walk (no change)",
                             line=dict(color=NAIVE, width=1.5, dash="dot"),
                             hovertemplate="No change ₹%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers", name="Forecast (median)",
                             line=dict(color=FORECAST, width=2, dash="dash"),
                             marker=dict(symbol="x", size=8),
                             hovertemplate="Forecast ₹%{y:,.2f}<extra></extra>"))
    _last_close_marker(fig, last_x)
    _layout(fig, f"{company_name} · {str(interval).lower()} bars")
    _trading_axis(fig, list(sub[dh]) + fx, interval)
    return fig


def comparison_chart(model_results, company_name, interval, lookback=30, style="line"):
    """
    Every model's median path from the last close, their median path, and the
    median of their 95% ranges. ``model_results`` maps a model's name to
    ``{'history': DataFrame, 'prediction': DataFrame}``. None when no model has
    a usable forecast.
    """
    paths, lowers, uppers = {}, [], []
    hist = None
    for name, data in model_results.items():
        pred = data.get("prediction")
        if hist is None and data.get("history") is not None:
            hist = data["history"]
        if pred is None or not len(pred):
            continue
        dp, pp = col(pred, DATE_COLS), col(pred, PRED_COLS)
        if not (dp and pp):
            continue
        paths[name] = (list(pd.to_datetime(pred[dp])), [float(v) for v in pred[pp]])
        lo, hi = col(pred, ["lower_bound", "lower"]), col(pred, ["upper_bound", "upper"])
        if lo and hi:
            lowers.append([float(v) for v in pred[lo]])
            uppers.append([float(v) for v in pred[hi]])
    if hist is None or not paths:
        return None
    sub, dh, ph = _history(hist, lookback)
    last_x, last_y = sub[dh].iloc[-1], float(sub[ph].iloc[-1])
    horizon = min(len(p[1]) for p in paths.values())
    dates = [last_x] + next(iter(paths.values()))[0][:horizon]

    fig = go.Figure()
    fig.add_trace(_history_trace(sub, dh, ph, style))
    if lowers:
        lo = np.median(np.array([v[:horizon] for v in lowers]), axis=0)
        hi = np.median(np.array([v[:horizon] for v in uppers]), axis=0)
        up = [last_y] + list(hi)
        fig.add_trace(go.Scatter(x=dates, y=up, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dates, y=[last_y] + list(lo), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(120, 120, 120, 0.15)",
                                 name=f"Median 95% range ({len(lowers)} models)", customdata=up,
                                 hovertemplate="Median range ₹%{y:,.2f} – ₹%{customdata:,.2f}<extra></extra>"))
    # One legend entry for all the model lines: one entry each made a 24-row
    # legend on a phone. Hover still names every model and its value.
    for i, (name, (_, ys)) in enumerate(paths.items()):
        fig.add_trace(go.Scatter(x=dates, y=[last_y] + ys[:horizon], mode="lines", name=name,
                                 legendgroup="models", showlegend=(i == 0),
                                 legendgrouptitle=None,
                                 line=dict(color=MODEL_COLOURS[i % len(MODEL_COLOURS)], width=1.3),
                                 opacity=0.85, hovertemplate=f"{name} ₹%{{y:,.2f}}<extra></extra>"))
    first = next(t for t in fig.data if getattr(t, "legendgroup", None) == "models")
    first.name = f"Each model ({len(paths)}, hover for names)"
    first.hovertemplate = f"{next(iter(paths))} ₹%{{y:,.2f}}<extra></extra>"
    med = np.median(np.array([ys[:horizon] for _, ys in paths.values()]), axis=0)
    fig.add_trace(go.Scatter(x=dates, y=[last_y] + list(med), mode="lines+markers",
                             name="Median of the models", line=dict(color=CONSENSUS, width=3),
                             marker=dict(size=6), hovertemplate="Median ₹%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=dates, y=[last_y] * len(dates), mode="lines", name="Random walk (no change)",
                             line=dict(color=NAIVE, width=1.5, dash="dot"),
                             hovertemplate="No change ₹%{y:,.2f}<extra></extra>"))
    _last_close_marker(fig, last_x)
    _layout(fig, f"{company_name} · all {len(paths)} models", legend_rows=2)
    _trading_axis(fig, list(sub[dh]) + dates, interval)
    return fig
```
