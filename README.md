- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Announced ex-dividend dates inside a forecast horizon.

Mladjenovic, *Stock Investing For Dummies* (7th ed.), Ch. 6, "Using News about
Dividends": on the ex-dividend date the stock "trades at a slightly lower price
to reflect the amount of the dividend". Every forecast in this app is on the
**adjusted** (total-return) basis, so it does not show that drop. The quoted
price does. When an announced ex-date falls inside the horizon, a user
comparing the forecast with the quote a few days later sees a "miss" the size
of the dividend that is not a forecast error at all.

Measured on 2,171 NSE cash dividends, 2018-2024 (``run_dividend_dates.py``):
the price falls by 81-94% of the dividend (median drop ratio, market-adjusted)
for yields of 0.5% and above, and by the full dividend at 5%+. 42% of
dividends are at least a tenth of a 5-day 95% cone's half-width, 15% at least
a quarter. Large enough to say so.

The ex-dates come from NSE's corporate-action feed through
``bhavcopy.corporate_actions``, which lists announced actions ahead of their
ex-date. Every failure path returns ``None``: the notice is advisory and must
never be able to stop a forecast.
"""

import pandas as pd

#: Typical drop on the ex-date as a share of the dividend, and the measured
#: range of bucket medians (yields >= 0.5%), from run_dividend_dates.py. The
#: top was once given as 1.00, the 5%+ band's YIELD-WEIGHTED figure; its median
#: is 0.94 (README section 60).
DROP_RATIO = 0.85
DROP_RANGE = (0.81, 0.94)


def nse_symbol(ticker):
    """'INFY.NS' -> 'INFY'. None for anything not on the NSE (BSE codes differ)."""
    t = str(ticker or "").strip()
    return t[:-3].upper() if t.upper().endswith(".NS") else None


def upcoming_dividends(symbol, start, end, fetch_fn=None):
    """
    Cash dividends for ``symbol`` with an ex-date in ``(start, end]``.

    Returns a DataFrame with ``ex_date`` and ``dividend`` (per share, rupees),
    or an empty one. ``fetch_fn`` is ``bhavcopy.corporate_actions``' injectable
    nselib-shaped fetcher, for offline tests.
    """
    import bhavcopy as bc
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    acts = bc.corporate_actions(s, e, fetch_fn=fetch_fn, chunk_months=3, snapshot=False)
    if acts is None or acts.empty:
        return pd.DataFrame(columns=["ex_date", "dividend"])
    a = acts[(acts["symbol"].astype(str).str.upper() == str(symbol).upper())
             & (acts["dividend"] > 0) & (acts["ex_date"] > s) & (acts["ex_date"] <= e)]
    return a[["ex_date", "dividend"]].sort_values("ex_date").reset_index(drop=True)


def dividend_notice(ticker, last_price, forecast_dates, fetch_fn=None):
    """
    The announced dividends going ex inside the forecast window, or ``None``.

    Parameters
    ----------
    ticker : str
        yfinance symbol, e.g. 'INFY.NS'.
    last_price : float
        Last observed price (the forecast anchor).
    forecast_dates : sequence of datetime-like
        The forecast's own dates; the window is (first - 1 day, last].

    Returns
    -------
    dict or None
        ``{'events': [{'ex_date', 'dividend', 'yield', 'expected_drop'}],
        'total_dividend', 'total_yield', 'message'}``.
    """
    try:
        sym = nse_symbol(ticker)
        if not sym or last_price is None or not (float(last_price) > 0):
            return None
        dts = pd.to_datetime(pd.Series(list(forecast_dates)))
        if dts.empty:
            return None
        start = dts.min().normalize() - pd.Timedelta(days=1)
        divs = upcoming_dividends(sym, start, dts.max(), fetch_fn=fetch_fn)
        if divs.empty:
            return None
        px = float(last_price)
        events = [{"ex_date": pd.Timestamp(r.ex_date).date().isoformat(),
                   "dividend": float(r.dividend), "yield": float(r.dividend) / px,
                   "expected_drop": DROP_RATIO * float(r.dividend)}
                  for r in divs.itertuples()]
        tot = sum(ev["dividend"] for ev in events)
        first = events[0]
        msg = (f"Ex-dividend inside this horizon: Rs {first['dividend']:,.2f} per share "
               f"goes ex on {first['ex_date']} ({first['yield']:.1%} of the price)"
               + (f", plus {len(events) - 1} more" if len(events) > 1 else "") + ". "
               f"On NSE the price falls by {DROP_RANGE[0]:.0%}-{DROP_RANGE[1]:.0%} of the "
               f"dividend that day, so expect the QUOTED price to drop about "
               f"Rs {DROP_RATIO * tot:,.2f}. The forecasts and bands are total-return "
               "(adjusted) prices and do not show that drop; it is not a forecast miss.")
        return {"events": events, "total_dividend": tot, "total_yield": tot / px,
                "message": msg}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Listing and dividend history, for the company panel (README section 81)
# ---------------------------------------------------------------------------

LISTING_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def listing_info(symbol, fetch_fn=None):
    """
    An NSE equity's listing date, face value, ISIN and series, or ``None``.

    From NSE's equity listing file (``LISTING_URL``), the one the company list
    is built from. The IPO issue price is not in it, so it is not shown.
    ``fetch_fn()`` returns that file as a DataFrame, for offline tests.
    """
    try:
        df = fetch_fn() if fetch_fn else pd.read_csv(LISTING_URL)
        df = df.rename(columns=lambda c: str(c).strip().upper())
        row = df[df["SYMBOL"].astype(str).str.strip().str.upper() == str(symbol).strip().upper()]
        if row.empty:
            return None
        r = row.iloc[0]
        listed = pd.to_datetime(str(r["DATE OF LISTING"]).strip(), format="%d-%b-%Y", errors="coerce")
        return {"symbol": str(r["SYMBOL"]).strip(), "name": str(r["NAME OF COMPANY"]).strip(),
                "listed": None if pd.isna(listed) else listed.date().isoformat(),
                "face_value": float(r["FACE VALUE"]), "isin": str(r["ISIN NUMBER"]).strip(),
                "series": str(r["SERIES"]).strip()}
    except Exception:
        return None


def dividend_history(symbol, actions):
    """
    One company's cash dividends from an all-company corporate-action frame.

    ``actions`` is ``bhavcopy.corporate_actions`` output (``symbol, ex_date,
    factor, dividend, subject``), fetched once and shared by every company.
    Each dividend also gets ``adj_dividend``: the amount per share of today,
    divided by every split or bonus that went ex after it, so a yield measured
    against today's price is not inflated by a later split.
    """
    cols = ["ex_date", "dividend", "adj_dividend", "subject"]
    if actions is None or actions.empty:
        return pd.DataFrame(columns=cols)
    a = actions[actions["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
    a["ex_date"] = pd.to_datetime(a["ex_date"])
    splits = a[a["factor"] != 1.0][["ex_date", "factor"]]
    d = a[a["dividend"] > 0].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    later = [float(splits.loc[splits["ex_date"] > x, "factor"].prod()) for x in d["ex_date"]]
    d["adj_dividend"] = d["dividend"].to_numpy() / pd.Series(later).to_numpy()
    return d[cols].sort_values("ex_date", ascending=False).reset_index(drop=True)


def dividend_summary(divs, last_price, today=None):
    """
    The trailing-12-month dividend and yield, how many in five years, and what is announced.

    Returns ``None`` without dividends or a price. ``ttm_yield`` uses the
    split-adjusted amounts against ``last_price``.
    """
    if divs is None or divs.empty or not last_price or not (float(last_price) > 0):
        return None
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    past = divs[divs["ex_date"] <= today]
    ttm = past[past["ex_date"] > today - pd.Timedelta(days=365)]
    five = past[past["ex_date"] > today - pd.DateOffset(years=5)]
    upcoming = divs[divs["ex_date"] > today].sort_values("ex_date")
    return {"ttm_dividend": float(ttm["adj_dividend"].sum()),
            "ttm_yield": float(ttm["adj_dividend"].sum()) / float(last_price),
            "n_5y": int(len(five)),
            "n_past": int(len(past)),
            "last": (None if past.empty else
                     {"ex_date": past["ex_date"].iloc[0].date().isoformat(),
                      "dividend": float(past["dividend"].iloc[0])}),
            "upcoming": [{"ex_date": r.ex_date.date().isoformat(), "dividend": float(r.dividend)}
                         for r in upcoming.itertuples()]}


# ---------------------------------------------------------------------------
# Results dates (Benklifa, *Think Like an Option Trader*, Ch. 5: a scheduled
# event is a "known unknown" -- its date is public, its outcome is not)
# ---------------------------------------------------------------------------

def _is_results(purpose, desc):
    text = f"{purpose} {desc}".lower()
    return "result" in text


def results_calendar(start, end, fetch_fn=None, chunk_days=60):
    """
    Board meetings to approve financial results between ``start`` and ``end``.

    NSE's event calendar (``nselib.capital_market.event_calendar_for_equity``),
    fetched in chunks. SEBI's listing rules require a company to announce a
    results meeting at least two working days ahead, so a date in this calendar
    is public before the event; most are announced one to three weeks ahead.

    Returns
    -------
    pd.DataFrame
        ``symbol, date`` (one row per symbol and meeting date), sorted.
    """
    if fetch_fn is None:
        from nselib import capital_market as cm
        fetch_fn = cm.event_calendar_for_equity
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    while s <= e:
        stop = min(s + pd.Timedelta(days=chunk_days - 1), e)
        df = fetch_fn(s.strftime("%d-%m-%Y"), stop.strftime("%d-%m-%Y"))
        if df is not None and len(df):
            frames.append(df)
        s = stop + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["symbol", "date"])
    df = pd.concat(frames, ignore_index=True)
    df = df[[_is_results(p, d) for p, d in zip(df["purpose"].astype(str),
                                                df["bm_desc"].astype(str))]]
    out = pd.DataFrame({"symbol": df["symbol"].astype(str).str.upper(),
                        "date": pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")})
    return out.dropna().drop_duplicates().sort_values(["symbol", "date"]).reset_index(drop=True)


def upcoming_results(ticker, start, end, fetch_fn=None):
    """Results meeting dates for one NSE ticker in [start, end], or [] (advisory)."""
    sym = nse_symbol(ticker)
    if not sym:
        return []
    try:
        cal = results_calendar(start, end, fetch_fn=fetch_fn)
    except Exception:
        return []
    return [pd.Timestamp(d) for d in cal.loc[cal["symbol"] == sym, "date"]]
```
