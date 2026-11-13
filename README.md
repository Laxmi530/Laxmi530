- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Forecast the fixed watchlist and log it, exactly as the app would serve it.

The forward prediction log (README section 28) grows only as fast as the app
is used. This runs the app's own pipeline (``serving.py``: fetch, every
model, the walk-forward confidence gate, the data-quality and novelty gates)
on a watchlist chosen by a rule before the first forecast (``watchlist.json``),
and writes each forecast to the log with ``source="watchlist"``. Nothing is
selected after the fact: every name, every week, every model is in the record.
README section 78.

Usage::

    python run_watchlist.py              # forecast and log every name
    python run_watchlist.py --dry-run    # run everything, log nothing (timing, checks)
    python run_watchlist.py --only "Infosys Limited"
    python run_watchlist.py --jobs 3     # companies in parallel (default 3)

A company already logged for the same origin (its last bar) is skipped, so a
re-run or a retried scheduled run cannot write the same forecast twice. A daily
bar from a session still open is dropped, so a run during market hours forecasts
from the last close rather than from a half-finished day. Workers only compute;
the log is written by this process alone, one company at a time.
"""

import argparse
import concurrent.futures as cf
import contextlib
import io
import json
import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
WATCHLIST = os.path.join(ROOT, "watchlist.json")
sys.path.insert(0, ROOT)


def load_watchlist(path=WATCHLIST):
    with open(path, encoding="utf-8") as f:
        w = json.load(f)
    for key in ("frozen_on", "companies", "interval", "horizon", "models", "assess"):
        if key not in w:
            raise ValueError(f"{path}: missing '{key}'")
    return w


def already_logged(log_df, company, origin):
    """True when this company's forecast from this origin is in the log."""
    if log_df.empty:
        return False
    m = ((log_df["source"] == "watchlist") & (log_df["ticker"] == company)
         & (log_df["origin_datetime"] == str(origin)))
    return bool(m.any())


#: NSE closes at 15:30 IST; a daily bar dated today is final after this.
SESSION_FINAL = "16:00"


def drop_open_session(hist, now=None):
    """History without today's daily bar while today's session may still be open."""
    import pandas as pd
    now = pd.Timestamp.now(tz="Asia/Kolkata") if now is None else pd.Timestamp(now)
    dt = pd.to_datetime(hist["datetime"])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_localize(None)
    today = (now.tz_localize(None) if now.tzinfo else now).normalize()
    if dt.max().normalize() == today and now.strftime("%H:%M") < SESSION_FINAL:
        return hist[(dt < today).to_numpy()].reset_index(drop=True)
    return hist


def forecast_company(company, w, models, log=True, quiet=True, skip_logged=None):
    """
    Serve one company the way the app's Submit does, and log every model.

    Same order as UI.py: run the models, take the anchor from the history,
    the data-quality gate, the walk-forward confidence gate, the data-quality
    override, then the novelty gate. Returns a dict summary whose ``entries``
    are the ``log_forecast`` calls (written here when ``log`` is True).
    ``skip_logged`` (default: ``log``) skips a company already logged for
    this origin.
    """
    import pandas as pd

    import prediction_log as pl
    import serving
    from config import USE_MARKET_CONTEXT, USE_REGIME_NOVELTY
    from data_quality import assess_data_quality
    from gating import apply_data_quality_gate, apply_horizon_gate, apply_novelty_gate
    from reproducibility import GLOBAL_SEED, set_all_seeds

    interval, horizon = w["interval"], int(w["horizon"])
    hush = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    set_all_seeds(GLOBAL_SEED)
    with hush:
        hist = serving.fetch_history(company, interval, USE_MARKET_CONTEXT)
    if hist is None:
        return {"company": company, "status": "no data"}
    if interval == "1d":
        hist = drop_open_session(hist)
    origin = pd.to_datetime(hist["datetime"]).max()
    if (log if skip_logged is None else skip_logged) and already_logged(pl.read_log(), company, origin):
        return {"company": company, "status": f"already logged for {origin.date()}"}

    results, failed = {}, []
    for name in models:
        try:
            with hush:
                pred = serving.run_model(hist, company, interval, horizon, name)
            if serving.is_usable(pred):
                results[name] = pred
            else:
                failed.append((name, "non-finite price"))
        except Exception as exc:
            failed.append((name, str(exc)[:80]))
    if not results:
        return {"company": company, "status": "no model produced a forecast", "failed": failed}

    anchor = serving.last_price(hist)
    dq = assess_data_quality(hist, interval, horizon)
    conf = {}
    if w["assess"]:
        for name in results:
            try:
                with hush:
                    conf[name] = serving.model_confidence(hist, company, interval, horizon, name)
            except Exception as exc:
                conf[name] = {"bucket": "N/A", "note": str(exc)[:80]}
    conf = apply_data_quality_gate(conf, list(results), dq.mode in ("risk_cone_only", "abstain"))
    novelty = {"available": False, "verdict": "unknown"}
    if USE_REGIME_NOVELTY:
        with hush:
            novelty = serving.regime_novelty(hist, interval)
        conf = apply_novelty_gate(conf, list(results), novelty)
    conf = apply_horizon_gate(conf, list(results), horizon, interval)

    run_id = "wl-" + uuid.uuid4().hex[:9]
    entries = [dict(ticker=company, interval=interval, horizon=horizon, model=name,
                    prediction_df=pred, anchor_price=float(anchor), origin_datetime=origin,
                    confidence_bucket=conf.get(name, {}).get("bucket"),
                    dq_mode=dq.mode, novelty_verdict=novelty.get("verdict"),
                    seed=GLOBAL_SEED, run_id=run_id, source="watchlist")
               for name, pred in results.items()]
    r = {"company": company, "status": "ok", "origin": str(origin.date()),
         "models": len(results), "logged": 0, "failed": failed,
         "dq_mode": dq.mode, "novelty": novelty.get("verdict"), "entries": entries}
    if log:
        write_entries(r)
    return r


def write_entries(r):
    """Write a result's forecasts to the log; sets and returns ``r['logged']``."""
    import prediction_log as pl
    r["logged"] = sum(1 for e in r.get("entries", []) if pl.log_forecast(**e))
    return r["logged"]


def _work(args):
    """A worker process: compute one company, log nothing (the parent writes)."""
    company, w, models, quiet = args
    os.environ.setdefault("STOCKAPP_QUIET", "1")
    try:
        return forecast_company(company, w, models, log=False, quiet=quiet, skip_logged=True)
    except Exception as exc:
        return {"company": company, "status": f"failed: {str(exc)[:120]}"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true", help="run everything, log nothing")
    p.add_argument("--only", help="one company from the watchlist")
    p.add_argument("--verbose", action="store_true", help="show every model's own output")
    p.add_argument("--jobs", type=int, default=3, help="companies in parallel (default 3)")
    p.add_argument("--watchlist", default=WATCHLIST, help="watchlist file (default watchlist.json)")
    a = p.parse_args(argv)

    os.environ.setdefault("STOCKAPP_QUIET", "1")
    w = load_watchlist(a.watchlist)
    companies = w["companies"]
    if a.only:
        if a.only not in companies:
            print(f"'{a.only}' is not on the watchlist")
            return 2
        companies = [a.only]
    import serving
    models = serving.MODEL_NAMES if w["models"] == "all" else list(w["models"])
    unknown = [m for m in models if m not in serving.MODEL_NAMES]
    if unknown:
        print(f"watchlist names unknown models: {unknown}")
        return 2

    print(f"Watchlist frozen {w['frozen_on']}: {len(companies)} companies, {len(models)} models, "
          f"{w['horizon']}-step {w['interval']}" + ("  [dry run: nothing is logged]" if a.dry_run else ""))
    t_all = time.time()
    done = {}

    def report(r):
        if r["status"] == "ok" and not a.dry_run:
            write_entries(r)
        done[r["company"]] = r
        line = f"  {r['company']:<34} {r['status']}"
        if r["status"] == "ok":
            line += (f" | origin {r['origin']} | {r['models']} models, {r['logged']} logged"
                     f" | data {r['dq_mode']}, regime {r['novelty']}")
        print(line + f" | {(time.time() - t_all) / 60:.0f} min elapsed", flush=True)
        for name, why in r.get("failed", []):
            print(f"      {name} failed: {why}")

    tasks = [(c, w, models, not a.verbose) for c in companies]
    try:
        with cf.ProcessPoolExecutor(max_workers=max(1, min(a.jobs, len(tasks)))) as ex:
            for fut in cf.as_completed([ex.submit(_work, task) for task in tasks]):
                report(fut.result())
    except cf.process.BrokenProcessPool:
        # A worker died (out of memory, a native crash). What finished is
        # already logged; the rest run here, one at a time, so a scheduled
        # week is not lost.
        left = [task for task in tasks if task[0] not in done]
        print(f"  a worker process died; running the {len(left)} remaining one at a time", flush=True)
        for task in left:
            report(_work(task))
    ok = sum(r["status"] == "ok" for r in done.values())
    print(f"Done: {ok} of {len(companies)} served in {(time.time() - t_all) / 60:.0f} min.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```
