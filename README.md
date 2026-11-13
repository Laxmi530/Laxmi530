- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
The forward record, run from the app instead of from scheduled tasks.

README sections 77 and 78 left two routines to Windows scheduled tasks: score
the forecasts that have matured (``run_monitor.bat``) and forecast the frozen
watchlist once a week (``run_watchlist.bat``). The owner chose not to set
those up, so the app does it (README section 82):

* ``record_status`` says what the record holds and whether this week's
  watchlist forecast is due (none yet, or the last one is 7 or more days old).
* ``score_matured`` fills in the outcomes that have arrived. It is idempotent
  and quick, so the app runs it once a day when it opens.
* ``launch_watchlist`` starts ``run_watchlist.py`` as its own process, with its
  output in ``.prediction_log/watchlist_runs/``, so the app stays usable and
  the run carries on if the browser tab is closed. It refuses to start a second
  run while one is alive; ``run_watchlist.py`` also skips any company already
  logged for the same origin, so even a double start cannot log a forecast twice.

Nothing here is scheduled or left behind after the run: one click, one process.
"""

import json
import os
import subprocess
import sys

import pandas as pd

import prediction_log as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(pl.LOG_DIR, "watchlist_runs")
RUN_FILE = os.path.join(RUN_DIR, "current.json")
DUE_AFTER_DAYS = 7


def record_status(log_df=None, now=None):
    """What the forward record holds, and whether the watchlist is due."""
    df = pl.read_log() if log_df is None else log_df
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    resolved = df["realised_price"].notna() if len(df) else pd.Series(dtype=bool)
    wl = df[df["source"] == "watchlist"] if len(df) else df
    last = None
    if len(wl):
        last = pd.to_datetime(wl["logged_at_utc"], utc=True, errors="coerce").max()
        last = None if pd.isna(last) else last.tz_localize(None)
    days = None if last is None else (now - last).days
    return {
        "rows": int(len(df)),
        "resolved": int(resolved.sum()),
        "pending": int((~resolved).sum()),
        "watchlist_rows": int(len(wl)),
        "last_watchlist": last,
        "days_since_watchlist": days,
        "watchlist_due": days is None or days >= DUE_AFTER_DAYS,
    }


def score_matured(verbose=False):
    """Fill in matured outcomes through the same fetch the monitor uses."""
    from get_data import get_whole_stock_data
    from safe_fetch import fetch_with_fallback

    def _fetch(ticker, interval):
        return fetch_with_fallback(ticker, interval, fetch_fn=get_whole_stock_data)

    return pl.resolve_outcomes(_fetch, verbose=verbose)


def _alive(pid):
    """True while process ``pid`` is running (never signals it)."""
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)   # POSIX only: signal 0 checks without signalling
        return True
    except OSError:
        return False


def current_run(run_file=RUN_FILE):
    """The watchlist run in progress, as {'pid', 'started', 'log'}, or None."""
    try:
        with open(run_file, encoding="utf-8") as f:
            run = json.load(f)
    except (OSError, ValueError):
        return None
    return run if _alive(int(run.get("pid", -1))) else None


def launch_watchlist(run_file=RUN_FILE, popen=subprocess.Popen, now=None):
    """
    Start ``run_watchlist.py`` in its own process and return its run record.

    Returns the run already in progress instead if there is one.
    """
    run = current_run(run_file)
    if run:
        return dict(run, already_running=True)
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    os.makedirs(os.path.dirname(run_file), exist_ok=True)
    log = os.path.join(os.path.dirname(run_file), f"watchlist_{now:%Y%m%d_%H%M}.log")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    env = dict(os.environ, STOCKAPP_QUIET="1", PYTHONUNBUFFERED="1")
    with open(log, "w", encoding="utf-8") as out:
        proc = popen([sys.executable, os.path.join(ROOT, "run_watchlist.py")],
                     cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, env=env, creationflags=flags)
    run = {"pid": proc.pid, "started": now.isoformat(timespec="minutes"), "log": log}
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(run, f)
    return dict(run, already_running=False)


def last_lines(path, n=3):
    """The last ``n`` non-empty lines of a run's log (progress for the sidebar)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
    except OSError:
        return []
    return lines[-n:]
```
