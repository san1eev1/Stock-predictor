"""Quarterly results dates (Yahoo), for results-day features.

Stocks move differently around results: big gaps on the reaction day, and a drift for
weeks after. Yahoo lists ~20 years of past announcement times plus the next scheduled
ones. Stored in the git data store as earnings.csv (symbol, announced_ist, react_date):
react_date is the first session that can react (the same day if announced before the
15:30 close, else the next weekday).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

FILE = "earnings.csv"
CLOSE = pd.Timedelta(hours=15, minutes=30)
SOON_DAYS = 10     # an upcoming date is used only this close to it (by then it is announced)


def fetch(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    e = yf.Ticker(f"{symbol}.NS").get_earnings_dates(limit=100)
    if e is None or e.empty:
        return pd.DataFrame(columns=["symbol", "announced_ist", "react_date"])
    ist = e.index.tz_convert("Asia/Kolkata").tz_localize(None)
    df = pd.DataFrame({"symbol": symbol, "announced_ist": ist})
    after_close = (df["announced_ist"] - df["announced_ist"].dt.normalize()) >= CLOSE
    react = df["announced_ist"].dt.normalize() + pd.to_timedelta(after_close.astype(int), "D")
    df["react_date"] = react + pd.offsets.BDay(0)     # a weekend rolls to Monday
    return df.drop_duplicates("react_date")


def update(store_dir: Path, symbols: list[str], progress=print) -> int:
    path = Path(store_dir) / FILE
    parts, failed = [], 0
    for s in symbols:
        try:
            parts.append(fetch(s))
        except Exception:
            failed += 1
        time.sleep(0.2)
    if not parts:
        return 0
    new = pd.concat(parts, ignore_index=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["announced_ist", "react_date"])
        new = pd.concat([old[~old["symbol"].isin(new["symbol"])], new], ignore_index=True)
    new.sort_values(["symbol", "react_date"]).to_csv(path, index=False)
    if failed:
        progress(f"earnings: {failed} stocks failed (kept their previous dates)")
    return len(new)


def load(store_dir: Path) -> pd.DataFrame:
    path = Path(store_dir) / FILE
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "announced_ist", "react_date"])
    return pd.read_csv(path, parse_dates=["announced_ist", "react_date"])


def add_features(s: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """days_since_results: weekdays since the last reaction day (0 = today is the reaction
    day); days_to_results: weekdays to the next one, only when within SOON_DAYS (else NaN);
    results_today: 1 on a reaction day."""
    s = s.copy()
    s["days_since_results"], s["days_to_results"], s["results_today"] = np.nan, np.nan, 0.0
    if earnings is None or earnings.empty:
        return s
    ev = {sym: np.sort(g["react_date"].to_numpy().astype("datetime64[D]"))
          for sym, g in earnings.groupby("symbol")}
    since = np.full(len(s), np.nan)
    to = np.full(len(s), np.nan)
    dates = s["date"].to_numpy().astype("datetime64[D]")
    for sym, pos in pd.Series(np.arange(len(s))).groupby(s["symbol"].to_numpy()).indices.items():
        e = ev.get(sym)
        if e is None or not len(e):
            continue
        d = dates[pos]
        i = np.searchsorted(e, d, side="right") - 1           # last reaction day <= d
        has = i >= 0
        since[pos[has]] = np.busday_count(e[i[has]], d[has])
        j = np.searchsorted(e, d, side="right")               # next reaction day > d
        ok = j < len(e)
        nxt = np.busday_count(d[ok], e[j[ok]])
        to[pos[ok]] = np.where(nxt <= SOON_DAYS, nxt, np.nan)
    s["days_since_results"], s["days_to_results"] = since, to
    s["results_today"] = (s["days_since_results"] == 0).astype(float)
    return s
