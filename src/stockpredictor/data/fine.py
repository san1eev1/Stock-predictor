"""Fine-grained opening features from 1-minute bars (and 3-minute bars built from them).

The 9:45 picks use the first 30 minutes. 5-minute bars hide what happened inside them;
1-minute bars show the first 5 and 15 minutes, whether the price broke out of its
first-15-minute range by 9:45, bursts of volume, how steadily it rose or fell. 3-minute
bars (built from the 1-minute ones) give the last push before 9:45 and the trend's
consistency.

History: Angel One 1-minute candles (30 days per request), summarised to one row per stock
per day in data/intraday_fine.csv (Angel One data: local on the Mac, GitHub's private
cache in the cloud; never committed).
"""

from __future__ import annotations

import os
import time as _time
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.config import DATA_DIR
from stockpredictor.data import intraday as I

FINE_PATH = DATA_DIR / "intraday_fine.csv"
FINE_COLS = ["r5", "r15", "orb15", "orb15_dist", "vol_burst", "up1_share", "r3_last",
             "up3_share", "vwap_slope"]
KEY = ["symbol", "date"]
START, ENTRY = time(9, 15), I.ENTRY_TIME


def summarize_fine(bars: pd.DataFrame) -> dict | None:
    """One stock-day's 1-minute bars (ts = bar start, IST) -> fine opening features."""
    b = bars[(bars["ts"].dt.time >= START) & (bars["ts"].dt.time < ENTRY)].sort_values("ts")
    if len(b) < 20:
        return None
    o = float(b["open"].iloc[0])
    c = b["close"].astype(float).to_numpy()
    t = b["ts"].dt.time
    first5, first15 = b[t < time(9, 20)], b[t < time(9, 30)]
    hi15, lo15, c30 = float(first15["high"].max()), float(first15["low"].min()), c[-1]
    vol = b["volume"].astype(float).to_numpy()
    tp = ((b["high"] + b["low"] + b["close"]) / 3).to_numpy(float)
    vwap = lambda n: (tp[:n] * vol[:n]).sum() / vol[:n].sum() if vol[:n].sum() > 0 else c[n - 1]  # noqa: E731
    three = b.set_index("ts").resample("3min", origin="start_day", offset="15min").agg(
        {"open": "first", "close": "last"}).dropna()
    return {
        "r5": float(first5["close"].iloc[-1]) / o - 1 if len(first5) else np.nan,
        "r15": float(first15["close"].iloc[-1]) / o - 1,
        "orb15": 1.0 if c30 > hi15 else -1.0 if c30 < lo15 else 0.0,
        "orb15_dist": (c30 - hi15) / c30 if c30 > hi15 else (c30 - lo15) / c30 if c30 < lo15 else 0.0,
        "vol_burst": float(vol.max() / vol.mean()) if vol.mean() > 0 else np.nan,
        "up1_share": float((b["close"] > b["open"]).mean()),
        "r3_last": float(three["close"].iloc[-1] / three["open"].iloc[-1] - 1) if len(three) else np.nan,
        "up3_share": float((three["close"] > three["open"]).mean()) if len(three) else np.nan,
        "vwap_slope": float(vwap(len(b)) / vwap(min(15, len(b))) - 1),
    }


def summarize_days(bars: pd.DataFrame, symbol: str) -> list[dict]:
    out = []
    for d, g in bars.groupby(bars["ts"].dt.date):
        f = summarize_fine(g)
        if f:
            out.append({"symbol": symbol, "date": f"{d:%Y-%m-%d}", **f})
    return out


def load(path: Path = FINE_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KEY + FINE_COLS)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def upsert(rows: list[dict], path: Path = FINE_PATH) -> int:
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=KEY + FINE_COLS)
    with I.WRITE_LOCK:
        if path.exists():
            old = pd.read_csv(path, dtype={"date": str})
            keys = set(zip(new["symbol"], new["date"]))
            new = pd.concat([old[[k not in keys for k in zip(old["symbol"], old["date"])]], new])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        new.sort_values(["date", "symbol"]).to_csv(tmp, index=False, float_format="%.6f")
        os.replace(tmp, path)
    return len(rows)


def coverage(path: Path = FINE_PATH) -> pd.Series:
    """Days of fine data per stock."""
    if not path.exists():
        return pd.Series(dtype=int)
    return pd.read_csv(path, usecols=["symbol"])["symbol"].value_counts()


def backfill(client, tokens: dict[str, str], symbols: list[str], start: date, end: date,
             progress=print, deadline: float | None = None) -> int:
    """1-minute Angel One history -> fine features, stock by stock (resumable)."""
    total, fails = 0, 0
    for i, sym in enumerate(symbols, 1):
        if deadline is not None and _time.monotonic() > deadline:
            progress(f"fine data: time budget used after {i - 1} of {len(symbols)} stocks")
            break
        if sym not in tokens:
            continue
        try:
            bars = I.angel_bars(client, tokens[sym], start, end, interval="ONE_MINUTE")
            total += upsert(summarize_days(bars, sym))
            fails = 0
        except Exception as exc:
            fails += 1
            progress(f"fine data {sym}: {exc}")
            if fails >= I.MAX_FAILS:
                progress("fine data: Angel One is refusing requests; stopping for now")
                break
    return total


def attach(summ: pd.DataFrame, path: Path = FINE_PATH) -> pd.DataFrame:
    """summ + fine columns: stored history, keeping values already in summ (today's live
    row carries its own)."""
    fine = load(path)
    s = summ.copy()
    for col in FINE_COLS:
        if col not in s:
            s[col] = np.nan
    if fine.empty:
        return s
    s["date"] = pd.to_datetime(s["date"])
    m = s[KEY].merge(fine, on=KEY, how="left")
    for col in FINE_COLS:
        s[col] = s[col].fillna(pd.Series(m[col].to_numpy(), index=s.index))
    return s
