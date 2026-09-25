"""Intraday data as one compact summary row per stock per day.

Raw 5-minute bars are never stored. Each day is reduced to what the
intraday model and backtest need:

  first 30 minutes (9:15-9:45): open, high, low, close (= 9:45 entry price),
                                volume, VWAP
  after 9:45 until 15:15:       high, low, price at 12:30 (trading exit), price at 15:15,
                                day close
  first-hit times:              for each level in LEVELS (% from the 9:45
                                price), minutes after 9:45 until price first
                                reached +level (u...) or -level (d...), so any
                                stop-loss/target combination can be replayed.

Sources: Yahoo Finance 5-minute bars (free, last ~60 days; collected daily
by GitHub Actions into market-data/intraday/<YEAR>.csv) and Angel One
(years of history; backfilled once on the Mac into data/intraday_backfill.csv).
"""

from __future__ import annotations

import logging
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.config import DATA_DIR

log = logging.getLogger(__name__)

MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)
ENTRY_TIME = time(9, 45)
EXIT_TIME = time(15, 15)          # end of the stored window (first-hit times, px_1515)
TRADE_EXIT = time(12, 30)         # intraday trades are squared off here
EXIT_COL = "px_1230"              # price at TRADE_EXIT: the model's target and square-off price
EXIT_MINUTES = 165                # TRADE_EXIT minus the 9:45 entry
WINDOW_MINUTES = 330              # EXIT_TIME minus 9:45: last minute with stored first-hit times
BAR_MINUTES = 5
LEVELS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)   # percent
LEVEL_COLS = [f"{p}{int(round(lv * 100)):03d}" for p in "ud" for lv in LEVELS]
SUMMARY_COLS = ["symbol", "date", "open", "h30", "l30", "c30", "v30", "vwap30",
                "high_after", "low_after", "px_1515", "close", *LEVEL_COLS, "source", EXIT_COL]
BACKFILL_PATH = DATA_DIR / "intraday_backfill.csv"


def level_col(side: str, pct: float) -> str:
    return f"{side}{int(round(pct * 100)):03d}"


# --- Summaries -------------------------------------------------------------------

def summarize_first30(bars: pd.DataFrame) -> dict | None:
    """Opening-range summary from bars starting 9:15..9:40 (ts = bar start, IST)."""
    first = bars[bars["ts"].dt.time < ENTRY_TIME]
    if len(first) < 4:
        return None
    vol = first["volume"].astype(float)
    tp = (first["high"] + first["low"] + first["close"]) / 3
    return {
        "open": float(first["open"].iloc[0]), "h30": float(first["high"].max()),
        "l30": float(first["low"].min()), "c30": float(first["close"].iloc[-1]),
        "v30": float(vol.sum()),
        "vwap30": float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else float(first["close"].iloc[-1]),
    }


def summarize_day(bars: pd.DataFrame) -> dict | None:
    """Full-day summary for one stock and one date (bars sorted by ts)."""
    head = summarize_first30(bars)
    if head is None:
        return None
    after = bars[(bars["ts"].dt.time >= ENTRY_TIME) & (bars["ts"].dt.time < EXIT_TIME)]
    if len(after) < 10:
        return None
    c30 = head["c30"]
    minutes = ((after["ts"] - after["ts"].iloc[0]).dt.total_seconds() / 60 + BAR_MINUTES).to_numpy()
    highs, lows = after["high"].to_numpy(), after["low"].to_numpy()
    to_exit = after[after["ts"].dt.time < TRADE_EXIT]      # bars ending by 12:30
    row = {**head, "high_after": float(highs.max()), "low_after": float(lows.min()),
           "px_1515": float(after["close"].iloc[-1]), "close": float(bars["close"].iloc[-1]),
           EXIT_COL: float(to_exit["close"].iloc[-1]) if len(to_exit) else np.nan}
    for lv in LEVELS:
        up = np.nonzero(highs >= c30 * (1 + lv / 100))[0]
        dn = np.nonzero(lows <= c30 * (1 - lv / 100))[0]
        row[level_col("u", lv)] = int(minutes[up[0]]) if len(up) else np.nan
        row[level_col("d", lv)] = int(minutes[dn[0]]) if len(dn) else np.nan
    return row


def summarize(bars: pd.DataFrame, symbol: str, source: str) -> list[dict]:
    """Bars for one symbol (any number of days) -> one summary row per day."""
    if bars.empty:
        return []
    bars = bars.sort_values("ts")
    rows = []
    for d, day in bars.groupby(bars["ts"].dt.date):
        s = summarize_day(day)
        if s is not None:
            rows.append({"symbol": symbol, "date": d.isoformat(), **s, "source": source})
    return rows


def _to_ist_naive(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts


# --- Sources ---------------------------------------------------------------------

def yahoo_bars(symbols: list[str], period: str = "60d") -> dict[str, pd.DataFrame]:
    """5-minute bars from Yahoo Finance (free, ~60 days of history)."""
    import yfinance as yf

    out = {}
    for i in range(0, len(symbols), 25):
        chunk = symbols[i:i + 25]
        tickers = [f"{s}.NS" for s in chunk]
        df = yf.download(tickers, period=period, interval="5m", progress=False,
                         group_by="ticker", auto_adjust=False, threads=True)
        for s, t in zip(chunk, tickers):
            try:
                part = df[t] if len(tickers) > 1 else df
            except KeyError:
                continue
            part = part.dropna(subset=["Close"])
            if part.empty:
                continue
            out[s] = pd.DataFrame({
                "ts": _to_ist_naive(part.index.to_series()).values,
                "open": part["Open"].values, "high": part["High"].values,
                "low": part["Low"].values, "close": part["Close"].values,
                "volume": part["Volume"].fillna(0).values})
        _time.sleep(1)
    return out


def chunk_ranges(start: date, end: date, max_days: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into request windows covering full market sessions."""
    ranges = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=max_days - 1), end)
        ranges.append((datetime(cur.year, cur.month, cur.day, *MARKET_OPEN),
                       datetime(stop.year, stop.month, stop.day, *MARKET_CLOSE)))
        cur = stop + timedelta(days=1)
    return ranges


def angel_bars(client, token: str, start: date, end: date,
               interval: str = "FIVE_MINUTE") -> pd.DataFrame:
    """5-minute bars from Angel One for [start, end] (chunked requests)."""
    from stockpredictor.data.angelone import MAX_DAYS_PER_REQUEST

    rows = []
    for frm, to in chunk_ranges(start, end, MAX_DAYS_PER_REQUEST[interval]):
        rows += client.candles(token, interval, frm, to)
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = _to_ist_naive(df["ts"])
    return df


def angel_backfill(client, tokens: dict[str, str], symbols: list[str], start: date, end: date,
                   progress=print) -> int:
    """Download Angel One 5-minute history for `symbols` and save daily summaries locally."""
    total = 0
    for i, sym in enumerate(symbols, 1):
        if sym not in tokens:
            progress(f"[{i}/{len(symbols)}] {sym}: no Angel One token")
            continue
        try:
            rows = summarize(angel_bars(client, tokens[sym], start, end), sym, "angelone")
            total += upsert_backfill(rows)
            progress(f"[{i}/{len(symbols)}] {sym}: {len(rows)} days")
        except Exception as exc:
            progress(f"[{i}/{len(symbols)}] {sym}: error: {exc}")
    return total


def backfill_symbols(path: Path = BACKFILL_PATH) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["symbol"])["symbol"].unique())


def backfill_missing_exit(path: Path = BACKFILL_PATH) -> set[str]:
    """Stocks whose Angel One history lacks the 12:30 exit price (older downloads)."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=lambda c: c in ("symbol", EXIT_COL))
    if EXIT_COL not in df:
        return set(df["symbol"])
    has = df.groupby("symbol")[EXIT_COL].apply(lambda x: x.notna().mean() > 0.5)
    return set(has[~has].index)


def backfill_days(path: Path = BACKFILL_PATH) -> int:
    if not path.exists():
        return 0
    return int(pd.read_csv(path, usecols=["date"])["date"].nunique())


# --- Storage ---------------------------------------------------------------------

def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["date", "symbol"]).reindex(columns=SUMMARY_COLS)
    df.to_csv(path, index=False, float_format="%.4f")


def upsert_store(store_dir: Path, rows: list[dict]) -> int:
    """Merge summary rows into market-data/intraday/<YEAR>.csv (new rows win)."""
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=SUMMARY_COLS)
    for year, part in new.groupby(new["date"].str[:4]):
        path = store_dir / "intraday" / f"{year}.csv"
        if path.exists():
            old = pd.read_csv(path, dtype={"date": str})
            key = set(zip(part["symbol"], part["date"]))
            old = old[[k not in key for k in zip(old["symbol"], old["date"])]]
            part = pd.concat([old, part], ignore_index=True)
        _write(part, path)
    return len(new)


def upsert_backfill(rows: list[dict], path: Path = BACKFILL_PATH) -> int:
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=SUMMARY_COLS)
    if path.exists():
        old = pd.read_csv(path, dtype={"date": str})
        key = set(zip(new["symbol"], new["date"]))
        new = pd.concat([old[[k not in key for k in zip(old["symbol"], old["date"])]], new])
    _write(new, path)
    return len(rows)


def load_summaries(store_dir: Path, backfill: Path = BACKFILL_PATH) -> pd.DataFrame:
    """All intraday summaries: git store + local Angel backfill (Angel wins on overlap)."""
    parts = [pd.read_csv(f, dtype={"date": str}) for f in sorted((store_dir / "intraday").glob("*.csv"))]
    if backfill.exists():
        parts.append(pd.read_csv(backfill, dtype={"date": str}))
    if not parts:
        return pd.DataFrame(columns=SUMMARY_COLS)
    df = pd.concat(parts, ignore_index=True).reindex(columns=SUMMARY_COLS)
    df["_pref"] = (df["source"] == "angelone").astype(int)
    df = df.sort_values("_pref").drop_duplicates(["symbol", "date"], keep="last").drop(columns="_pref")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def split_factor(corporate_actions: pd.DataFrame, symbol: str, on: date) -> float:
    """Multiply a raw price on `on` by this to match split-adjusted daily prices."""
    ca = corporate_actions
    later = ca[(ca["symbol"] == symbol) & (ca["kind"] == "split") & (ca["date"] > on.isoformat())]
    return float(np.prod(1 / later["value"])) if len(later) else 1.0
