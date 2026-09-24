"""Daily OHLCV history for stocks and indices from Yahoo Finance (free).

Yahoo's Open/High/Low/Close are already split- and bonus-adjusted; Adj Close is
additionally dividend-adjusted. Because the adjustment is applied at download
time, a new split makes previously stored rows inconsistent — so whenever an
incremental update sees a new split, that stock's full history is re-downloaded.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import date, timedelta

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_START = date(2005, 1, 1)
OVERLAP_DAYS = 5

# Market context indices: our name -> Yahoo ticker. (Yahoo has no usable history
# for most other NSE sector indices; sector strength is computed from peers instead.)
INDICES = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
    "NIFTYIT": "^CNXIT",
    "NIFTYPHARMA": "^CNXPHARMA",
}


def yahoo_ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def download_history(ticker: str, start: date, end: date | None = None,
                     retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=(end or date.today() + timedelta(days=1)).isoformat(),
                auto_adjust=False, actions=True,
            )
            return df
        except Exception as exc:  # network hiccups, Yahoo throttling
            if attempt == retries:
                raise
            log.warning("%s download failed (%s), retrying", ticker, exc)
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def history_to_rows(symbol: str, df: pd.DataFrame, source: str = "yahoo") -> list[tuple]:
    """Convert a yfinance history frame into daily_prices rows, dropping bad candles."""
    rows = []
    for ts, r in df.iterrows():
        o, h, l, c = r.get("Open"), r.get("High"), r.get("Low"), r.get("Close")
        if any(pd.isna(x) for x in (o, h, l, c)) or c <= 0:
            continue
        if h < max(o, c, l) or l > min(o, c, h):
            log.warning("%s %s: inconsistent OHLC skipped", symbol, ts)
            continue
        adj = r.get("Adj Close", c)
        vol = r.get("Volume", 0)
        rows.append((
            symbol, pd.Timestamp(ts).strftime("%Y-%m-%d"),
            float(o), float(h), float(l), float(c),
            float(c if pd.isna(adj) else adj),
            int(0 if pd.isna(vol) else vol),
            source,
        ))
    return rows


def history_to_actions(symbol: str, df: pd.DataFrame) -> list[tuple]:
    """Extract (symbol, date, kind, value) rows for splits/bonuses and dividends."""
    actions = []
    for ts, r in df.iterrows():
        d = pd.Timestamp(ts).strftime("%Y-%m-%d")
        split = r.get("Stock Splits", 0)
        div = r.get("Dividends", 0)
        if split and not pd.isna(split) and split != 0:
            actions.append((symbol, d, "split", float(split)))
        if div and not pd.isna(div) and div != 0:
            actions.append((symbol, d, "dividend", float(div)))
    return actions


def last_date(conn: sqlite3.Connection, table: str, symbol: str) -> date | None:
    row = conn.execute(f"SELECT MAX(date) FROM {table} WHERE symbol = ?", (symbol,)).fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None


def _known_split_dates(conn: sqlite3.Connection, symbol: str) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT date FROM corporate_actions WHERE symbol = ? AND kind = 'split'", (symbol,))}


def _store(conn, table: str, rows: list[tuple]) -> None:
    conn.executemany(
        f"""INSERT OR REPLACE INTO {table}
            (symbol, date, open, high, low, close, adj_close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def _fetch_from(last: date | None, start: date) -> date:
    # Re-fetch a few recent days: Yahoo may have stored a partial candle
    # for the day that was still trading at the previous update.
    return last - timedelta(days=OVERLAP_DAYS) if last else start


def update_stock(conn: sqlite3.Connection, symbol: str,
                 start: date = DEFAULT_START, downloader=download_history) -> int:
    """Download new daily candles for one stock. Returns rows written."""
    last = last_date(conn, "daily_prices", symbol)
    df = downloader(yahoo_ticker(symbol), _fetch_from(last, start))
    if df is None or df.empty:
        return 0

    actions = history_to_actions(symbol, df)
    split_dates = {a[1] for a in actions if a[2] == "split"}
    if last is not None and split_dates - _known_split_dates(conn, symbol):
        log.info("%s: new split/bonus detected, re-downloading full history", symbol)
        df = downloader(yahoo_ticker(symbol), start)
        conn.execute("DELETE FROM daily_prices WHERE symbol = ?", (symbol,))
        actions = history_to_actions(symbol, df)

    rows = history_to_rows(symbol, df)
    _store(conn, "daily_prices", rows)
    conn.executemany(
        "INSERT OR REPLACE INTO corporate_actions (symbol, date, kind, value) VALUES (?, ?, ?, ?)",
        actions,
    )
    conn.commit()
    return len(rows)


def first_date(conn: sqlite3.Connection, table: str, symbol: str) -> date | None:
    row = conn.execute(f"SELECT MIN(date) FROM {table} WHERE symbol = ?", (symbol,)).fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None


def extend_back(conn: sqlite3.Connection, symbol: str, start: date, table: str = "daily_prices",
                ticker: str | None = None, downloader=download_history) -> int:
    """Download history older than what is stored (e.g. 2005-2009). Returns rows added."""
    first = first_date(conn, table, symbol)
    if first is None or (first - start).days <= 10:
        return 0
    df = downloader(ticker or yahoo_ticker(symbol), start, first)
    if df is None or df.empty:
        return 0
    rows = [r for r in history_to_rows(symbol, df) if r[1] < first.isoformat()]
    _store(conn, table, rows)
    if table == "daily_prices":
        conn.executemany("INSERT OR REPLACE INTO corporate_actions (symbol, date, kind, value) "
                         "VALUES (?, ?, ?, ?)", history_to_actions(symbol, df))
    conn.commit()
    return len(rows)


def update_index(conn: sqlite3.Connection, name: str, ticker: str,
                 start: date = DEFAULT_START, downloader=download_history) -> int:
    last = last_date(conn, "index_prices", name)
    df = downloader(ticker, _fetch_from(last, start))
    if df is None or df.empty:
        return 0
    rows = history_to_rows(name, df)
    _store(conn, "index_prices", rows)
    conn.commit()
    return len(rows)


def update_all(conn: sqlite3.Connection, symbols: list[str], start: date = DEFAULT_START,
               downloader=download_history, pause: float = 0.3,
               extend: bool = False) -> dict[str, str]:
    """Update every stock and index; returns {name: 'N rows' | 'error: ...'}."""
    conn.execute(f"DELETE FROM index_prices WHERE symbol NOT IN ({','.join('?' * len(INDICES))})",
                 list(INDICES))
    results = {}
    def stock_job(s):
        added = extend_back(conn, s, start, downloader=downloader) if extend else 0
        return update_stock(conn, s, start, downloader) + added

    def index_job(n, t):
        added = extend_back(conn, n, start, "index_prices", t, downloader) if extend else 0
        return update_index(conn, n, t, start, downloader) + added

    jobs = [(s, lambda s=s: stock_job(s)) for s in symbols]
    jobs += [(n, lambda n=n, t=t: index_job(n, t)) for n, t in INDICES.items()]
    for i, (name, job) in enumerate(jobs, 1):
        try:
            results[name] = f"{job()} rows"
        except Exception as exc:
            results[name] = f"error: {exc}"
            log.error("%s failed: %s", name, exc)
        print(f"[{i}/{len(jobs)}] {name}: {results[name]}", flush=True)
        time.sleep(pause)
    return results
