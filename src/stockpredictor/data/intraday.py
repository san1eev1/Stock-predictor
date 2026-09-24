"""Intraday candle history from Angel One SmartAPI.

Candles are stored raw (not split-adjusted). Intraday features are computed
within a single day, where adjustment does not matter; split_factor() is
available when intraday and daily prices must be compared across a split.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

from stockpredictor.data.angelone import MAX_DAYS_PER_REQUEST

MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)


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


def last_ts(conn: sqlite3.Connection, symbol: str, interval: str) -> datetime | None:
    row = conn.execute(
        "SELECT MAX(ts) FROM intraday_prices WHERE symbol = ? AND interval = ?",
        (symbol, interval)).fetchone()
    return datetime.fromisoformat(row[0]) if row and row[0] else None


def update_symbol(conn: sqlite3.Connection, client, symbol: str, token: str,
                  interval: str, start: date, end: date | None = None) -> int:
    """Fetch and store intraday candles for one stock, resuming from the last stored day."""
    end = end or date.today()
    last = last_ts(conn, symbol, interval)
    if last:
        start = max(start, last.date())  # re-fetch: that day may have been partial
    total = 0
    for frm, to in chunk_ranges(start, end, MAX_DAYS_PER_REQUEST[interval]):
        candles = client.candles(token, interval, frm, to)
        rows = [(symbol, str(ts), interval, float(o), float(h), float(l), float(c), int(v))
                for ts, o, h, l, c, v in candles]
        conn.executemany(
            """INSERT OR REPLACE INTO intraday_prices
               (symbol, ts, interval, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        total += len(rows)
    return total


def split_factor(conn: sqlite3.Connection, symbol: str, on: date) -> float:
    """Multiply a raw price on `on` by this to match Yahoo's split-adjusted daily prices."""
    factor = 1.0
    for (value,) in conn.execute(
            "SELECT value FROM corporate_actions WHERE symbol = ? AND kind = 'split' AND date > ?",
            (symbol, on.isoformat())):
        factor /= value
    return factor
