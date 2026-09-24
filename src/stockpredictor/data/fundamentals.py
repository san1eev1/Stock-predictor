"""Weekly fundamentals snapshots from Yahoo Finance (free).

Free sources do not offer point-in-time fundamental history, so snapshots
are collected from now on (fundamentals/snapshots.csv). They are shown in the
app and used as quality filters now, and can become model inputs once
enough history has built up.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd

FIELDS = {
    "pe": "trailingPE",
    "pb": "priceToBook",
    "roe": "returnOnEquity",
    "debt_to_equity": "debtToEquity",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "profit_margin": "profitMargins",
    "dividend_yield": "dividendYield",
    "market_cap_cr": "marketCap",
}
COLS = ["symbol", "date", *FIELDS]


def parse_info(symbol: str, info: dict, on: date) -> dict:
    row = {"symbol": symbol, "date": on.isoformat()}
    for col, key in FIELDS.items():
        val = info.get(key)
        row[col] = float(val) if isinstance(val, (int, float)) else None
    if row["market_cap_cr"] is not None:
        row["market_cap_cr"] = round(row["market_cap_cr"] / 1e7, 1)  # rupees -> crore
    return row


def fetch_snapshot(symbols: list[str], on: date | None = None,
                   pause: float = 0.5) -> tuple[list[dict], list[str]]:
    import yfinance as yf

    on = on or date.today()
    rows, errors = [], []
    for sym in symbols:
        try:
            rows.append(parse_info(sym, yf.Ticker(f"{sym}.NS").info or {}, on))
        except Exception as exc:
            errors.append(f"{sym}: {exc}")
        time.sleep(pause)
    return rows, errors


def append_snapshot(store_dir: Path, rows: list[dict]) -> None:
    path = store_dir / "fundamentals" / "snapshots.csv"
    new = pd.DataFrame(rows, columns=COLS)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["date"].isin(new["date"])], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["date", "symbol"]).to_csv(path, index=False, float_format="%.4f")


def load_latest(store_dir: Path) -> pd.DataFrame:
    path = store_dir / "fundamentals" / "snapshots.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path)
    return df[df["date"] == df["date"].max()].reset_index(drop=True)
