"""Nifty 100 stock universe, downloaded from the official NSE constituents list."""

from __future__ import annotations

import csv
import io
import sqlite3

import requests

NIFTY100_CSV_URLS = [
    "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
]
# NSE rejects requests without a browser-like User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"}


def parse_constituents(csv_text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        if not row.get("Symbol"):
            continue
        rows.append({
            "symbol": row["Symbol"],
            "name": row.get("Company Name", ""),
            "industry": row.get("Industry", ""),
            "isin": row.get("ISIN Code", ""),
        })
    return rows


def fetch_nifty100() -> list[dict]:
    errors = []
    for url in NIFTY100_CSV_URLS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            stocks = parse_constituents(resp.text)
            if len(stocks) >= 90:  # sanity check: Nifty 100 has 100 stocks
                return stocks
            errors.append(f"{url}: only {len(stocks)} rows")
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download Nifty 100 list:\n" + "\n".join(errors))


def save_universe(conn: sqlite3.Connection, stocks: list[dict]) -> None:
    """Upsert current constituents; mark stocks that left the index inactive.

    Old stocks are kept (not deleted) so historical data stays usable and
    backtests avoid survivorship bias.
    """
    symbols = [s["symbol"] for s in stocks]
    if len(symbols) != len(set(symbols)):
        raise ValueError("duplicate symbols in constituents list")
    conn.execute("UPDATE stocks SET active = 0")
    conn.executemany(
        """
        INSERT INTO stocks (symbol, name, industry, isin, active, updated_at)
        VALUES (:symbol, :name, :industry, :isin, 1, datetime('now'))
        ON CONFLICT(symbol) DO UPDATE SET
            name = excluded.name, industry = excluded.industry,
            isin = excluded.isin, active = 1, updated_at = excluded.updated_at
        """,
        stocks,
    )
    conn.commit()


def active_symbols(conn: sqlite3.Connection) -> list[str]:
    return [r["symbol"] for r in conn.execute(
        "SELECT symbol FROM stocks WHERE active = 1 ORDER BY symbol")]
