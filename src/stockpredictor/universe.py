"""Nifty LargeMidcap 250 stock universe (Nifty 100 + Midcap 150), from the official NSE lists."""

from __future__ import annotations

import csv
import io
import sqlite3

import requests

LIST_URLS = [
    "https://archives.nseindia.com/content/indices/{name}.csv",
    "https://www.niftyindices.com/IndexConstituent/{name}.csv",
]
# Both models train on, and pick from, the same 250 stocks. If that list can't be
# downloaded, fall back to smaller lists (each is a subset of the 250).
UNIVERSE_LISTS = [("ind_niftylargemidcap250list", 250), ("ind_nifty200list", 200),
                  ("ind_nifty100list", 100)]
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


def fetch_list(name: str, expected: int) -> list[dict]:
    errors = []
    for url in LIST_URLS:
        url = url.format(name=name)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            stocks = parse_constituents(resp.text)
            if len(stocks) >= expected * 0.9:
                return stocks
            errors.append(f"{url}: only {len(stocks)} rows")
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Could not download {name}:\n" + "\n".join(errors))


SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def parse_sp500(csv_text: str) -> list[dict]:
    """datasets/s-and-p-500-companies: Symbol, Security, GICS Sector, ..., CIK."""
    rows = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        sym = (row.get("Symbol") or "").strip()
        if sym:
            rows.append({"symbol": sym, "name": (row.get("Security") or "").strip(),
                         "industry": (row.get("GICS Sector") or "").strip(),
                         "isin": (row.get("CIK") or "").strip()})
    return rows


def fetch_sp500() -> list[dict]:
    resp = requests.get(SP500_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    stocks = parse_sp500(resp.text)
    if len(stocks) < 450:
        raise RuntimeError(f"S&P 500 list has only {len(stocks)} rows")
    return [{**s, "tradable": 1} for s in sorted(stocks, key=lambda s: s["symbol"])]


def fetch_universe() -> list[dict]:
    """Nifty LargeMidcap 250 (India) or the S&P 500 (MARKET=us): every stock is used for
    training and is tradable."""
    from stockpredictor.config import MARKET

    if MARKET.name == "us":
        return fetch_sp500()
    errors = []
    for name, expected in UNIVERSE_LISTS:
        try:
            stocks = fetch_list(name, expected)
            return [{**s, "tradable": 1} for s in sorted(stocks, key=lambda s: s["symbol"])]
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError("\n".join(errors))


def save_universe(conn: sqlite3.Connection, stocks: list[dict]) -> None:
    """Upsert current constituents; mark stocks that left the index inactive.

    Old stocks are kept (not deleted) so historical data stays usable and
    backtests avoid survivorship bias.
    """
    symbols = [s["symbol"] for s in stocks]
    if len(symbols) != len(set(symbols)):
        raise ValueError("duplicate symbols in constituents list")
    stocks = [{"tradable": 1, **s} for s in stocks]
    conn.execute("UPDATE stocks SET active = 0, tradable = 0")
    conn.executemany(
        """
        INSERT INTO stocks (symbol, name, industry, isin, active, tradable, updated_at)
        VALUES (:symbol, :name, :industry, :isin, 1, :tradable, datetime('now'))
        ON CONFLICT(symbol) DO UPDATE SET
            name = excluded.name, industry = excluded.industry, isin = excluded.isin,
            active = 1, tradable = excluded.tradable, updated_at = excluded.updated_at
        """,
        stocks,
    )
    conn.commit()


def active_symbols(conn: sqlite3.Connection) -> list[str]:
    return [r["symbol"] for r in conn.execute(
        "SELECT symbol FROM stocks WHERE active = 1 ORDER BY symbol")]
