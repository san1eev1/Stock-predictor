"""NSE daily FII/FPI and DII cash-market buying and selling (Rs crore).

NSE publishes only the latest day, so it is collected every evening (GitHub data
workflow) into fiidii.csv; once a history has built up it can be tested as a market-wide
feature (foreign/domestic institutional money flowing in or out). Lead from the repo study
(research/repository_matrix.md, round 3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

URL = "https://www.nseindia.com/api/fiidiiTradeReact"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36",
           "Referer": "https://www.nseindia.com/reports/fii-dii", "Accept": "application/json"}
COLS = ["date", "category", "buy", "sell", "net"]
FILE = "fiidii.csv"


def parse(js: list[dict]) -> pd.DataFrame:
    rows = [(pd.to_datetime(r["date"], format="%d-%b-%Y"),
             "FII" if r["category"].upper().startswith("FII") else "DII",
             float(r["buyValue"]), float(r["sellValue"]), float(r["netValue"]))
            for r in js if r.get("date") and r.get("category")]
    return pd.DataFrame(rows, columns=COLS)


def fetch() -> pd.DataFrame:
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return parse(r.json())


def update(store_dir: Path) -> int:
    new = fetch()
    if new.empty:
        return 0
    path = Path(store_dir) / FILE
    old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=COLS)
    out = pd.concat([old, new]).drop_duplicates(["date", "category"], keep="last")
    out.sort_values(["date", "category"]).to_csv(path, index=False, date_format="%Y-%m-%d")
    return len(out) - len(old)


def load(store_dir: Path) -> pd.DataFrame:
    path = Path(store_dir) / FILE
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=COLS)
