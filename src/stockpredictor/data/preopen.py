"""NSE pre-open auction (9:00-9:08): the indicative opening price and the orders behind it.

The pre-open shows, before trading starts, how many shares are waiting to buy vs sell and
how much traded in the auction - order-flow information the opening price alone hides.
NSE publishes only the current day (no history), so it is collected every trading day
(GitHub data workflow; the Mac fetches it live for the 9:45 picks) into
preopen/<year>.csv; the model can use it once a few months have accumulated (until then
the features are empty and the model ignores them).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://www.nseindia.com/api/market-data-pre-open?key=ALL"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36",
           "Referer": "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market",
           "Accept": "application/json"}
COLS = ["symbol", "date", "iep", "prev_close", "po_qty", "buy_qty", "sell_qty"]
FEATURES = ["po_gap", "po_imbalance", "po_qty_adv"]


def fetch() -> pd.DataFrame:
    """Today's pre-open for all NSE stocks (the latest session NSE shows)."""
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = []
    for item in js.get("data", []):
        d = item.get("detail", {}).get("preOpenMarket", {})
        m = item.get("metadata", {})
        ts = d.get("lastUpdateTime") or js.get("timestamp")
        if not m.get("symbol") or not ts:
            continue
        rows.append((m["symbol"], pd.to_datetime(ts, format="%d-%b-%Y %H:%M:%S").normalize(),
                     d.get("IEP") or d.get("finalPrice"), m.get("previousClose") or d.get("prevClose"),
                     d.get("finalQuantity") or d.get("totalTradedVolume"),
                     d.get("totalBuyQuantity"), d.get("totalSellQuantity")))
    return pd.DataFrame(rows, columns=COLS)


def update(store_dir: Path, symbols: set[str]) -> int:
    df = fetch()
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return 0
    folder = Path(store_dir) / "preopen"
    folder.mkdir(parents=True, exist_ok=True)
    for year, g in df.groupby(df["date"].dt.year):
        path = folder / f"{year}.csv"
        old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else g.iloc[:0]
        out = pd.concat([old, g]).drop_duplicates(["symbol", "date"], keep="last")
        out.sort_values(["date", "symbol"]).to_csv(path, index=False, date_format="%Y-%m-%d")
    return len(df)


def load(store_dir: Path) -> pd.DataFrame:
    files = sorted((Path(store_dir) / "preopen").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=COLS)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach(summ: pd.DataFrame, store_dir: Path, live: pd.DataFrame | None = None) -> pd.DataFrame:
    """summ + raw pre-open columns (stored history, plus today's `live` fetch)."""
    po = load(store_dir)
    if live is not None and not live.empty:
        po = pd.concat([po, live], ignore_index=True).drop_duplicates(["symbol", "date"],
                                                                     keep="last")
    s = summ.copy()
    s["date"] = pd.to_datetime(s["date"])
    if po.empty:
        for c in ("iep", "po_qty", "buy_qty", "sell_qty"):
            s[c] = np.nan
        return s
    return s.merge(po[["symbol", "date", "iep", "po_qty", "buy_qty", "sell_qty"]],
                   on=["symbol", "date"], how="left")


def add_features(s: pd.DataFrame) -> pd.DataFrame:
    """po_gap: auction price vs yesterday's close; po_imbalance: (buy - sell) / (buy + sell)
    pending orders; po_qty_adv: auction volume vs the 20-day average daily volume."""
    s = s.copy()
    for c in ("iep", "po_qty", "buy_qty", "sell_qty"):
        if c not in s:
            s[c] = np.nan
    s["po_gap"] = s["iep"] / s["prev_close"] - 1
    tot = (s["buy_qty"] + s["sell_qty"]).replace(0, np.nan)
    s["po_imbalance"] = (s["buy_qty"] - s["sell_qty"]) / tot
    s["po_qty_adv"] = s["po_qty"] / s["adv20"]
    return s
