"""NSE delivery data: the share of each day's traded quantity that was taken for delivery.

A rising delivery share (people buying to hold, not to trade intraday) is a classic
Indian-market sign of accumulation by institutions. Source: NSE's daily 'Security-wise
delivery position' file (MTO_DDMMYYYY.DAT, available back to 2005). Stored in the git
data store as delivery/<year>.csv (symbol, date, traded_qty, deliv_qty, deliv_pct).
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

URL = "https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d:%d%m%Y}.DAT"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"}
COLS = ["symbol", "date", "traded_qty", "deliv_qty", "deliv_pct"]
CHECKED = "_checked.txt"     # days already fetched (or with no file: holidays)


def parse(text: str, day: date) -> pd.DataFrame:
    """Rows of record type 20, EQ series only."""
    rows = []
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 7 and p[0] == "20" and p[3] == "EQ":
            try:
                rows.append((p[2], pd.Timestamp(day), float(p[4]), float(p[5]), float(p[6])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=COLS)


def fetch(day: date, session: requests.Session | None = None) -> pd.DataFrame | None:
    """One day's file; None when NSE has none (holiday) or it is not published yet."""
    s = session or requests
    r = s.get(URL.format(d=day), headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return parse(r.text, day)


def load(store_dir: Path) -> pd.DataFrame:
    files = sorted((Path(store_dir) / "delivery").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=COLS)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _checked(folder: Path) -> set[str]:
    f = folder / CHECKED
    return set(f.read_text().split()) if f.exists() else set()


def update(store_dir: Path, symbols: set[str], start: date, end: date | None = None,
           minutes: float = 20, progress=print) -> int:
    """Download missing days between start and end (newest first, so recent data comes
    first), keep `symbols` only. Stops after `minutes` and resumes next time."""
    folder = Path(store_dir) / "delivery"
    folder.mkdir(parents=True, exist_ok=True)
    done = _checked(folder)
    end = end or date.today()
    days = [end - timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5 and f"{d:%Y-%m-%d}" not in done]
    deadline = time.monotonic() + minutes * 60
    new, checked, rows = 0, [], []
    session = requests.Session()
    for d in days:
        if time.monotonic() > deadline:
            progress(f"delivery: time budget used; {len(days) - len(checked)} days left for next run")
            break
        try:
            df = fetch(d, session)
        except requests.RequestException as exc:
            progress(f"delivery {d}: {exc}")
            time.sleep(5)
            continue
        if df is None and d >= date.today() - timedelta(days=3):
            continue                      # maybe not published yet: try again later
        checked.append(f"{d:%Y-%m-%d}")
        if df is not None and not df.empty:
            rows.append(df[df["symbol"].isin(symbols)])
            new += 1
        time.sleep(0.3)                   # be polite to NSE
    if rows:
        add = pd.concat(rows, ignore_index=True)
        for year, g in add.groupby(add["date"].dt.year):
            path = folder / f"{year}.csv"
            old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else g.iloc[:0]
            out = pd.concat([old, g]).drop_duplicates(["symbol", "date"], keep="last")
            out.sort_values(["date", "symbol"]).to_csv(path, index=False, date_format="%Y-%m-%d")
    with open(folder / CHECKED, "a") as f:
        f.write("".join(f"{x}\n" for x in checked))
    return new


def features(delivery: pd.DataFrame) -> pd.DataFrame:
    """Per symbol/date: delivery share today, its 20-day average, and whether delivery is
    running above normal (20-day vs 120-day average share and quantity)."""
    if delivery.empty:
        return pd.DataFrame(columns=["symbol", "date"])
    d = delivery.sort_values(["symbol", "date"]).copy()
    d["deliv_pct"] = d["deliv_pct"] / 100
    g = d.groupby("symbol")
    avg = lambda col, n: g[col].transform(lambda x: x.rolling(n, min_periods=n // 2).mean())  # noqa: E731
    d["deliv_pct_20"] = avg("deliv_pct", 20)
    d["deliv_pct_rel"] = d["deliv_pct_20"] / avg("deliv_pct", 120)
    d["deliv_qty_rel"] = avg("deliv_qty", 20) / avg("deliv_qty", 120)
    return d[["symbol", "date", "deliv_pct", "deliv_pct_20", "deliv_pct_rel", "deliv_qty_rel"]]
