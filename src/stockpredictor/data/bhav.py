"""NSE daily bhavcopy: every traded stock, every day since 2005 - including companies that
later fell out of the index, crashed or were delisted.

Backtests on today's Nifty 250 members only are flattered by survivorship bias: the losers
that left the index are missing from the history. This history (the 400 most-traded stocks
of each day, as they were at the time) lets research rebuild the universe point-in-time.
Stored in the git data store as bhav/<year>.parquet (research on GitHub only; the Mac's
sparse checkout leaves it out).
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

OLD = "https://nsearchives.nseindia.com/content/historical/EQUITIES/{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip"
NEW = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
NEW_FROM = date(2024, 7, 8)
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"}
COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "value", "isin"]
TOP_N = 400          # most-traded stocks kept per day
CHECKED = "_checked.txt"


def url(day: date) -> str:
    if day >= NEW_FROM:
        return NEW.format(d=day)
    return OLD.format(d=day, mon=day.strftime("%b").upper())


def parse(raw: bytes, day: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip() for c in df.columns]
    if "TckrSymb" in df:                                   # new format (from July 2024)
        df = df[df["SctySrs"].astype(str).str.strip() == "EQ"]
        out = pd.DataFrame({"symbol": df["TckrSymb"], "open": df["OpnPric"],
                            "high": df["HghPric"], "low": df["LwPric"], "close": df["ClsPric"],
                            "volume": df["TtlTradgVol"], "value": df["TtlTrfVal"],
                            "isin": df["ISIN"]})
    else:
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        out = pd.DataFrame({"symbol": df["SYMBOL"], "open": df["OPEN"], "high": df["HIGH"],
                            "low": df["LOW"], "close": df["CLOSE"], "volume": df["TOTTRDQTY"],
                            "value": df["TOTTRDVAL"],
                            "isin": df["ISIN"] if "ISIN" in df else ""})   # none before ~2011
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["date"] = pd.Timestamp(day)
    return out[COLS]


def fetch(day: date, session: requests.Session | None = None) -> pd.DataFrame | None:
    r = (session or requests).get(url(day), headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return parse(r.content, day)


def update(store_dir: Path, start: date = date(2005, 1, 1), end: date | None = None,
           minutes: float = 20, progress=print) -> int:
    """Download missing days (newest first), keep each day's TOP_N by traded value.
    Stops after `minutes`; resumes next run."""
    folder = Path(store_dir) / "bhav"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / CHECKED
    done = set(f.read_text().split()) if f.exists() else set()
    end = end or date.today()
    days = [end - timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5 and f"{d:%Y-%m-%d}" not in done]
    deadline = time.monotonic() + minutes * 60
    rows, checked, session = [], [], requests.Session()
    for d in days:
        if time.monotonic() > deadline:
            progress(f"bhavcopy: time budget used; {len(days) - len(checked)} days left")
            break
        try:
            df = fetch(d, session)
        except (requests.RequestException, zipfile.BadZipFile, KeyError, ValueError) as exc:
            progress(f"bhavcopy {d}: {exc}")
            time.sleep(3)
            continue
        if df is None and d >= date.today() - timedelta(days=3):
            continue
        checked.append(f"{d:%Y-%m-%d}")
        if df is not None and not df.empty:
            rows.append(df.nlargest(TOP_N, "value"))
        time.sleep(0.3)
    if rows:
        add = pd.concat(rows, ignore_index=True)
        for year, g in add.groupby(add["date"].dt.year):
            path = folder / f"{year}.parquet"
            old = pd.read_parquet(path) if path.exists() else g.iloc[:0]
            out = pd.concat([old, g]).drop_duplicates(["symbol", "date"], keep="last")
            out.sort_values(["date", "symbol"]).to_parquet(path, index=False,
                                                           compression="zstd")
    with open(f, "a") as fh:
        fh.write("".join(f"{x}\n" for x in checked))
    return len(rows)


def load(store_dir: Path) -> pd.DataFrame:
    files = sorted((Path(store_dir) / "bhav").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=COLS)
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
