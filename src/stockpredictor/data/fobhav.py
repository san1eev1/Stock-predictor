"""NSE F&O bhavcopy, reduced to one row per stock and day: positioning in its futures and
options (open interest), for research on whether it predicts the stock (E-011).

Per stock (stock futures / options only, index contracts dropped):
  fut_close, fut_oi   near-month future's close and open interest
  fut_oi_all          open interest summed over all futures expiries
  call_oi, put_oi     open interest summed over all option strikes and expiries
  call_vol, put_vol   contracts traded
Stored in the git data store as fobhav/<year>.parquet (research on GitHub; the Mac's sparse
checkout leaves it out). Same archive pattern as data/bhav.py.
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

OLD = ("https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d:%Y}/{mon}/"
       "fo{d:%d}{mon}{d:%Y}bhav.csv.zip")
NEW = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
NEW_FROM = date(2024, 7, 8)
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"}
COLS = ["symbol", "date", "fut_close", "fut_oi", "fut_oi_all", "call_oi", "put_oi",
        "call_vol", "put_vol"]
CHECKED = "_checked.txt"


def url(day: date) -> str:
    if day >= NEW_FROM:
        return NEW.format(d=day)
    return OLD.format(d=day, mon=day.strftime("%b").upper())


def reduce(rows: pd.DataFrame, day: date) -> pd.DataFrame:
    """rows: kind ('fut'|'opt'), symbol, expiry, opt ('CE'|'PE'|''), close, oi, vol."""
    rows = rows.assign(symbol=rows["symbol"].astype(str).str.strip())
    fut = rows[rows["kind"] == "fut"].sort_values(["symbol", "expiry"])
    near = fut.groupby("symbol").first()
    out = pd.DataFrame({"fut_close": near["close"], "fut_oi": near["oi"],
                        "fut_oi_all": fut.groupby("symbol")["oi"].sum()})
    opt = rows[rows["kind"] == "opt"]
    for side, name in (("CE", "call"), ("PE", "put")):
        g = opt[opt["opt"] == side].groupby("symbol")
        out[f"{name}_oi"] = g["oi"].sum()
        out[f"{name}_vol"] = g["vol"].sum()
    out = out.reset_index().rename(columns={"index": "symbol"})
    out["date"] = pd.Timestamp(day)
    return out.reindex(columns=COLS)


def _expiry(col: pd.Series) -> pd.Series:
    """'30-Nov-2023' or, in some 2012 files, '31-May-12'."""
    full = pd.to_datetime(col, format="%d-%b-%Y", errors="coerce")
    return full.fillna(pd.to_datetime(col, format="%d-%b-%y", errors="coerce"))


def parse(raw: bytes, day: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "FinInstrmTp" in df:                                   # new format (from July 2024)
        df = df[df["FinInstrmTp"].isin(["STF", "STO"])]
        rows = pd.DataFrame({"kind": df["FinInstrmTp"].map({"STF": "fut", "STO": "opt"}),
                             "symbol": df["TckrSymb"], "expiry": pd.to_datetime(df["XpryDt"]),
                             "opt": df["OptnTp"].fillna("").astype(str).str.strip(),
                             "close": df["ClsPric"], "oi": df["OpnIntrst"],
                             "vol": df["TtlTradgVol"]})
    else:
        df = df[df["INSTRUMENT"].isin(["FUTSTK", "OPTSTK"])]
        rows = pd.DataFrame({"kind": df["INSTRUMENT"].map({"FUTSTK": "fut", "OPTSTK": "opt"}),
                             "symbol": df["SYMBOL"],
                             "expiry": _expiry(df["EXPIRY_DT"]),
                             "opt": df["OPTION_TYP"].astype(str).str.strip(),
                             "close": df["CLOSE"], "oi": df["OPEN_INT"], "vol": df["CONTRACTS"]})
    return reduce(rows, day)


def fetch(day: date, session: requests.Session | None = None) -> pd.DataFrame | None:
    r = (session or requests).get(url(day), headers=HEADERS, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return parse(r.content, day)


def update(store_dir: Path, start: date = date(2012, 1, 1), end: date | None = None,
           minutes: float = 20, progress=print) -> int:
    """Download missing days, newest first; stops after `minutes` and resumes next run."""
    folder = Path(store_dir) / "fobhav"
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
            progress(f"F&O bhavcopy: time budget used; {len(days) - len(checked)} days left")
            break
        try:
            df = fetch(d, session)
        except (requests.RequestException, zipfile.BadZipFile, KeyError, ValueError) as exc:
            progress(f"F&O bhavcopy {d}: {exc}")
            time.sleep(3)
            continue
        if df is None and d >= date.today() - timedelta(days=3):
            continue
        checked.append(f"{d:%Y-%m-%d}")
        if df is not None and not df.empty:
            rows.append(df)
        time.sleep(0.3)
    if rows:
        add = pd.concat(rows, ignore_index=True)
        for year, g in add.groupby(add["date"].dt.year):
            path = folder / f"{year}.parquet"
            old = pd.read_parquet(path) if path.exists() else g.iloc[:0]
            out = pd.concat([old, g]).drop_duplicates(["symbol", "date"], keep="last")
            out.sort_values(["date", "symbol"]).to_parquet(path, index=False, compression="zstd")
    with open(f, "a") as fh:
        fh.write("".join(f"{x}\n" for x in checked))
    return len(rows)


def load(store_dir: Path) -> pd.DataFrame:
    files = sorted((Path(store_dir) / "fobhav").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=COLS)
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def features(fo: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Positioning features per stock and day (known after that day's close).
    close: date x symbol stock closes (for returns and the futures basis)."""
    fo = fo.sort_values(["symbol", "date"]).copy()
    g = fo.groupby("symbol")
    out = fo[["symbol", "date"]].copy()
    for n in (1, 5, 20):
        out[f"fo_oi_chg_{n}"] = g["fut_oi_all"].pct_change(n, fill_method=None)
    spot = close.stack(future_stack=True).rename("spot").reset_index()
    spot.columns = ["date", "symbol", "spot"]
    fo = fo.merge(spot, on=["date", "symbol"], how="left")
    out["fo_basis"] = (fo["fut_close"] / fo["spot"] - 1).values
    ret5 = fo.groupby("symbol")["spot"].pct_change(5, fill_method=None).values
    # long build-up (price up, OI up) > 0; short build-up (down, up) < 0; covering /
    # unwinding (OI down) near the opposite sign, scaled by the OI change
    out["fo_buildup_5"] = out["fo_oi_chg_5"].values * pd.Series(ret5).apply(
        lambda x: 1 if x > 0 else -1 if x < 0 else 0).values
    pcr = fo["put_oi"] / fo["call_oi"].where(fo["call_oi"] > 0)
    out["fo_pcr"] = pcr.values
    out["fo_pcr_chg_5"] = pcr.groupby(fo["symbol"]).pct_change(5, fill_method=None).values
    out["fo_opt_activity"] = ((fo["call_vol"] + fo["put_vol"])
                              / (fo["call_oi"] + fo["put_oi"]).where(lambda x: x > 0)).values
    return out
