"""NSE IPOs: the list of past issues (2012 onward), the issues open now / coming up, and each
issue's subscription by investor category (QIB, NII, retail: times subscribed).

Stored in the git store (GitHub data workflow): ipo/issues.csv and ipo/subscription.csv.
Listing-day results come from the bhavcopy (data/bhav.py) when the IPO model is trained.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.nseindia.com/api/"
LANDING = "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36",
           "Referer": LANDING, "Accept": "application/json"}
ISSUE_COLS = ["symbol", "company", "segment", "start", "end", "listing_date", "price_low",
              "price_high", "issue_price"]
SUB_COLS = ["symbol", "qib_x", "nii_x", "rii_x", "total_x", "shares_offered", "updated"]
SEGMENTS = {"Equity": "mainboard", "EQ": "mainboard", "SME": "sme"}


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(LANDING, timeout=20)                     # NSE wants its cookies first
    except requests.RequestException:
        pass
    return s


def _get(s: requests.Session, path: str):
    r = s.get(BASE + path, timeout=30)
    r.raise_for_status()
    return r.json()


def _num(x) -> float | None:
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _band(text) -> tuple[float | None, float | None]:
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(text or "").replace(",", ""))]
    return (nums[0], nums[-1]) if nums else (None, None)


def _date(x):
    return pd.to_datetime(x, format="mixed", dayfirst=True, errors="coerce")


def parse_issues(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows or []:
        sym = r.get("symbol")
        if not sym:
            continue
        lo, hi = _band(r.get("priceRange") or r.get("issuePrice"))
        seg = r.get("securityType") or r.get("series") or ""
        out.append({"symbol": sym.strip(), "company": (r.get("companyName") or r.get("company") or "").strip(),
                    "segment": SEGMENTS.get(seg.strip(), seg.strip().lower()),
                    "start": _date(r.get("ipoStartDate") or r.get("issueStartDate")),
                    "end": _date(r.get("ipoEndDate") or r.get("issueEndDate")),
                    "listing_date": _date(r.get("listingDate")),
                    "price_low": lo, "price_high": hi,
                    "issue_price": _num(r.get("issuePrice")) if r.get("listingDate") else None})
    return pd.DataFrame(out, columns=ISSUE_COLS)


def past_issues(s: requests.Session) -> pd.DataFrame:
    return parse_issues(_get(s, "public-past-issues"))


def open_issues(s: requests.Session) -> pd.DataFrame:
    """Issues open for bidding now plus the ones announced (NSE's current/upcoming lists)."""
    rows = []
    for path in ("ipo-current-issue", "all-upcoming-issues?category=ipo"):
        try:
            data = _get(s, path)
            rows += data if isinstance(data, list) else data.get("data", [])
        except (requests.RequestException, ValueError):
            pass
    return parse_issues(rows).drop_duplicates("symbol")


def parse_subscription(js: dict) -> dict:
    """Times subscribed per category from NSE's 'active category' table."""
    out = {"qib_x": None, "nii_x": None, "rii_x": None, "total_x": None, "shares_offered": None}
    for r in (js or {}).get("dataList", []):
        cat, times = str(r.get("category", "")), _num(r.get("noOfTotalMeant"))
        if cat.startswith("Qualified Institutional"):
            out["qib_x"] = times
        elif cat == "Non Institutional Investors":
            out["nii_x"] = times
        elif cat.startswith("Retail Individual"):
            out["rii_x"] = times
        elif cat == "Total":
            out["total_x"], out["shares_offered"] = times, _num(r.get("noOfShareOffered"))
    m = re.search(r"Updated as on (.+)", str((js or {}).get("updateTime", "")))
    out["updated"] = m.group(1) if m else ""
    return out


def subscription(s: requests.Session, symbol: str) -> dict:
    return {"symbol": symbol, **parse_subscription(_get(s, f"ipo-active-category?symbol={symbol}"))}


def _path(store_dir: Path, name: str) -> Path:
    return Path(store_dir) / "ipo" / name


def load_issues(store_dir: Path) -> pd.DataFrame:
    p = _path(store_dir, "issues.csv")
    if not p.exists():
        return pd.DataFrame(columns=ISSUE_COLS)
    return pd.read_csv(p, parse_dates=["start", "end", "listing_date"])


def load_subscription(store_dir: Path) -> pd.DataFrame:
    p = _path(store_dir, "subscription.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=SUB_COLS)


def update(store_dir: Path, backfill: int = 150, pause: float = 0.7, progress=print) -> dict:
    """Refresh the past-issue list and fetch final subscription for issues that don't have
    it yet (newest first, at most `backfill` per run so a night's run stays short)."""
    s = session()
    issues = past_issues(s)
    folder = Path(store_dir) / "ipo"
    folder.mkdir(parents=True, exist_ok=True)
    old = load_issues(store_dir)
    issues = pd.concat([old, issues]).drop_duplicates("symbol", keep="last") \
        .sort_values("listing_date", ascending=False)
    issues.to_csv(folder / "issues.csv", index=False, date_format="%Y-%m-%d")
    subs = load_subscription(store_dir)
    have = set(subs["symbol"])
    todo = [x for x in issues.loc[issues["segment"].isin(["mainboard", "sme"]), "symbol"]
            if x not in have][:backfill]
    rows, failed = [], 0
    for sym in todo:
        try:
            rows.append(subscription(s, sym))
        except (requests.RequestException, ValueError) as exc:
            failed += 1
            progress(f"subscription {sym}: {exc}")
            if failed >= 10:
                break
        time.sleep(pause)
    if rows:
        subs = pd.concat([subs, pd.DataFrame(rows, columns=SUB_COLS)]).drop_duplicates("symbol", keep="last")
        subs.to_csv(folder / "subscription.csv", index=False)
    return {"issues": len(issues), "new_subscriptions": len(rows), "missing": len(todo) - len(rows)}
