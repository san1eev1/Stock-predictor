"""Company filings on NSE (corporate announcements): the first public source of company news.

A company must tell the exchange first (SEBI LODR Regulation 30). Media stories follow minutes
to hours later, often after the price has moved: Hero MotoCorp's Harley-Davidson press release
was filed at 12:00:07 on 26 Nov 2026; the first article came at 12:12 and "shares rise 0.67%"
at 13:01. So filings are used as news, ahead of Google News:

  - turned into headline rows (NEWS_COLS, source "NSE filing") from NSE's own summary of
    the document, so they flow through the same scoring, relevance and news signals;
  - routine filings (analyst-meet schedules, newspaper copies, trading-window notices,
    clarifications of price moves) are dropped;
  - a rule-based tone by filing type until FinBERT scores them on GitHub (the Mac does not
    run FinBERT).

Also the coming events NSE publishes in advance: board meetings with their purpose
(results, dividend, fund raising, bonus, split, buyback) and corporate-action ex-dates.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import pandas as pd

SOURCE = "NSE filing"
ROUTINE = {
    "Analysts/Institutional Investor Meet/Con. Call Updates", "Copy of Newspaper Publication",
    "Trading Window", "Shareholders meeting", "ESOP/ESOS/ESPS", "Spurt in Volume",
    "Price movement", "Certificate under SEBI (Depositories and Participants) Regulations, 2018",
    "Loss of Share Certificates", "Duplicate Share Certificate", "Change in Address",
    "Compliance Certificate", "Statement of deviation(s) or variation(s) under Reg. 32",
    "Disclosure under SEBI Takeover Regulations", "Record Date", "Book Closure",
    "Closure of Trading Window", "Investor Complaints", "Newspaper Publication",
    "Clarification - Financial Results", "Reply to Clarification- Financial results",
    "Change in Registrar and Share Transfer Agent", "Certificate under Reg. 74 (5) of SEBI (DP) Regulations, 2018",
}
# Prior tone by filing type (FinBERT replaces it on GitHub).
TONE = {
    "Bagging/Receiving of orders/contracts": 0.6, "Acquisition": 0.2, "Amalgamation / Merger": 0.1,
    "Resignation of Director/KMP/SMP": -0.3, "Resignation of Statutory Auditors": -0.8,
    "Action(s) taken or orders passed": -0.5, "Action(s) initiated or orders passed": -0.4,
    "Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company": -0.3,
    "Corporate Insolvency Resolution Process": -0.8, "Fraud/Default/Arrest": -0.9,
    "Default by listed entity": -0.8, "Credit Rating": 0.0, "Credit Rating- Revision": 0.0,
    "Fund Raising": -0.1, "Buyback": 0.5, "Bonus": 0.4, "Dividend": 0.2,
}
POSITIVE = re.compile(r"upgrad|wins?\b|bagged|secured|awarded|approv|record (?:high|revenue|profit)|"
                      r"increase in (?:revenue|profit)|launch|commission(?:ed|ing) of|expan", re.I)
NEGATIVE = re.compile(r"downgrad|penalt|\bfined?\b|show cause|\bsearch(?:es)?\b|\braids?\b|fraud|resign|\bdefault|"
                      r"investigation|insider trading|violation|"
                      r"loss widen|decline in (?:revenue|profit)|shut ?down|suspend|fire at|strike",
                      re.I)


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%d-%b-%Y %H:%M:%S")


def fetch(session, start: date, end: date | None = None, symbol: str | None = None) -> pd.DataFrame:
    """Filings from `start` to `end` (IST dates), newest first."""
    from stockpredictor.data import ipo as IPO

    end = end or start
    q = (f"corporate-announcements?index=equities&from_date={start:%d-%m-%Y}"
         f"&to_date={end:%d-%m-%Y}" + (f"&symbol={symbol}" if symbol else ""))
    rows = IPO._get(session, q) or []
    rows = rows if isinstance(rows, list) else rows.get("data", [])
    out = []
    for r in rows:
        try:
            when = _t(r["an_dt"])
        except (KeyError, ValueError, TypeError):
            continue
        summary = (r.get("voiceSummary") or r.get("attchmntText") or "").strip()
        out.append({"symbol": str(r.get("symbol") or "").strip(), "time": when,
                    "category": str(r.get("desc") or "").strip(), "summary": summary,
                    "url": r.get("attchmntFile") or ""})
    return pd.DataFrame(out, columns=["symbol", "time", "category", "summary", "url"])


def headline(category: str, summary: str, limit: int = 260) -> str:
    """A headline-length text: the summary's first sentences (the category if empty)."""
    s = re.sub(r"\s+", " ", summary or "").strip()
    s = re.sub(r"^(?:Pursuant to|In terms of|Please find|We wish to inform|This is to inform)[^.]*\.\s*",
               "", s, flags=re.I)
    if not s or s.lower().startswith("please find"):
        return category
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", s):
        if len(out) + len(sent) > limit and out:
            break
        out = f"{out} {sent}".strip()
    return out[:limit]


def prior_tone(category: str, text: str) -> float:
    t = TONE.get(category, 0.0)
    if POSITIVE.search(text or ""):
        t = max(t, 0.4) if t >= 0 else t + 0.3
    if NEGATIVE.search(text or ""):
        t = min(t, -0.4) if t <= 0 else t - 0.5
    return round(max(-0.9, min(0.9, t)), 2)


ROUTINE_TEXT = re.compile(
    r"postal ballot|e-?voting|registrar and share transfer|\brta\b|form c\b|timely payment of "
    r"interest|payment of interest|dispatch of|annual report|newspaper|trading window|"
    r"loss of share certificate|duplicate share|change in (?:the )?address|relocat|"
    r"corporate presentation|investor presentation for the quarter|transcript|audio recording|"
    r"schedule of (?:analyst|investor)|one-on-one|cut-off date|book closure|"
    r"compliance certificate|reg(?:ulation)?\.? 74|statement of investor complaints|"
    r"certificate under|forbes|award(?:ed)? (?:as|the) best|recogni[sz]ed as|commercial paper|"
    r"allot(?:ted|ment of) .{0,60}(?:esos|esop|employee stock|stock option)|esg rating|"
    r"\baward\b.{0,40}from (?:google|microsoft|aws|sap|oracle|salesforce)", re.I)


def material(df: pd.DataFrame) -> pd.DataFrame:
    """Filings that can move the stock: routine notices and paperwork dropped."""
    if df.empty:
        return df
    keep = ~df["category"].isin(ROUTINE) & ~df["category"].str.contains(
        r"newspaper|trading window|investor meet|con\. call|clarification", case=False)
    keep &= ~df["summary"].fillna("").str.contains(ROUTINE_TEXT)
    return df[keep]


LIVE_FILE = "filings_live.csv"


def save_live(cache_dir, rows: list[dict]) -> int:
    """Today's filings polled by the Mac (rule-based tone), kept beside the synced data."""
    from pathlib import Path

    path = Path(cache_dir) / LIVE_FILE
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(rows[0]) if rows else [])
    new = pd.DataFrame(rows)
    allr = pd.concat([old, new], ignore_index=True).drop_duplicates(["symbol", "title"])
    if not allr.empty:
        allr["published"] = pd.to_datetime(allr["published"], utc=True)
        allr = allr[allr["published"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=4)]
        path.parent.mkdir(parents=True, exist_ok=True)
        allr.assign(published=allr["published"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(path, index=False)
    return len(allr) - len(old)


def load_live(cache_dir) -> pd.DataFrame:
    from pathlib import Path

    path = Path(cache_dir) / LIVE_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df.assign(relevance=1.0)


def merge_news(news: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Stored news + the Mac's live filings (a stored, FinBERT-scored copy wins)."""
    if live is None or live.empty:
        return news
    if news is None or news.empty:
        return live
    seen = set(zip(news["symbol"], news["title"]))
    add = live[[k not in seen for k in zip(live["symbol"], live["title"])]]
    return pd.concat([news, add], ignore_index=True)


def to_news_rows(df: pd.DataFrame, symbols: set[str] | None = None,
                 with_tone: bool = False) -> list[dict]:
    """Material filings as news rows (published in UTC). `with_tone`: rule-based tone (the
    Mac); otherwise left empty for FinBERT on GitHub."""
    df = material(df)
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]
    rows = []
    for r in df.itertuples():
        title = headline(r.category, r.summary)
        utc = pd.Timestamp(r.time).tz_localize("Asia/Kolkata").tz_convert("UTC")
        rows.append({"symbol": r.symbol, "published": utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "source": SOURCE, "title": title, "url": r.url,
                     "sentiment": prior_tone(r.category, r.summary) if with_tone else None})
    return rows


def coming_events(session, start: date, days: int = 14) -> pd.DataFrame:
    """Board meetings (with purpose) and corporate-action ex-dates in the next `days` days."""
    from stockpredictor.data import ipo as IPO

    end = start + timedelta(days=days)
    out = []
    try:
        for r in IPO._get(session, f"event-calendar?index=equities&from_date={start:%d-%m-%Y}"
                                   f"&to_date={end:%d-%m-%Y}") or []:
            out.append({"symbol": r.get("symbol"), "date": r.get("date"), "kind": "Board meeting",
                        "what": r.get("purpose") or "", "detail": r.get("bm_desc") or ""})
    except Exception:
        pass
    try:
        for r in IPO._get(session, f"corporates-corporateActions?index=equities&from_date="
                                   f"{start:%d-%m-%Y}&to_date={end:%d-%m-%Y}") or []:
            if r.get("series") not in (None, "EQ"):
                continue
            out.append({"symbol": r.get("symbol"), "date": r.get("exDate"), "kind": "Ex-date",
                        "what": r.get("subject") or "", "detail": f"record date {r.get('recDate')}"})
    except Exception:
        pass
    df = pd.DataFrame(out, columns=["symbol", "date", "kind", "what", "detail"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["date"]).sort_values(["date", "symbol"])
    return df
