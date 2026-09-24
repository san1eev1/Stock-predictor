"""Company news from Google News RSS (free), stored as monthly CSV files.

Headlines are scored with FinBERT (see nlp/sentiment.py) by the GitHub
Actions news job, so the Mac never needs the model. Historical news is not
available for free, so news is used as a live overlay (skip buys / flag
exits on strongly negative news) until enough history exists to train on.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests

RSS_URL = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"}
NEWS_COLS = ["symbol", "published", "source", "title", "url", "sentiment"]
STRONG_NEGATIVE = -0.6

_SUFFIXES = re.compile(r"\b(ltd|limited|corporation|corp|company|co|inc|india)\b\.?", re.I)


def company_query(name: str) -> str:
    """'Reliance Industries Ltd.' -> '"Reliance Industries" share'."""
    clean = _SUFFIXES.sub("", name).replace("&", "and").strip(" .,")
    clean = re.sub(r"\s+", " ", clean)
    return f'"{clean}" share'


def parse_rss(xml_text: str, symbol: str) -> list[dict]:
    items = []
    for item in ET.fromstring(xml_text).iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        pub = item.findtext("pubDate")
        if not title or not pub:
            continue
        items.append({
            "symbol": symbol,
            "published": parsedate_to_datetime(pub).astimezone(timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source,
            "title": title,
            "url": (item.findtext("link") or "").strip(),
            "sentiment": None,
        })
    return items


def fetch_company_news(symbol: str, name: str) -> list[dict]:
    resp = requests.get(RSS_URL.format(q=quote_plus(company_query(name))),
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return parse_rss(resp.text, symbol)


def fetch_all(universe: pd.DataFrame, pause: float = 0.5) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    for sym, name in universe[["symbol", "name"]].itertuples(index=False):
        try:
            rows += fetch_company_news(sym, name or sym)
        except Exception as exc:  # one bad feed must not stop the rest
            errors.append(f"{sym}: {exc}")
        time.sleep(pause)
    return rows, errors


# --- CSV storage ----------------------------------------------------------------

def load_news(store_dir: Path) -> pd.DataFrame:
    files = sorted((store_dir / "news").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=NEWS_COLS)
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df


def append_news(store_dir: Path, rows: list[dict]) -> pd.DataFrame:
    """Merge new headlines into monthly files; returns only the rows that were new."""
    if not rows:
        return pd.DataFrame(columns=NEWS_COLS)
    new = pd.DataFrame(rows, columns=NEWS_COLS)
    new["published"] = pd.to_datetime(new["published"], utc=True)
    new = new.drop_duplicates(["symbol", "title"])
    old = load_news(store_dir)
    if not old.empty:
        seen = set(zip(old["symbol"], old["title"]))
        new = new[[k not in seen for k in zip(new["symbol"], new["title"])]]
    if new.empty:
        return new
    allrows = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    allrows["published"] = pd.to_datetime(allrows["published"], utc=True)
    month = allrows["published"].dt.strftime("%Y-%m")
    for m in new["published"].dt.strftime("%Y-%m").unique():
        part = allrows[month == m].sort_values(["published", "symbol", "title"])
        out = part.assign(published=part["published"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        (store_dir / "news").mkdir(parents=True, exist_ok=True)
        out.to_csv(store_dir / "news" / f"{m}.csv", index=False, float_format="%.4f")
    return new


def score_missing(store_dir: Path, scorer) -> int:
    """Fill in sentiment for headlines that have none yet. Returns count scored."""
    scored = 0
    for path in sorted((store_dir / "news").glob("*.csv")):
        df = pd.read_csv(path)
        todo = df["sentiment"].isna()
        if todo.any():
            df.loc[todo, "sentiment"] = scorer(df.loc[todo, "title"].tolist())
            df.to_csv(path, index=False, float_format="%.4f")
            scored += int(todo.sum())
    return scored


# --- Features ---------------------------------------------------------------------

def news_summary(news: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Per-stock news signals using headlines published up to `asof` (UTC)."""
    cols = ["symbol", "news_count_7d", "sent_mean_7d", "sent_min_3d", "strong_negative"]
    if news.empty:
        return pd.DataFrame(columns=cols)
    asof = pd.Timestamp(asof)
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    recent = news[(news["published"] <= asof)
                  & (news["published"] > asof - pd.Timedelta(days=7))]
    last3 = recent[recent["published"] > asof - pd.Timedelta(days=3)]
    g7, g3 = recent.groupby("symbol")["sentiment"], last3.groupby("symbol")["sentiment"]
    out = pd.DataFrame({
        "news_count_7d": g7.size(),
        "sent_mean_7d": g7.mean(),
        "sent_min_3d": g3.min(),
    })
    out["strong_negative"] = out["sent_min_3d"] <= STRONG_NEGATIVE
    return out.reset_index()[cols]


def latest_headlines(news: pd.DataFrame, symbol: str, n: int = 5) -> pd.DataFrame:
    return news[news["symbol"] == symbol].sort_values("published", ascending=False).head(n)
