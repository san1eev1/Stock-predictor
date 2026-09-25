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
STRONG_NEGATIVE = -0.6      # a headline this negative counts as "bad news"
RELEVANT = 0.8              # ... if its learned relevance is at least this (1.0 = average)
MIN_BAD_HEADLINES = 2       # a flag needs a cluster of bad news within 3 days ...
MAX_MEAN_3D = -0.3          # ... and a negative average tone
SEVERE_BAD_HEADLINES = 3    # "severe" (enough to sell a holding) needs more evidence
SEVERE_MEAN_3D = -0.6
# Headlines that only report a price move; the model already sees prices.
PRICE_MOVE = re.compile(
    r"(?:share|stock|price)s?\b.*\b(?:fall|fell|drop|declin|slip|slump|down|tumbl|plung|los|"
    r"crash|sink|sank|tank)|top (?:loser|gainer)|\b(?:sensex|nifty)\b|52[- ]?w(?:ee)?k|trades? flat|"
    r"price target|target (?:price|cut)|stock (?:price|analysis)", re.I)

# Tips and recommendations, not news: we don't copy other people's buy/sell calls, the
# model makes its own. Dropped when fetching and ignored in stored files.
TIPS = re.compile(
    r"stocks? to (?:buy|sell|watch|trade|add|avoid|bet on)|stocks? in focus|shares in focus|"
    r"buy or sell|should you (?:buy|sell|hold|invest)|top (?:stock )?picks?|stock picks|"
    r"trading (?:ideas|calls|setup)|technical picks|\b\d+ (?:\w+ )?picks\b|multibagger|"
    r"^(?:buy|sell|accumulate|reduce|hold|add|neutral)\b.*\btarget|"
    r"\b(?:buy|sell|accumulate|reduce|hold|add)\b[^:]*;\s*target|"
    r"target (?:price|of rs)|price target|(?:raise|cut|hike)s? target|"
    r"(?:buy|sell|hold|outperform|underperform|overweight|underweight|neutral) rating|"
    r"(?:upgrade|downgrade)s?\b|brokerages? (?:see|bullish|bearish|recommend)|recommend|"
    r"(?:price|share) (?:forecast|prediction)|outlook to 20\d\d|"
    r"is (?:important|attractive) to you|presents an opportunity|"
    r"top (?:gainers?|losers?)|gainers (?:and|&) losers", re.I)

_SUFFIXES = re.compile(r"\b(ltd|limited|corporation|corp|inc)\b\.?", re.I)
# Headlines about foreign-listed namesakes (e.g. Cummins Inc on NYSE vs Cummins India).
FOREIGN = re.compile(r"\((?:NYSE|NASDAQ|OTC|LSE|TSX)\s*:", re.I)


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
        if not title or not pub or FOREIGN.search(title) or TIPS.search(title):
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
    return df[~df["title"].astype(str).str.contains(TIPS)].reset_index(drop=True)


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
    cols = ["symbol", "news_count_7d", "sent_mean_7d", "sent_min_3d", "bad_news_3d",
            "strong_negative", "severe_negative"]
    if news.empty:
        return pd.DataFrame(columns=cols)
    asof = pd.Timestamp(asof)
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    recent = news[(news["published"] <= asof)
                  & (news["published"] > asof - pd.Timedelta(days=7))].copy()
    # Learned relevance (nlp/relevance.py): headlines that historically came with real moves
    # in their stock count more; lists, namesakes and noise count less.
    recent["w"] = recent["relevance"] if "relevance" in recent else 1.0
    recent = recent[recent["sentiment"].notna()]
    last3 = recent[recent["published"] > asof - pd.Timedelta(days=3)]
    g7, g3 = recent.groupby("symbol"), last3.groupby("symbol")["sentiment"]
    out = pd.DataFrame({
        "news_count_7d": g7.size(),
        "sent_mean_7d": g7.apply(lambda x: (x["sentiment"] * x["w"]).sum() / x["w"].sum()
                                 if x["w"].sum() > 0 else x["sentiment"].mean()),
        "sent_min_3d": g3.min(),
    })
    events = last3[~last3["title"].str.contains(PRICE_MOVE) & ~last3["title"].str.contains(FOREIGN)
                   & (last3["w"] >= RELEVANT)]
    ge = events.groupby("symbol")["sentiment"]
    out["bad_news_3d"] = ge.apply(lambda x: int((x <= STRONG_NEGATIVE).sum()))
    out["bad_news_3d"] = out["bad_news_3d"].fillna(0).astype(int)
    mean3 = ge.mean().reindex(out.index)
    out["strong_negative"] = (out["bad_news_3d"] >= MIN_BAD_HEADLINES) & (mean3 <= MAX_MEAN_3D)
    out["severe_negative"] = (out["bad_news_3d"] >= SEVERE_BAD_HEADLINES) & (mean3 <= SEVERE_MEAN_3D)
    return out.reset_index(names="symbol")[cols]


def latest_headlines(news: pd.DataFrame, symbol: str, n: int = 5) -> pd.DataFrame:
    return news[news["symbol"] == symbol].sort_values("published", ascending=False).head(n)
