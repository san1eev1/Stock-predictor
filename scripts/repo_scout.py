"""Weekly repo scout (GitHub Actions: repo-scout.yml). Searches GitHub for quant / Indian-
market / backtesting / ML-for-trading repositories, and appends the ones not seen before to
research/repo_watchlist.md (name, stars, licence, last push, description). It only *records*
candidates; each one is studied and tested by hand (research/repository_matrix.md) before
anything is adopted.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

WATCHLIST = Path("research/repo_watchlist.md")
STUDIED = Path("research/repository_matrix.md")
QUERIES = [
    "quantitative trading machine learning", "backtesting python", "alpha factors stocks",
    "nse india algo trading", "indian stock market prediction", "zerodha kite backtest",
    "lightgbm stock ranking", "walk forward validation trading", "market microstructure python",
    "portfolio optimization python", "options backtesting india", "purged cross validation",
]
MIN_STARS = 150


def search(q: str) -> list[dict]:
    since = (date.today() - timedelta(days=365)).isoformat()
    url = ("https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": f"{q} pushed:>{since} stars:>={MIN_STARS}", "sort": "stars", "per_page": 15}))
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               **({"Authorization": f"Bearer {os.environ['GH_TOKEN']}"}
                                                  if os.environ.get("GH_TOKEN") else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("items", [])


def main() -> None:
    text = WATCHLIST.read_text() if WATCHLIST.exists() else (
        "# Repo watchlist (auto-appended weekly by scripts/repo_scout.py)\n\n"
        "Status: new → studied (see repository_matrix.md) → technique tested (experiment_log.md).\n\n"
        "| Found | Repo | Stars | Licence | Last push | About | Status |\n"
        "|---|---|---|---|---|---|---|\n")
    seen = set(re.findall(r"\[([\w.-]+/[\w.-]+)\]", text))
    seen |= set(re.findall(r"([\w.-]+/[\w.-]+) @", STUDIED.read_text())) if STUDIED.exists() else set()
    rows = []
    for q in QUERIES:
        try:
            items = search(q)
        except Exception as exc:
            print(f"search failed for {q!r}: {exc}")
            continue
        for it in items:
            name = it["full_name"]
            if name in seen:
                continue
            seen.add(name)
            about = (it.get("description") or "").replace("|", "/")[:110]
            lic = (it.get("license") or {}).get("spdx_id") or "none"
            rows.append(f"| {date.today()} | [{name}]({it['html_url']}) | {it['stargazers_count']} "
                        f"| {lic} | {it['pushed_at'][:10]} | {about} | new |")
    WATCHLIST.write_text(text + "".join(r + "\n" for r in rows))
    print(f"{len(rows)} new repositories added to {WATCHLIST}")


if __name__ == "__main__":
    main()
