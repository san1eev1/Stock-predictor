"""Data quality report for stored daily prices."""

from __future__ import annotations

import sqlite3
from datetime import date

GAP_DAYS = 10  # calendar days without data that count as a gap


def daily_report(conn: sqlite3.Connection, symbols: list[str]) -> list[dict]:
    report = []
    for sym in symbols:
        dates = [date.fromisoformat(r[0]) for r in conn.execute(
            "SELECT date FROM daily_prices WHERE symbol = ? ORDER BY date", (sym,))]
        zero_vol = conn.execute(
            "SELECT COUNT(*) FROM daily_prices WHERE symbol = ? AND volume = 0", (sym,)
        ).fetchone()[0]
        gaps = [(a, b) for a, b in zip(dates, dates[1:]) if (b - a).days > GAP_DAYS]
        report.append({
            "symbol": sym,
            "rows": len(dates),
            "first": dates[0].isoformat() if dates else None,
            "last": dates[-1].isoformat() if dates else None,
            "zero_volume_days": zero_vol,
            "gaps": len(gaps),
            "largest_gap": max(((b - a).days for a, b in gaps), default=0),
        })
    return report


def format_report(report: list[dict]) -> str:
    lines = [f"{'SYMBOL':<14}{'ROWS':>6}  {'FIRST':<11} {'LAST':<11}{'0-VOL':>6}{'GAPS':>6}"]
    for r in report:
        flag = "  <- no data" if not r["rows"] else ("  <- check" if r["gaps"] else "")
        lines.append(f"{r['symbol']:<14}{r['rows']:>6}  {r['first'] or '-':<11} "
                     f"{r['last'] or '-':<11}{r['zero_volume_days']:>6}{r['gaps']:>6}{flag}")
    empty = sum(1 for r in report if not r["rows"])
    lines.append(f"\n{len(report)} symbols, {empty} without data, "
                 f"{sum(1 for r in report if r['gaps'])} with gaps > {GAP_DAYS} days")
    return "\n".join(lines)


# --- Store-wide checks before training (stockpredictor quality-check) ------------------
# "error" = do not train on this data (the models from the last good run stay in use);
# "warning" = train, but flag it on the dashboard.

STALE_BUSINESS_DAYS = 3        # newest daily prices may be at most this many weekdays old
MIN_STOCK_SHARE = 0.9          # share of the universe that must have the newest day's price
MAX_UNEXPLAINED_MOVE = 0.4     # one-day move after cleaning that is flagged


def check_store(store_dir, today: date | None = None) -> list[dict]:
    """Problems in the market data store: [{"level": "error"|"warning", "check", "detail"}]."""
    import numpy as np
    import pandas as pd

    from stockpredictor import store

    issues: list[dict] = []

    def add(level, check, detail):
        issues.append({"level": level, "check": check, "detail": detail})

    today = today or date.today()
    try:
        daily = store.load_daily(store_dir)
        uni = store.load_universe(store_dir)
    except Exception as exc:                       # missing / unreadable files
        add("error", "data files", f"cannot read market data: {exc}")
        return issues
    active = set(uni.loc[uni["active"] == 1, "symbol"])
    if daily.empty:
        add("error", "daily prices", "no daily prices at all")
        return issues
    last = daily["date"].max().date()
    age = int(np.busday_count(last, today))
    if age > STALE_BUSINESS_DAYS:
        add("error", "stale data", f"newest daily prices are from {last} ({age} weekdays old)")
    latest = daily[daily["date"] == daily["date"].max()]
    have = set(latest["symbol"]) & active
    if active and len(have) < MIN_STOCK_SHARE * len(active):
        add("error", "missing stocks",
            f"newest day ({last}) has prices for only {len(have)} of {len(active)} stocks")
    bad = daily[(daily["close"] <= 0) | daily["close"].isna()]
    if len(bad):
        add("error", "bad prices", f"{len(bad)} rows with zero/negative/missing close")
    dups = int(daily.duplicated(["symbol", "date"]).sum())
    if dups:
        add("error", "duplicates", f"{dups} duplicate stock-days")
    recent = daily[daily["date"] >= daily["date"].max() - pd.Timedelta(days=14)]
    gone = sorted(active - set(recent["symbol"]))
    if gone:
        add("warning", "stocks stopped", f"no prices for 2 weeks: {', '.join(gone[:10])}"
            + (f" (+{len(gone) - 10})" if len(gone) > 10 else ""))
    d = recent.sort_values(["symbol", "date"])
    move = d.groupby("symbol")["close"].pct_change().abs()
    jumps = d.loc[move > MAX_UNEXPLAINED_MOVE, ["symbol", "date"]]
    if len(jumps):
        add("warning", "large moves", "one-day moves over 40% in the last 2 weeks: "
            + ", ".join(f"{s} {t:%d %b}" for s, t in jumps.values[:8]))
    return issues


def format_issues(issues: list[dict]) -> str:
    if not issues:
        return "Data quality: all checks passed"
    return "\n".join(f"Data quality {i['level'].upper()}: {i['check']} - {i['detail']}"
                     for i in issues)
