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
