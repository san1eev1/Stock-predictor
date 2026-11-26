"""Stock splits and bonus issues for paper positions held across them.

Prices in the data are split-adjusted (Yahoo), but a paper position keeps the quantity and
price it was bought at. Without this, a 1:8 split looks like an 88% loss and trips the stop.
On a split of ratio v after a position's entry: quantity x v, entry / stop / target / v.
`repair` also re-opens positions a split wrongly stopped out (before this existed).
"""

from __future__ import annotations

import json
import math
import sqlite3

import pandas as pd

KEY = "splits_applied"


def _applied(conn) -> set[str]:
    r = conn.execute("SELECT value FROM app_settings WHERE key = ?", (KEY,)).fetchone()
    return set(json.loads(r[0])) if r and r[0] else set()


def _save(conn, done: set[str]) -> None:
    conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                 (KEY, json.dumps(sorted(done))))


def _qty(q: float, v: float, fractional: bool) -> float:
    return round(q * v, 4) if fractional else float(math.floor(q * v + 1e-9))


def _splits(actions: pd.DataFrame) -> pd.DataFrame:
    s = actions[actions["kind"] == "split"].copy() if not actions.empty else actions
    if s.empty:
        return s
    s["date"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    return s[(s["value"] > 0) & (s["value"] != 1)]


def adjust_open(conn: sqlite3.Connection, actions: pd.DataFrame, today: str,
                fractional: bool = False) -> list[str]:
    """Apply splits dated after a position's entry and up to `today` to open positions."""
    s = _splits(actions)
    if s.empty:
        return []
    done, log = _applied(conn), []
    rows = conn.execute("SELECT id, horizon, symbol, qty, entry_price, entry_time, stop_loss, "
                        "target FROM paper_trades WHERE status = 'open'").fetchall()
    for t in rows:
        for a in s[s["symbol"] == t["symbol"]].itertuples():
            key = f"{t['id']}:{a.date}"
            if key in done or not (str(t["entry_time"])[:10] < a.date <= today):
                continue
            v = float(a.value)
            conn.execute("UPDATE paper_trades SET qty = ?, entry_price = ?, stop_loss = ?, "
                         "target = ? WHERE id = ?",
                         (_qty(t["qty"], v, fractional), t["entry_price"] / v,
                          None if t["stop_loss"] is None else t["stop_loss"] / v,
                          None if t["target"] is None else t["target"] / v, t["id"]))
            done.add(key)
            log.append(f"{t['symbol']} split {v:g}:1 on {a.date}: {t['horizon']} position adjusted")
    _save(conn, done)
    conn.commit()
    return log


def repair(conn: sqlite3.Connection, actions: pd.DataFrame, fractional: bool = False,
           max_drop: float = 0.4) -> list[str]:
    """Positions sold after a split at a 'loss' that the split explains (price fell by more
    than `max_drop` and by about the split ratio): re-opened at split-adjusted terms, the
    sale reversed in the cash account."""
    s = _splits(actions)
    if s.empty:
        return []
    done, log = _applied(conn), []
    rows = conn.execute("SELECT * FROM paper_trades WHERE status = 'closed' AND exit_price > 0 "
                        "AND exit_price < entry_price * ?", (1 - max_drop,)).fetchall()
    for t in rows:
        for a in s[s["symbol"] == t["symbol"]].itertuples():
            key = f"{t['id']}:{a.date}"
            if key in done or not (str(t["entry_time"])[:10] < a.date <= str(t["exit_time"])[:10]):
                continue
            v = float(a.value)
            if abs(t["exit_price"] * v / t["entry_price"] - 1) > max_drop:
                continue                       # a real crash, not only the split
            proceeds = t["qty"] * t["exit_price"] - (t["costs"] or 0) / 2
            conn.execute("UPDATE paper_accounts SET cash = cash - ? WHERE horizon = ?",
                         (proceeds, t["horizon"]))
            conn.execute("UPDATE paper_trades SET status = 'open', exit_time = NULL, "
                         "exit_price = NULL, pnl = NULL, exit_reason = NULL, qty = ?, "
                         "entry_price = ?, stop_loss = ?, target = ?, costs = ? WHERE id = ?",
                         (_qty(t["qty"], v, fractional), t["entry_price"] / v,
                          None if t["stop_loss"] is None else t["stop_loss"] / v,
                          None if t["target"] is None else t["target"] / v,
                          (t["costs"] or 0) / 2, t["id"]))
            done.add(key)
            log.append(f"{t['symbol']}: sale after the {v:g}:1 split on {a.date} reversed; "
                       f"position re-opened ({t['horizon']})")
    _save(conn, done)
    conn.commit()
    return log
