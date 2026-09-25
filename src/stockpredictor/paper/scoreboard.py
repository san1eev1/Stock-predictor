"""Accuracy right now: live score of today's picks plus everything judged so far.

Shown in the dashboard and printed to the terminal every 15 minutes while the
program runs.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from stockpredictor.models import longterm as M


def _live_prices(conn) -> dict[str, float]:
    return {r[0]: r[1] for r in conn.execute("SELECT symbol, price FROM live_prices")}


def intraday_today(conn: sqlite3.Connection, day: str, prices: dict[str, float] | None = None) -> dict:
    """How today's 10 buy / 10 sell intraday picks are doing since 9:45."""
    prices = prices if prices is not None else _live_prices(conn)
    preds = conn.execute("SELECT symbol, direction, entry_price, actual_exit FROM predictions "
                         "WHERE horizon = 'intraday' AND date = ?", (day,)).fetchall()
    out = {"picks": len(preds), "buy_n": 0, "buy_right": 0, "sell_n": 0, "sell_right": 0}
    for p in preds:
        px = p["actual_exit"] if p["actual_exit"] is not None else prices.get(p["symbol"])
        if px is None:
            continue
        up = px > p["entry_price"]
        if p["direction"] == "up":
            out["buy_n"] += 1
            out["buy_right"] += int(up)
        else:
            out["sell_n"] += 1
            out["sell_right"] += int(px < p["entry_price"])
    opens = dict(conn.execute("SELECT symbol, c30 FROM intraday_open WHERE date = ?", (day,)).fetchall())
    moves = [prices[s] > c for s, c in opens.items() if s in prices]
    out["random_up"] = sum(moves) / len(moves) if moves else None
    n = out["buy_n"] + out["sell_n"]
    out["right"] = (out["buy_right"] + out["sell_right"]) / n if n else None
    if out["random_up"] is not None and n:
        out["random"] = (out["buy_n"] * out["random_up"]
                         + out["sell_n"] * (1 - out["random_up"])) / n
    else:
        out["random"] = None
    return out


def judged(conn: sqlite3.Connection, horizon: str, horizon_days: int | None = None,
           last_days: int | None = None) -> dict:
    """Accuracy of all judged predictions (optionally only the most recent days)."""
    q = "SELECT date, correct, base_rate FROM predictions WHERE horizon = ? AND correct IS NOT NULL"
    args: list = [horizon]
    if horizon_days is not None:
        q += " AND horizon_days = ?"
        args.append(horizon_days)
    df = pd.read_sql(q, conn, params=args)
    if last_days and not df.empty:
        df = df[df["date"].isin(sorted(df["date"].unique())[-last_days:])]
    if df.empty:
        return {"n": 0, "days": 0}
    return {"n": len(df), "days": int(df["date"].nunique()), "accuracy": float(df["correct"].mean()),
            "random": float(df["base_rate"].mean())}


def longterm_open(conn: sqlite3.Connection) -> dict:
    """1-week predictions still running: how many are on track so far (updated daily)."""
    df = pd.read_sql("SELECT direction, actual_return FROM predictions WHERE horizon = 'longterm' "
                     "AND correct IS NULL AND horizon_days = ? AND actual_return IS NOT NULL",
                     conn, params=(M.HORIZON,))
    if df.empty:
        return {"n": 0}
    on_track = ((df["direction"] == "up") & (df["actual_return"] > 0)) | \
        ((df["direction"] == "down") & (df["actual_return"] < 0))
    return {"n": len(df), "on_track": int(on_track.sum())}


def compute(conn: sqlite3.Connection, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    day = f"{now:%Y-%m-%d}"
    return {
        "time": now.strftime("%Y-%m-%d %H:%M"),
        "intraday_today": intraday_today(conn, day),
        "intraday_judged": judged(conn, "intraday"),
        "intraday_recent": judged(conn, "intraday", last_days=10),
        "longterm_open": longterm_open(conn),
        "longterm_judged": judged(conn, "longterm", horizon_days=M.HORIZON),
    }


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def lines(sb: dict) -> list[str]:
    """Plain-text scoreboard for the terminal."""
    it, ij, lo, lj = (sb["intraday_today"], sb["intraday_judged"], sb["longterm_open"],
                      sb["longterm_judged"])
    out = [f"===== Accuracy now ({sb['time']}) ====="]
    if it["buy_n"] + it["sell_n"]:
        out.append(f"Intraday today : buys {it['buy_right']}/{it['buy_n']} up, sells "
                   f"{it['sell_right']}/{it['sell_n']} down -> {_pct(it['right'])} right "
                   f"(random picks: {_pct(it['random'])})")
    else:
        out.append("Intraday today : no picks yet (made at 9:46 on trading days)")
    out.append(f"Intraday judged: {_pct(ij.get('accuracy'))} right on {ij['n']} picks over "
               f"{ij['days']} days (random: {_pct(ij.get('random'))})" if ij["n"]
               else "Intraday judged: none yet")
    out.append(f"Long-term open : {lo['on_track']}/{lo['n']} picks on track (judged after 1 week)"
               if lo["n"] else "Long-term open : no running 1-week predictions yet")
    out.append(f"Long-term judged: {_pct(lj.get('accuracy'))} right on {lj['n']} picks over "
               f"{lj['days']} days (random: {_pct(lj.get('random'))})" if lj["n"]
               else "Long-term judged: first results one week after the first prediction")
    return out
