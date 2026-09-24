"""Intraday paper trading: 9:45 picks, stop-loss/target exits, 15:15 square-off.

Uses the same rules (backtest.intraday) as the backtest. Long and short
positions each block their full value from the paper account's cash.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from stockpredictor.backtest import intraday as B
from stockpredictor.costs import DEFAULT_INTRADAY_COSTS, IntradayCosts
from stockpredictor.data import intraday as I
from stockpredictor.data import news as N
from stockpredictor.features import intraday as FI
from stockpredictor.models import intraday as MI
from stockpredictor.paper import engine as E

HORIZON = "intraday"
N_CANDIDATES = 10        # buy and sell candidates shown and judged each day
HISTORY_DAYS = 45        # recent summaries needed for relative-volume features


def get_rules(conn: sqlite3.Connection) -> tuple[B.IntradayRules, bool]:
    s = dict(conn.execute("SELECT key, value FROM app_settings").fetchall())
    d = B.IntradayRules()
    rules = B.IntradayRules(
        n_long=int(s.get("id_n_long", d.n_long)), n_short=int(s.get("id_n_short", d.n_short)),
        stop_loss=float(s.get("id_stop_loss", d.stop_loss)),
        target=float(s.get("id_target", d.target)),
        skip_quantile=float(s.get("id_skip_q", d.skip_quantile)))
    return rules, s.get("id_enabled", "1") == "1"


def todays_features(ctx: E.MarketContext, first30: dict[str, dict], today: pd.Timestamp,
                    store_dir: Path) -> pd.DataFrame:
    """9:45 features for today from live first-30-minute data + recent history."""
    hist = I.load_summaries(store_dir)
    hist = hist[hist["date"] >= today - pd.Timedelta(days=HISTORY_DAYS * 1.6)]
    rows = [{"symbol": s, "date": today, **v, "source": "live"} for s, v in first30.items()]
    summ = pd.concat([hist[hist["date"] < today], pd.DataFrame(rows)], ignore_index=True)
    feats = FI.build(summ, ctx.daily[ctx.daily["date"] < today], ctx.feats,
                     ctx_actions(store_dir))
    return feats[feats["date"] == today]


def ctx_actions(store_dir: Path) -> pd.DataFrame:
    from stockpredictor import store

    return store.load_actions(store_dir)


def recent_strengths(model: MI.IntradayModel, store_dir: Path, ctx: E.MarketContext,
                     rules: B.IntradayRules, days: int = 60) -> pd.Series:
    hist = I.load_summaries(store_dir)
    hist = hist[hist["date"] >= hist["date"].max() - pd.Timedelta(days=days * 1.6)]
    if hist.empty:
        return pd.Series(dtype=float)
    f = FI.build(hist, ctx.daily, ctx.feats, ctx_actions(store_dir))
    f = f.assign(score=model.score(f))
    return f.groupby("date")["score"].apply(lambda s: B.signal_strength(s, rules)).tail(days)


def run_picks(conn: sqlite3.Connection, feats_today: pd.DataFrame, model: MI.IntradayModel,
              prices: dict[str, float], today: pd.Timestamp, rules: B.IntradayRules,
              capital: float, negative_news: set[str] = frozenset(),
              strengths: pd.Series | None = None,
              costs: IntradayCosts = DEFAULT_INTRADAY_COSTS) -> dict:
    """Score, pick, save predictions and open paper trades at live prices."""
    E.ensure_account(conn, capital, HORIZON)
    stamp = f"{today:%Y-%m-%d}"
    conn.executemany("INSERT OR REPLACE INTO intraday_open VALUES (?, ?, ?)",
                     [(stamp, r.symbol, float(r.c30)) for r in feats_today.itertuples()])
    day = feats_today.assign(score=model.score(feats_today).values)
    strength = B.signal_strength(day["score"], rules)
    if rules.skip_quantile > 0 and strengths is not None and len(strengths) >= 20 \
            and strength < strengths.quantile(rules.skip_quantile):
        conn.commit()
        return {"skipped": True, "strength": strength, "picks": []}

    # 10 buy and 10 sell candidates are saved and judged; paper trades are opened for the
    # strongest n_long / n_short of them (fewer, larger positions keep costs down).
    ranked = day.sort_values("score", ascending=False)
    n_buy, n_sell = max(N_CANDIDATES, rules.n_long), max(N_CANDIDATES, rules.n_short)
    longs = ranked[~ranked["symbol"].isin(negative_news)].head(n_buy).assign(side="long")
    shorts = ranked.tail(n_sell).iloc[::-1].assign(side="short")
    shorts = shorts[~shorts["symbol"].isin(longs["symbol"])]
    longs["rank"] = range(1, len(longs) + 1)
    shorts["rank"] = range(1, len(shorts) + 1)
    picks = pd.concat([longs, shorts])
    pct = day["score"].rank(pct=True)
    picks["confidence"] = pct.loc[picks.index].where(picks["side"] == "long",
                                                     1 - pct.loc[picks.index])
    reasons = [why if side == "long" else
               [x for x in why if x.startswith("-")] + [x for x in why if x.startswith("+")]
               for why, side in zip(model.explain(picks), picks["side"])]
    slot = capital / max(1, rules.n_long + rules.n_short)
    sl, tp = B.snap(rules.stop_loss), B.snap(rules.target)
    opened = []
    for (_, p), why in zip(picks.iterrows(), reasons):
        rank = int(p["rank"])
        entry = prices.get(p["symbol"], p["c30"])
        sign = 1 if p["side"] == "long" else -1
        stop = entry * (1 - sign * sl / 100)
        target = entry * (1 + sign * tp / 100) if tp else None
        conn.execute(
            "INSERT OR REPLACE INTO predictions (horizon, date, symbol, direction, confidence, rank, "
            "entry_price, stop_loss, target, reasons, model_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (HORIZON, stamp, p["symbol"], "up" if sign > 0 else "down", float(p["confidence"]),
             rank, entry, stop, target, json.dumps(why), model.train_to))
        limit = rules.n_long if sign > 0 else rules.n_short
        qty = int(slot / entry) if rank <= limit else 0
        if qty <= 0:
            continue
        c = costs.cost("buy" if sign > 0 else "sell", qty * entry)
        conn.execute(
            "INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, costs, "
            "status, reason, stop_loss, target) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)",
            (HORIZON, p["symbol"], p["side"], qty, f"{stamp} 09:45", entry, c, "9:45 pick", stop, target))
        conn.execute("UPDATE paper_accounts SET cash = cash - ? WHERE horizon = ?",
                     (qty * entry + c, HORIZON))
        opened.append(f"{p['side'].upper():5} {p['symbol']} {qty} @ {entry:.2f}")
    conn.commit()
    return {"skipped": False, "strength": strength, "picks": opened}


def open_trades(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'open'",
                        (HORIZON,)).fetchall()


def close_trade(conn, t, price: float, reason: str, stamp: str,
                costs: IntradayCosts = DEFAULT_INTRADAY_COSTS) -> str:
    sign = 1 if t["side"] == "long" else -1
    c = costs.cost("sell" if sign > 0 else "buy", t["qty"] * price)
    gross = sign * (price - t["entry_price"]) * t["qty"]
    conn.execute("UPDATE paper_trades SET exit_time = ?, exit_price = ?, costs = costs + ?, pnl = ?, "
                 "status = 'closed', exit_reason = ? WHERE id = ?",
                 (stamp, price, c, gross - t["costs"] - c, reason, t["id"]))
    conn.execute("UPDATE paper_accounts SET cash = cash + ? WHERE horizon = ?",
                 (t["qty"] * t["entry_price"] + gross - c, HORIZON))
    return f"{t['side']} {t['symbol']} closed @ {price:.2f} ({reason}), P&L {gross - t['costs'] - c:+.0f}"


def check_exits(conn, prices: dict[str, float], stamp: str) -> list[str]:
    log = []
    for t in open_trades(conn):
        p = prices.get(t["symbol"])
        if p is None:
            continue
        long_ = t["side"] == "long"
        if (long_ and p <= t["stop_loss"]) or (not long_ and p >= t["stop_loss"]):
            log.append(close_trade(conn, t, p, "stop-loss", stamp))
        elif t["target"] and ((long_ and p >= t["target"]) or (not long_ and p <= t["target"])):
            log.append(close_trade(conn, t, p, "target", stamp))
    conn.commit()
    return log


def square_off(conn, prices: dict[str, float], stamp: str) -> list[str]:
    log = [close_trade(conn, t, prices.get(t["symbol"], t["entry_price"]), "15:15 square-off", stamp)
           for t in open_trades(conn)]
    conn.commit()
    return log


def evaluate_day(conn, day: str, prices_1515: dict[str, float]) -> int:
    """Judge today's picks by the 9:45 -> 15:15 move, and store the random baseline."""
    opens = dict(conn.execute("SELECT symbol, c30 FROM intraday_open WHERE date = ?", (day,)).fetchall())
    moves = {s: prices_1515[s] / c - 1 for s, c in opens.items() if s in prices_1515}
    up_share = sum(m > 0 for m in moves.values()) / len(moves) if moves else None
    n = 0
    for p in conn.execute("SELECT * FROM predictions WHERE horizon = ? AND date = ?",
                          (HORIZON, day)).fetchall():
        px = prices_1515.get(p["symbol"])
        if px is None:
            continue
        ret = px / p["entry_price"] - 1
        up = p["direction"] == "up"
        conn.execute("UPDATE predictions SET actual_exit = ?, actual_return = ?, correct = ?, "
                     "base_rate = ?, evaluated_at = datetime('now') WHERE id = ?",
                     (px, ret, int(ret > 0 if up else ret < 0),
                      None if up_share is None else (up_share if up else 1 - up_share), p["id"]))
        n += 1
    v = value(conn, prices_1515)
    conn.execute("INSERT OR REPLACE INTO paper_equity VALUES (?, ?, ?, ?, ?)",
                 (HORIZON, day, v["cash"], v["holdings"], v["equity"]))
    conn.commit()
    return n


def value(conn, prices: dict[str, float]) -> dict:
    """Account value with open longs and shorts marked to `prices`."""
    acct = conn.execute("SELECT capital, cash FROM paper_accounts WHERE horizon = ?",
                        (HORIZON,)).fetchone()
    if acct is None:
        return {"capital": 0, "cash": 0, "holdings": 0, "equity": 0, "pnl": 0, "positions": 0}
    held = 0.0
    trades = open_trades(conn)
    for t in trades:
        sign = 1 if t["side"] == "long" else -1
        p = prices.get(t["symbol"], t["entry_price"])
        held += t["qty"] * t["entry_price"] + sign * (p - t["entry_price"]) * t["qty"]
    eq = acct["cash"] + held
    return {"capital": acct["capital"], "cash": acct["cash"], "holdings": held, "equity": eq,
            "pnl": eq - acct["capital"], "positions": len(trades)}
