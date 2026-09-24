"""Long-term paper trading: daily decisions, order fills, valuation, evaluation.

Flow (all state in the local SQLite database):
  1. After the close, `run_decision` scores all stocks, saves predictions and
     queues buy/sell orders using the shared portfolio rules.
  2. Orders are filled at the next available price: by the live monitor
     during market hours, or at the next day's close by the next decision run
     (the same next-day timing as the backtest).
  3. `evaluate_predictions` scores each prediction once its 3-month window ends.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from stockpredictor.backtest.portfolio import Position, Rules, decide
from stockpredictor.costs import DEFAULT_COSTS, DeliveryCosts
from stockpredictor.data import news as N
from stockpredictor.features import longterm as F
from stockpredictor.models import longterm as M

HORIZON = "longterm"
N_PICKS = 10    # predictions saved per side (up / down) each day


@dataclass
class MarketContext:
    daily: pd.DataFrame
    indices: pd.DataFrame
    universe: pd.DataFrame
    feats: pd.DataFrame
    news: pd.DataFrame

    @classmethod
    def load(cls, store_dir) -> "MarketContext":
        from stockpredictor import store

        daily, indices = store.load_daily(store_dir), store.load_indices(store_dir)
        universe = store.load_universe(store_dir)
        return cls(daily, indices, universe, F.build_features(daily, indices, universe),
                   N.load_news(store_dir))

    def closes_on(self, date: pd.Timestamp) -> dict[str, float]:
        d = self.daily[self.daily["date"] <= date]
        return d.groupby("symbol")["close"].last().to_dict()

    def nifty_close(self, date: pd.Timestamp) -> float:
        n = self.indices[(self.indices["symbol"] == F.MARKET_INDEX) & (self.indices["date"] <= date)]
        return float(n["close"].iloc[-1])


# --- Account ------------------------------------------------------------------------

def ensure_account(conn: sqlite3.Connection, capital: float, horizon: str = HORIZON) -> None:
    conn.execute("INSERT OR IGNORE INTO paper_accounts (horizon, capital, cash) VALUES (?, ?, ?)",
                 (horizon, capital, capital))
    conn.commit()


def cash(conn: sqlite3.Connection, horizon: str = HORIZON) -> float:
    return conn.execute("SELECT cash FROM paper_accounts WHERE horizon = ?", (horizon,)).fetchone()[0]


def holdings(conn: sqlite3.Connection, horizon: str = HORIZON) -> dict[str, Position]:
    return {r["symbol"]: Position(r["qty"], r["entry_price"], pd.Timestamp(r["entry_time"]))
            for r in conn.execute("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'open'",
                                  (horizon,))}


def value(conn: sqlite3.Connection, prices: dict[str, float], horizon: str = HORIZON) -> dict:
    pos = holdings(conn, horizon)
    hv = sum(p.qty * prices.get(s, p.entry_price) for s, p in pos.items())
    c = cash(conn, horizon)
    capital = conn.execute("SELECT capital FROM paper_accounts WHERE horizon = ?",
                           (horizon,)).fetchone()[0]
    return {"cash": c, "holdings": hv, "equity": c + hv, "capital": capital,
            "pnl": c + hv - capital, "positions": len(pos)}


# --- Orders -------------------------------------------------------------------------

def queue_orders(conn, sells: list[tuple[str, str]], buys: list[str], when: str,
                 horizon: str = HORIZON) -> None:
    pending = {(r["symbol"], r["side"]) for r in conn.execute(
        "SELECT symbol, side FROM paper_orders WHERE horizon = ? AND status = 'pending'", (horizon,))}
    rows = [(horizon, when, s, "sell", reason) for s, reason in sells if (s, "sell") not in pending]
    rows += [(horizon, when, s, "buy", "top rank") for s in buys if (s, "buy") not in pending]
    conn.executemany("INSERT INTO paper_orders (horizon, created, symbol, side, reason) "
                     "VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()


def fill_pending(conn, prices: dict[str, float], when: str, rules: Rules = Rules(),
                 costs: DeliveryCosts = DEFAULT_COSTS, horizon: str = HORIZON) -> list[str]:
    """Fill queued orders at `prices` (sells first so their cash funds buys)."""
    log = []
    orders = conn.execute(
        "SELECT * FROM paper_orders WHERE horizon = ? AND status = 'pending' "
        "ORDER BY side = 'buy', id", (horizon,)).fetchall()
    for o in orders:
        sym, price = o["symbol"], prices.get(o["symbol"])
        if price is None:
            continue
        pos = conn.execute("SELECT * FROM paper_trades WHERE horizon = ? AND symbol = ? "
                           "AND status = 'open'", (horizon, sym)).fetchone()
        if o["side"] == "sell":
            if pos is None:
                _set_order(conn, o["id"], "cancelled", when, None)
                continue
            proceeds = pos["qty"] * price
            c = costs.cost("sell", proceeds)
            conn.execute(
                "UPDATE paper_trades SET exit_time = ?, exit_price = ?, costs = costs + ?, "
                "pnl = ? , status = 'closed', exit_reason = ? WHERE id = ?",
                (when, price, c, (price - pos["entry_price"]) * pos["qty"] - pos["costs"] - c,
                 o["reason"], pos["id"]))
            _add_cash(conn, horizon, proceeds - c)
            log.append(f"SELL {sym} {pos['qty']} @ {price:.2f} ({o['reason']})")
        else:
            n_open = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE horizon = ? "
                                  "AND status = 'open'", (horizon,)).fetchone()[0]
            if pos is not None or n_open >= rules.n_hold:
                _set_order(conn, o["id"], "cancelled", when, None)
                continue
            equity = value(conn, prices, horizon)["equity"]
            qty = int(min(equity / rules.n_hold, cash(conn, horizon)) / (price * 1.003))
            if qty <= 0:
                _set_order(conn, o["id"], "cancelled", when, None)
                continue
            c = costs.cost("buy", qty * price)
            conn.execute(
                "INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
                "costs, status, reason) VALUES (?, ?, 'long', ?, ?, ?, ?, 'open', ?)",
                (horizon, sym, qty, when, price, c, o["reason"]))
            _add_cash(conn, horizon, -(qty * price + c))
            log.append(f"BUY  {sym} {qty} @ {price:.2f}")
        _set_order(conn, o["id"], "filled", when, price)
    conn.commit()
    return log


def _set_order(conn, oid, status, when, price):
    conn.execute("UPDATE paper_orders SET status = ?, fill_time = ?, fill_price = ? WHERE id = ?",
                 (status, when, price, oid))


def _add_cash(conn, horizon, amount):
    conn.execute("UPDATE paper_accounts SET cash = cash + ? WHERE horizon = ?", (amount, horizon))


# --- Daily decision -----------------------------------------------------------------

def _get_setting(conn, key, default=None):
    row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def _set_setting(conn, key, val):
    conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, str(val)))


def is_rebalance_day(conn, date: pd.Timestamp, mode: str) -> bool:
    if mode == "daily":
        return True
    last = _get_setting(conn, "lt_last_rebalance")
    return last is None or (date - pd.Timestamp(last)).days >= 7


def run_decision(conn: sqlite3.Connection, ctx: MarketContext, model: M.LongTermModel,
                 date: pd.Timestamp | None = None, rules: Rules = Rules(),
                 rebalance_mode: str = "weekly") -> dict:
    """Official end-of-day decision for `date` (default: latest data)."""
    date = pd.Timestamp(date or ctx.feats["date"].max())
    prices = ctx.closes_on(date)

    # 1. Orders from the previous decision are filled at today's close if the
    #    live monitor did not fill them earlier in the day.
    fills = fill_pending(conn, prices, f"{date:%Y-%m-%d} 15:30", rules)

    # 2. Score today's tradable stocks (current Nifty 100 members).
    active = set(ctx.universe.loc[ctx.universe["active"] == 1, "symbol"])
    today = ctx.feats[(ctx.feats["date"] == date) & ctx.feats["symbol"].isin(active)].copy()
    if today.empty:
        return {"date": date, "fills": fills, "sells": [], "buys": [], "note": "no data for date"}
    today["score"] = model.score(today).values
    today["confidence"] = today["score"].rank(pct=True)
    scores = today.set_index("symbol")["score"]

    # 3. News overlay: strongly negative headlines up to this evening.
    summary = N.news_summary(ctx.news, date + pd.Timedelta(hours=18))
    negative = set(summary.loc[summary["strong_negative"].astype(bool), "symbol"])

    # 4. Save predictions (top picks = up, bottom = down) with reasons.
    save_predictions(conn, today, model, date, ctx.nifty_close(date))

    # 5. Decide and queue orders for the next fill.
    rebalance = is_rebalance_day(conn, date, rebalance_mode)
    sells, buys = decide(holdings(conn), scores, prices, rules, rebalance, negative_news=negative)
    queue_orders(conn, sells, buys, f"{date:%Y-%m-%d} 18:00")
    if rebalance:
        _set_setting(conn, "lt_last_rebalance", f"{date:%Y-%m-%d}")
    _set_setting(conn, "lt_last_decision", f"{date:%Y-%m-%d}")

    v = value(conn, prices)
    conn.execute("INSERT OR REPLACE INTO paper_equity VALUES (?, ?, ?, ?, ?)",
                 (HORIZON, f"{date:%Y-%m-%d}", v["cash"], v["holdings"], v["equity"]))
    conn.commit()
    return {"date": date, "fills": fills, "sells": sells, "buys": buys,
            "rebalance": rebalance, "negative_news": sorted(negative), "value": v}


def save_predictions(conn, today: pd.DataFrame, model: M.LongTermModel,
                     date: pd.Timestamp, nifty: float) -> None:
    ranked = today.sort_values("score", ascending=False).reset_index(drop=True)
    picks = pd.concat([ranked.head(N_PICKS).assign(direction="up"),
                       ranked.tail(N_PICKS).assign(direction="down")])
    reasons = model.explain(picks)
    rows = []
    for (idx, r), why in zip(picks.iterrows(), reasons):
        rows.append((HORIZON, f"{date:%Y-%m-%d}", r["symbol"], r["direction"],
                     float(r["confidence"]), int(idx) + 1, float(r["close"]),
                     json.dumps(why), model.train_to, nifty))
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (horizon, date, symbol, direction, confidence, rank, "
        "entry_price, reasons, model_version, nifty_entry) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows)
    conn.commit()


# --- Evaluation & accuracy ----------------------------------------------------------

def evaluate_predictions(conn: sqlite3.Connection, ctx: MarketContext,
                         horizon_days: int = M.HORIZON) -> int:
    """Update running excess return; mark correct/incorrect once 3 months have passed."""
    close = ctx.daily.pivot(index="date", columns="symbol", values="close").sort_index()
    nifty = ctx.indices[ctx.indices["symbol"] == F.MARKET_INDEX].set_index("date")["close"]
    nifty = nifty.reindex(close.index).ffill()
    dates = close.index
    done = 0
    for p in conn.execute("SELECT * FROM predictions WHERE horizon = ? AND evaluated_at IS NULL",
                          (HORIZON,)).fetchall():
        d = pd.Timestamp(p["date"])
        if d not in dates or p["symbol"] not in close:
            continue
        i = dates.get_loc(d)
        j = min(i + horizon_days, len(dates) - 1)
        matured = i + horizon_days <= len(dates) - 1
        px = close[p["symbol"]].iloc[: j + 1].dropna()
        if px.empty:
            continue
        excess = (px.iloc[-1] / p["entry_price"] - 1) - (nifty.iloc[j] / nifty.iloc[i] - 1)
        correct = None
        base = None
        if matured:
            correct = int(excess > 0) if p["direction"] == "up" else int(excess < 0)
            all_ex = (close.iloc[j] / close.iloc[i] - 1) - (nifty.iloc[j] / nifty.iloc[i] - 1)
            share_up = float((all_ex.dropna() > 0).mean())
            base = share_up if p["direction"] == "up" else 1 - share_up
        conn.execute(
            "UPDATE predictions SET actual_exit = ?, actual_return = ?, correct = ?, base_rate = ?, "
            "evaluated_at = ? WHERE id = ?",
            (float(px.iloc[-1]), float(excess), correct, base,
             datetime.now().isoformat(timespec="seconds") if matured else None, p["id"]))
        done += 1
    conn.commit()
    return done


def accuracy(conn: sqlite3.Connection, horizon: str = HORIZON) -> dict:
    df = pd.read_sql("SELECT * FROM predictions WHERE horizon = ?", conn, params=(horizon,))
    out = {"total": len(df), "matured": 0}
    m = df.dropna(subset=["correct"])
    if not m.empty:
        m = m.assign(bucket=pd.cut(m["confidence"], [0, 0.2, 0.8, 0.9, 1.0],
                                   labels=["bottom 20%", "middle", "80-90%", "top 10%"]))
        out.update({
            "matured": len(m),
            "accuracy": m["correct"].mean(),
            "accuracy_up": m.loc[m["direction"] == "up", "correct"].mean(),
            "accuracy_down": m.loc[m["direction"] == "down", "correct"].mean(),
            "random_baseline": m["base_rate"].mean(),
            "by_confidence": m.groupby("bucket", observed=True)["correct"].mean().to_dict(),
            "rolling": m.groupby("date")["correct"].mean().rolling(30, min_periods=1).mean(),
        })
    trades = pd.read_sql("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'closed'",
                         conn, params=(horizon,))
    out["closed_trades"] = len(trades)
    if len(trades):
        out["profitable_trades"] = (trades["pnl"] > 0).mean()
    return out
