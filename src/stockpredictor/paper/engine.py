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

import numpy as np
import pandas as pd

from stockpredictor.backtest.portfolio import Position, Rules, decide
from stockpredictor.store import tradable as store_tradable
from stockpredictor.costs import DEFAULT_COSTS, DeliveryCosts
from stockpredictor.data import news as N
from stockpredictor.features import longterm as F
from stockpredictor.models import longterm as M

HORIZON = "longterm"
SHORT_HORIZON = "longterm_short"     # virtual short book for the 10 sell candidates


def book_sign(horizon: str) -> int:
    return -1 if horizon.endswith("_short") else 1
N_PICKS = 10    # long-term: the top 10 buy (up) predictions each day; no down predictions
# Strategy variants raced live on paper: momentum share of the score.
SHADOW_VARIANTS = {"AI model": 0.0, "50/50 blend": 0.5, "Momentum only": 1.0}


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

        from stockpredictor.data import delivery as DL

        daily, indices = store.load_daily(store_dir), store.load_indices(store_dir)
        universe = store.load_universe(store_dir)
        from stockpredictor.nlp import relevance as R

        feats = cached_features(store_dir, daily, indices, universe, DL.load(store_dir))
        return cls(daily, indices, universe, feats,
                   R.add_relevance(N.load_news(store_dir), universe))

    def closes_on(self, date: pd.Timestamp) -> dict[str, float]:
        d = self.daily[self.daily["date"] <= date]
        return d.groupby("symbol")["close"].last().to_dict()

    def nifty_close(self, date: pd.Timestamp) -> float:
        n = self.indices[(self.indices["symbol"] == F.MARKET_INDEX) & (self.indices["date"] <= date)]
        return float(n["close"].iloc[-1])


FEATURE_CACHE_VERSION = 4     # 4: cleaned prices (data/clean.py)


def cached_features(store_dir, daily, indices, universe, delivery=None) -> pd.DataFrame:
    """Build features once per data update and reuse them (~1 s instead of ~15 s per load).
    Stored as compressed float32 (~100 MB); set FEATURE_CACHE=0 in .env to turn it off."""
    import hashlib
    import os
    from pathlib import Path

    from stockpredictor.config import DATA_DIR

    if os.getenv("FEATURE_CACHE", "1") == "0":
        return F.build_features(daily, indices, universe, delivery)
    from stockpredictor.store import local_overlay_path

    files = sorted(Path(store_dir).glob("daily/*.csv")) + sorted(Path(store_dir).glob("indices/*.csv")) \
        + sorted(Path(store_dir).glob("delivery/*.csv")) \
        + [Path(store_dir) / "universe.csv", local_overlay_path("daily"), local_overlay_path("indices")]
    sig = "|".join(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files if f.exists())
    key = hashlib.sha1(f"{FEATURE_CACHE_VERSION}|{sig}".encode()).hexdigest()[:16]
    cache_dir = DATA_DIR / "cache"
    path = cache_dir / f"longterm_features_{key}.pkl.gz"
    if path.exists():
        try:
            return pd.read_pickle(path).copy()
        except Exception:
            path.unlink(missing_ok=True)
    feats = F.build_features(daily, indices, universe, delivery)
    floats = feats.select_dtypes("float64").columns
    feats[floats] = feats[floats].astype(np.float32)
    feats = feats.copy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("longterm_features_*"):
        old.unlink(missing_ok=True)
    feats.to_pickle(path, compression={"method": "gzip", "compresslevel": 1})
    return feats


# --- Account ------------------------------------------------------------------------

def ensure_account(conn: sqlite3.Connection, capital: float, horizon: str = HORIZON) -> None:
    conn.execute("INSERT OR IGNORE INTO paper_accounts (horizon, capital, cash) VALUES (?, ?, ?)",
                 (horizon, capital, capital))
    conn.commit()


def cash_capital(conn: sqlite3.Connection) -> float:
    """The short book starts with the same capital as the long-term buy book."""
    row = conn.execute("SELECT capital FROM paper_accounts WHERE horizon = ?", (HORIZON,)).fetchone()
    return row[0] if row else 100_000


def cash(conn: sqlite3.Connection, horizon: str = HORIZON) -> float:
    return conn.execute("SELECT cash FROM paper_accounts WHERE horizon = ?", (horizon,)).fetchone()[0]


def holdings(conn: sqlite3.Connection, horizon: str = HORIZON) -> dict[str, Position]:
    return {r["symbol"]: Position(r["qty"], r["entry_price"], pd.Timestamp(r["entry_time"]))
            for r in conn.execute("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'open'",
                                  (horizon,))}


def value(conn: sqlite3.Connection, prices: dict[str, float], horizon: str = HORIZON) -> dict:
    pos = holdings(conn, horizon)
    sign = book_sign(horizon)
    hv = sum(p.qty * p.entry_price + sign * (prices.get(s, p.entry_price) - p.entry_price) * p.qty
             for s, p in pos.items())
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
    """Fill queued orders at `prices` (exits first so their cash funds new positions).

    Order side 'buy' opens a position and 'sell' closes it; in the short book that means
    open = short sale, close = buy back (cover).
    """
    sign = book_sign(horizon)
    open_word, close_word = ("BUY ", "SELL") if sign > 0 else ("SHORT", "COVER")
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
            gross = sign * (price - pos["entry_price"]) * pos["qty"]
            c = costs.cost("sell" if sign > 0 else "buy", pos["qty"] * price)
            conn.execute(
                "UPDATE paper_trades SET exit_time = ?, exit_price = ?, costs = costs + ?, "
                "pnl = ? , status = 'closed', exit_reason = ? WHERE id = ?",
                (when, price, c, gross - pos["costs"] - c, o["reason"], pos["id"]))
            _add_cash(conn, horizon, pos["qty"] * pos["entry_price"] + gross - c)
            log.append(f"{close_word} {sym} {pos['qty']} @ {price:.2f} ({o['reason']})")
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
            c = costs.cost("buy" if sign > 0 else "sell", qty * price)
            conn.execute(
                "INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
                "costs, status, reason) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                (horizon, sym, "long" if sign > 0 else "short", qty, when, price, c, o["reason"]))
            _add_cash(conn, horizon, -(qty * price + c))
            log.append(f"{open_word} {sym} {qty} @ {price:.2f}")
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

    # 2. Score today's tradable stocks (current Nifty 250 members).
    active = set(store_tradable(ctx.universe))
    today = ctx.feats[(ctx.feats["date"] == date) & ctx.feats["symbol"].isin(active)].copy()
    if today.empty:
        return {"date": date, "fills": fills, "sells": [], "buys": [], "note": "no data for date"}
    today["score"] = model.score(today).values
    today["confidence"] = today["score"].rank(pct=True)
    # Trades use the steadier trading score (20-day average + momentum) to limit costs.
    scores = M.trading_scores(model, ctx.feats, date, active)
    save_trading_scores(conn, date, scores)

    # 3. News overlay: strongly negative headlines up to this evening.
    summary = N.news_summary(ctx.news, date + pd.Timedelta(hours=18))
    negative = set(summary.loc[summary["strong_negative"].astype(bool), "symbol"])
    severe = set(summary.loc[summary["severe_negative"].astype(bool), "symbol"])

    # 4. Save predictions (top picks = up, bottom = down) with reasons, and the
    #    shadow predictions of each strategy variant for the live race.
    today["close"] = today["symbol"].map(prices).fillna(today["close"]).astype(float)
    save_predictions(conn, today, model, date, ctx.nifty_close(date))
    save_shadow(conn, today, model, date, ctx.nifty_close(date))

    # 5. Decide and queue orders for the next fill.
    rebalance = is_rebalance_day(conn, date, rebalance_mode)
    sells, buys = decide(holdings(conn), scores, prices, rules, rebalance,
                         negative_news=negative, severe_news=severe)
    queue_orders(conn, sells, buys, f"{date:%Y-%m-%d} 18:00")
    shorts, covers = [], []           # long-term paper trading is buy-only
    if rebalance:
        _set_setting(conn, "lt_last_rebalance", f"{date:%Y-%m-%d}")
    _set_setting(conn, "lt_last_decision", f"{date:%Y-%m-%d}")

    v = value(conn, prices)
    for h in (HORIZON,):
        vh = value(conn, prices, h)
        conn.execute("INSERT OR REPLACE INTO paper_equity VALUES (?, ?, ?, ?, ?)",
                     (h, f"{date:%Y-%m-%d}", vh["cash"], vh["holdings"], vh["equity"]))
    conn.commit()
    return {"date": date, "fills": fills, "sells": sells, "buys": buys,
            "shorts": shorts, "covers": covers,
            "rebalance": rebalance, "negative_news": sorted(negative), "value": v}


def save_predictions(conn, today: pd.DataFrame, model: M.LongTermModel,
                     date: pd.Timestamp, nifty: float) -> None:
    ranked = today.sort_values("score", ascending=False).reset_index(drop=True)
    picks = ranked.head(N_PICKS).assign(direction="up")
    reasons = model.explain(picks)
    rows = []
    for (idx, r), why in zip(picks.iterrows(), reasons):
        up = r["direction"] == "up"
        # Confidence = how strongly the model expects this direction.
        conf = float(r["confidence"]) if up else 1 - float(r["confidence"])
        # For sell candidates, lead with the signals that pulled the score down.
        if not up:
            why = [x for x in why if x.startswith("-")] + [x for x in why if x.startswith("+")]
        rows.append((HORIZON, f"{date:%Y-%m-%d}", r["symbol"], r["direction"], conf,
                     int(idx) + 1, float(r["close"]), json.dumps(why), model.train_to, nifty))
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (horizon, date, symbol, direction, confidence, rank, "
        "entry_price, reasons, model_version, nifty_entry, horizon_days) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [r + (M.HORIZON,) for r in rows])
    conn.commit()


def save_trading_scores(conn, date: pd.Timestamp, scores: pd.Series) -> None:
    """Today's trading rank of every tradable stock (used for 'sell now' signals)."""
    rank = scores.rank(ascending=False, method="first")
    conn.execute("DELETE FROM lt_scores")
    conn.executemany("INSERT INTO lt_scores VALUES (?, ?, ?, ?)",
                     [(f"{date:%Y-%m-%d}", s, float(v), int(rank[s])) for s, v in scores.items()])
    conn.commit()


def save_shadow(conn, today: pd.DataFrame, model: M.LongTermModel, date: pd.Timestamp,
                nifty: float) -> None:
    raw = pd.Series(model.model.predict(today[model.features].astype(np.float32)),
                    index=today.index)
    rows = []
    for name, w in SHADOW_VARIANTS.items():
        score = M.blend(raw, today, w)
        ranked = today.assign(s=score.values).sort_values("s", ascending=False)
        for direction, part in (("up", ranked.head(N_PICKS)),):
            rows += [(name, f"{date:%Y-%m-%d}", r.symbol, direction, float(r.close), nifty)
                     for r in part.itertuples()]
    conn.executemany("INSERT OR REPLACE INTO shadow_predictions (variant, date, symbol, direction, "
                     "entry_price, nifty_entry, horizon_days) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     [r + (M.HORIZON,) for r in rows])
    conn.commit()


# --- Evaluation & accuracy ----------------------------------------------------------

def evaluate_predictions(conn: sqlite3.Connection, ctx: MarketContext,
                         horizon_days: int | None = None) -> int:
    """Update running excess return; mark correct/incorrect once each prediction's own
    horizon (1 week now; 3 months for predictions made before the change) has passed."""
    n = _evaluate(conn, ctx, horizon_days, "predictions", "horizon = 'longterm'")
    _evaluate(conn, ctx, horizon_days, "shadow_predictions", "1 = 1")   # live strategy race
    return n


def _evaluate(conn, ctx, horizon_days: int, table: str, where: str) -> int:
    close = ctx.daily.pivot(index="date", columns="symbol", values="close").sort_index()
    nifty = ctx.indices[ctx.indices["symbol"] == F.MARKET_INDEX].set_index("date")["close"]
    nifty = nifty.reindex(close.index).ffill()
    dates = close.index
    done = 0
    for p in conn.execute(f"SELECT * FROM {table} WHERE {where} AND evaluated_at IS NULL").fetchall():
        d = pd.Timestamp(p["date"])
        if d not in dates or p["symbol"] not in close:
            continue
        i = dates.get_loc(d)
        h = horizon_days or (p["horizon_days"] if "horizon_days" in p.keys() and p["horizon_days"]
                             else 63)
        j = min(i + h, len(dates) - 1)
        matured = i + h <= len(dates) - 1
        if j == i:
            continue          # no trading day has passed yet: nothing to measure
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
            f"UPDATE {table} SET actual_exit = ?, actual_return = ?, correct = ?, base_rate = ?, "
            "evaluated_at = ? WHERE id = ?",
            (float(px.iloc[-1]), float(excess), correct, base,
             datetime.now().isoformat(timespec="seconds") if matured else None, p["id"]))
        done += 1
    conn.commit()
    return done


def strategy_race(conn: sqlite3.Connection, last_days: int = 60,
                  horizon_days: int | None = None) -> pd.DataFrame:
    """Live accuracy of each strategy variant over its most recent judged prediction days."""
    df = pd.read_sql("SELECT variant, date, direction, correct, base_rate, actual_return, "
                     "horizon_days FROM shadow_predictions WHERE correct IS NOT NULL", conn)
    if horizon_days is not None:
        df = df[df["horizon_days"].fillna(63) == horizon_days]
    cols = ["variant", "days", "predictions", "accuracy", "random", "avg_excess_up"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    recent = sorted(df["date"].unique())[-last_days:]
    df = df[df["date"].isin(recent)]
    g = df.groupby("variant")
    out = pd.DataFrame({
        "days": g["date"].nunique(), "predictions": g.size(), "accuracy": g["correct"].mean(),
        "random": g["base_rate"].mean(),
        "avg_excess_up": df[df["direction"] == "up"].groupby("variant")["actual_return"].mean(),
    }).reset_index()
    return out[cols].sort_values("accuracy", ascending=False)


def accuracy(conn: sqlite3.Connection, horizon: str = HORIZON) -> dict:
    df = pd.read_sql("SELECT * FROM predictions WHERE horizon = ?", conn, params=(horizon,))
    out = {"total": len(df), "matured": 0}
    m = df.dropna(subset=["correct"])
    if not m.empty:
        up = m["direction"] == "up"
        if horizon.startswith("intraday"):   # ranks count from the strongest pick on each side
            strong = m["rank"] <= 2
            names = ("Long 1-2", "Long 3+", "Short 3+", "Short 1-2")
        else:
            n = m.groupby("date")["rank"].transform("max")
            strong = (up & (m["rank"] <= 3)) | (~up & (m["rank"] > n - 3))
            names = ("Top 1-3", "Top 4-10", "Weakest 4-10", "Weakest 3")
        m = m.assign(bucket=pd.Categorical(
            [(names[0] if s_ else names[1]) if u else (names[3] if s_ else names[2])
             for u, s_ in zip(up, strong)], categories=list(names), ordered=True))
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
