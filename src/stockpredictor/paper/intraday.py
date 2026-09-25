"""Intraday paper trading: 9:45 picks, stop-loss/target exits, 12:30 square-off.

Uses the same rules (backtest.intraday) as the backtest. Long and short
positions each block their full value from the paper account's cash. Every day
starts fresh with the full capital (Rs 1 lakh); each day's result is kept and
compared day by day (daily_results), and the judged picks feed back into training.
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

HORIZON = "intraday"                 # 9:45 -> 12:30 book
CLOSE_HORIZON = MI.CLOSE.horizon     # 9:45 -> 15:15 book ("until the close")
BOOKS = (HORIZON, CLOSE_HORIZON)
N_CANDIDATES = 10        # buy and sell candidates shown and judged each day
HISTORY_DAYS = 45        # recent summaries needed for relative-volume features


def learned_rules(horizon: str = HORIZON) -> dict | None:
    """Trading rules chosen by profit on GitHub for this book (trainer.tune_rules)."""
    t = next((t for t in MI.TARGETS if t.horizon == horizon), None)
    path = t.model_dir / "rules.json" if t is not None else None
    try:
        return json.loads(path.read_text()) if path is not None and path.exists() else None
    except (OSError, ValueError):
        return None


def get_rules(conn: sqlite3.Connection, horizon: str = HORIZON) -> tuple[B.IntradayRules, bool]:
    """Settings page values; the numbers of buys/sells there are the MAXIMUM. Rules learned
    by profit on GitHub (fewer trades, skipping weak days, stop/target) apply within them."""
    s = dict(conn.execute("SELECT key, value FROM app_settings").fetchall())
    d = B.IntradayRules()
    rules = B.IntradayRules(
        n_long=int(s.get("id_n_long", d.n_long)), n_short=int(s.get("id_n_short", d.n_short)),
        stop_loss=float(s.get("id_stop_loss", d.stop_loss)),
        target=float(s.get("id_target", d.target)),
        skip_quantile=float(s.get("id_skip_q", d.skip_quantile)))
    learned = (learned_rules(horizon) or {}).get("rules")
    if learned and s.get("id_learned_rules", "1") == "1":
        rules = B.IntradayRules(
            n_long=min(rules.n_long, int(learned["n_long"])),
            n_short=min(rules.n_short, int(learned["n_short"])),
            stop_loss=float(learned["stop_loss"]), target=float(learned["target"]),
            skip_quantile=float(learned["skip_quantile"]),
            min_prob=float(learned.get("min_prob", 0.0)))
    return rules, s.get("id_enabled", "1") == "1"


def todays_features(ctx: E.MarketContext, first30: dict[str, dict], today: pd.Timestamp,
                    store_dir: Path) -> pd.DataFrame:
    """9:45 features for today from live first-30-minute data + recent history."""
    hist = I.load_summaries(store_dir)
    hist = hist[hist["date"] >= today - pd.Timedelta(days=HISTORY_DAYS * 1.6)]
    rows = [{"symbol": s, "date": today, **v, "source": "live"} for s, v in first30.items()]
    summ = pd.concat([hist[hist["date"] < today], pd.DataFrame(rows)], ignore_index=True)
    try:                           # today's pre-open auction (NSE shows the current day)
        from stockpredictor.data import preopen

        live_po = preopen.fetch()
        live_po = live_po[live_po["date"] == today]
    except Exception:
        live_po = None
    feats = FI.build_for(summ, ctx, store_dir, daily=ctx.daily[ctx.daily["date"] < today],
                         live_preopen=live_po)
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
    f = FI.build_for(hist, ctx, store_dir)
    if f.empty:                    # no usable recent days: no skipping decision possible
        return pd.Series(dtype=float)
    f = f.assign(score=model.score(f))
    return f.groupby("date")["score"].apply(lambda s: B.signal_strength(s, rules)).tail(days)


def run_picks(conn: sqlite3.Connection, feats_today: pd.DataFrame, model: MI.IntradayModel,
              prices: dict[str, float], today: pd.Timestamp, rules: B.IntradayRules,
              capital: float, negative_news: set[str] = frozenset(),
              strengths: pd.Series | None = None,
              costs: IntradayCosts = DEFAULT_INTRADAY_COSTS, horizon: str = HORIZON) -> dict:
    """Score, pick, save predictions and open paper trades at live prices (in `horizon`'s
    own Rs 1 lakh book)."""
    E.ensure_account(conn, capital, horizon)
    start_day(conn, capital, horizon)
    stamp = f"{today:%Y-%m-%d}"
    conn.executemany("INSERT OR REPLACE INTO intraday_open VALUES (?, ?, ?)",
                     [(stamp, r.symbol, float(r.c30)) for r in feats_today.itertuples()])
    day = feats_today.assign(score=model.score(feats_today).values)
    strength = B.signal_strength(day["score"], rules)
    # Weak day: no trades, but the candidates are still saved and judged (the models learn).
    skipped = bool(rules.skip_quantile > 0 and strengths is not None and len(strengths) >= 20
                   and strength < strengths.quantile(rules.skip_quantile))

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
    take = take_probs(day, feats_today, horizon) if rules.min_prob > 0 else {}
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
            (horizon, stamp, p["symbol"], "up" if sign > 0 else "down", float(p["confidence"]),
             rank, entry, stop, target, json.dumps(why), model.train_to))
        limit = 0 if skipped else rules.n_long if sign > 0 else rules.n_short
        # 'take this trade?' filter: skip picks unlikely to make money after costs
        prob = take.get((p["symbol"], p["side"]))
        taken = rules.min_prob <= 0 or (prob is not None and prob >= rules.min_prob)
        qty = int(slot / entry) if rank <= limit and taken else 0
        if qty <= 0:
            continue
        c = costs.cost("buy" if sign > 0 else "sell", qty * entry)
        conn.execute(
            "INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, costs, "
            "status, reason, stop_loss, target) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)",
            (horizon, p["symbol"], p["side"], qty, f"{stamp} 09:45", entry, c, "9:45 pick", stop, target))
        conn.execute("UPDATE paper_accounts SET cash = cash - ? WHERE horizon = ?",
                     (qty * entry + c, horizon))
        opened.append(f"{p['side'].upper():5} {p['symbol']} {qty} @ {entry:.2f}")
    conn.commit()
    return {"skipped": skipped, "strength": strength, "picks": opened}


def take_probs(day: pd.DataFrame, feats_today: pd.DataFrame, horizon: str) -> dict:
    """{(symbol, side): chance the trade makes money after costs} from the book's
    'take this trade?' model (trained on GitHub); {} if there is none."""
    from stockpredictor.models import take as K

    t = next((t for t in MI.TARGETS if t.horizon == horizon), None)
    loaded = K.load(t.model_dir) if t is not None else None
    if loaded is None:
        return {}
    c = K.candidates(day[["symbol", "date", "score"]], feats_today)
    return dict(zip(zip(c["symbol"], c["side"]), K.predict(*loaded, c)))


def exit_comparison(conn: sqlite3.Connection, closes: pd.DataFrame) -> pd.DataFrame:
    """Live, judged days only: the traded picks at 12:30 vs the same picks held to the close,
    and the close model's own picks at the close. `closes`: daily symbol/date/close."""
    q = ("SELECT date, symbol, direction, entry_price, actual_return, correct, base_rate "
         "FROM predictions WHERE horizon = ? AND correct IS NOT NULL")
    trade = pd.read_sql(q, conn, params=(HORIZON,))
    close_model = pd.read_sql(q, conn, params=(MI.CLOSE.horizon,))
    rows = []

    def add(name, df, ret, right, base):
        if df.empty:
            return
        sign = df["direction"].map({"up": 1, "down": -1})
        rows.append({"How": name, "Days": df["date"].nunique(), "Picks": len(df),
                     "Accuracy": float(right.mean()),
                     "Random": None if base is None else float(base.mean()),
                     "Avg move in our favour": float((sign * ret).mean())})

    if not trade.empty:
        add("12:30 model, exit at 12:30 (traded)", trade, trade["actual_return"],
            trade["correct"], trade["base_rate"])
        c = closes.assign(date=closes["date"].dt.strftime("%Y-%m-%d"))
        held = trade.merge(c[["symbol", "date", "close"]], on=["symbol", "date"])
        if not held.empty:
            ret = held["close"] / held["entry_price"] - 1
            right = ((ret > 0) & (held["direction"] == "up")) | ((ret < 0) & (held["direction"] == "down"))
            add("12:30 model, same picks held to the close", held, ret, right.astype(int),
                None)                        # no random baseline stored for the close here
    if not close_model.empty:
        add("Close model, exit at the close", close_model, close_model["actual_return"],
            close_model["correct"], close_model["base_rate"])
    return pd.DataFrame(rows)


def start_day(conn: sqlite3.Connection, capital: float, horizon: str = HORIZON) -> bool:
    """A new day starts with the full capital again (only when nothing is still open)."""
    if open_trades(conn, horizon):
        return False
    conn.execute("UPDATE paper_accounts SET capital = ?, cash = ? WHERE horizon = ?",
                 (capital, capital, horizon))
    conn.commit()
    return True


def daily_results(conn: sqlite3.Connection, capital: float = 100_000,
                  horizon: str = HORIZON) -> pd.DataFrame:
    """One row per trading day: paper result on a fresh `capital` and how the day's 10 buy /
    10 sell picks did (9:45 -> 12:30), next to picking stocks at random."""
    trades = pd.read_sql("SELECT substr(entry_time, 1, 10) AS date, side, pnl, status "
                         "FROM paper_trades WHERE horizon = ?", conn, params=(horizon,))
    preds = pd.read_sql("SELECT date, direction, correct, base_rate FROM predictions "
                        "WHERE horizon = ? AND correct IS NOT NULL", conn, params=(horizon,))
    days = sorted(set(trades["date"]) | set(preds["date"]))
    rows = []
    for d in days:
        t = trades[(trades["date"] == d) & (trades["status"] == "closed")]
        p = preds[preds["date"] == d]
        up, dn = p[p["direction"] == "up"], p[p["direction"] == "down"]
        pnl = float(t["pnl"].sum()) if len(t) else 0.0
        rows.append({
            "date": d, "trades": len(t), "won": int((t["pnl"] > 0).sum()), "pnl": pnl,
            "pnl_pct": pnl / capital, "end_value": capital + pnl,
            "buy_right": int(up["correct"].sum()), "buy_n": len(up),
            "sell_right": int(dn["correct"].sum()), "sell_n": len(dn),
            "accuracy": float(p["correct"].mean()) if len(p) else None,
            "random": float(p["base_rate"].mean()) if len(p) and p["base_rate"].notna().any()
            else None,
            "open": int((trades[(trades["date"] == d)]["status"] == "open").sum())})
    return pd.DataFrame(rows)


def open_trades(conn: sqlite3.Connection, horizon: str = HORIZON) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'open'",
                        (horizon,)).fetchall()


def close_trade(conn, t, price: float, reason: str, stamp: str,
                costs: IntradayCosts = DEFAULT_INTRADAY_COSTS) -> str:
    sign = 1 if t["side"] == "long" else -1
    c = costs.cost("sell" if sign > 0 else "buy", t["qty"] * price)
    gross = sign * (price - t["entry_price"]) * t["qty"]
    conn.execute("UPDATE paper_trades SET exit_time = ?, exit_price = ?, costs = costs + ?, pnl = ?, "
                 "status = 'closed', exit_reason = ? WHERE id = ?",
                 (stamp, price, c, gross - t["costs"] - c, reason, t["id"]))
    conn.execute("UPDATE paper_accounts SET cash = cash + ? WHERE horizon = ?",
                 (t["qty"] * t["entry_price"] + gross - c, t["horizon"]))
    return f"{t['side']} {t['symbol']} closed @ {price:.2f} ({reason}), P&L {gross - t['costs'] - c:+.0f}"


DAILY_LOSS_LIMIT = 0.015   # a book that is down 1.5% of its capital today stops for the day


def check_exits(conn, prices: dict[str, float], stamp: str,
                books: tuple[str, ...] = BOOKS) -> list[str]:
    log = []
    for t in [t for h in books for t in open_trades(conn, h)]:
        p = prices.get(t["symbol"])
        if p is None:
            continue
        long_ = t["side"] == "long"
        if (long_ and p <= t["stop_loss"]) or (not long_ and p >= t["stop_loss"]):
            log.append(close_trade(conn, t, p, "stop-loss", stamp))
        elif t["target"] and ((long_ and p >= t["target"]) or (not long_ and p <= t["target"])):
            log.append(close_trade(conn, t, p, "target", stamp))
    # Daily loss limit: close everything left in a book once its day's loss reaches the limit.
    for h in books:
        v = value(conn, prices, h)
        if v["positions"] and v["capital"] and v["pnl"] <= -DAILY_LOSS_LIMIT * v["capital"]:
            for t in open_trades(conn, h):
                log.append(close_trade(conn, t, prices.get(t["symbol"], t["entry_price"]),
                                       "daily loss limit", stamp))
    conn.commit()
    return log


def square_off(conn, prices: dict[str, float], stamp: str, horizon: str = HORIZON) -> list[str]:
    label = "12:30 square-off" if horizon == HORIZON else "15:15 square-off"
    log = [close_trade(conn, t, prices.get(t["symbol"], t["entry_price"]), label, stamp)
           for t in open_trades(conn, horizon)]
    conn.commit()
    return log


def evaluate_day(conn, day: str, exit_prices: dict[str, float],
                 horizon: str = HORIZON) -> int:
    """Judge `horizon`'s picks by the move from 9:45 to its square-off (12:30 or 15:15) at
    `exit_prices`, and store the random baseline."""
    opens = dict(conn.execute("SELECT symbol, c30 FROM intraday_open WHERE date = ?", (day,)).fetchall())
    moves = {s: exit_prices[s] / c - 1 for s, c in opens.items() if s in exit_prices}
    up_share = sum(m > 0 for m in moves.values()) / len(moves) if moves else None
    n = 0
    for p in conn.execute("SELECT * FROM predictions WHERE horizon = ? AND date = ?",
                          (horizon, day)).fetchall():
        px = exit_prices.get(p["symbol"])
        if px is None:
            continue
        ret = px / p["entry_price"] - 1
        up = p["direction"] == "up"
        conn.execute("UPDATE predictions SET actual_exit = ?, actual_return = ?, correct = ?, "
                     "base_rate = ?, evaluated_at = datetime('now') WHERE id = ?",
                     (px, ret, int(ret > 0 if up else ret < 0),
                      None if up_share is None else (up_share if up else 1 - up_share), p["id"]))
        n += 1
    if horizon in BOOKS:                     # book the paper result too
        v = value(conn, exit_prices, horizon)
        conn.execute("INSERT OR REPLACE INTO paper_equity VALUES (?, ?, ?, ?, ?)",
                     (horizon, day, v["cash"], v["holdings"], v["equity"]))
    conn.commit()
    return n


def value(conn, prices: dict[str, float], horizon: str = HORIZON) -> dict:
    """Account value with open longs and shorts marked to `prices`."""
    acct = conn.execute("SELECT capital, cash FROM paper_accounts WHERE horizon = ?",
                        (horizon,)).fetchone()
    if acct is None:
        return {"capital": 0, "cash": 0, "holdings": 0, "equity": 0, "pnl": 0, "positions": 0}
    held = 0.0
    trades = open_trades(conn, horizon)
    for t in trades:
        sign = 1 if t["side"] == "long" else -1
        p = prices.get(t["symbol"], t["entry_price"])
        held += t["qty"] * t["entry_price"] + sign * (p - t["entry_price"]) * t["qty"]
    eq = acct["cash"] + held
    return {"capital": acct["capital"], "cash": acct["cash"], "holdings": held, "equity": eq,
            "pnl": eq - acct["capital"], "positions": len(trades)}


def preview(ctx: E.MarketContext, store_dir: Path) -> tuple[pd.DataFrame, pd.Timestamp] | None:
    """The 10 buy / 10 sell picks the model would have made on the latest completed day,
    with how they actually did (9:45 -> 12:30). Trains a model first if there is none."""
    from stockpredictor import store
    from stockpredictor.models import trainer as T

    feats = T.intraday_feats(ctx, store_dir)
    if feats.empty or feats["date"].nunique() < MI.MIN_TRAIN_DAYS:
        return None
    day = feats["date"].max()
    # Honest preview: a model trained only on days before the preview day.
    model = MI.IntradayModel.train(feats[feats["date"] < day], {**MI.current_params(), "n_seeds": 1})
    if MI.model_path(MI.TRADE) is None:
        MI.IntradayModel.train(feats).save()          # first real model for live picks
    t = feats[(feats["date"] == day) & feats["symbol"].isin(store.tradable(ctx.universe))].copy()
    t["score"] = model.score(t).values
    t = t.sort_values("score", ascending=False)
    pct = t["score"].rank(pct=True)
    longs = t.head(N_CANDIDATES).assign(side="long", confidence=pct)
    shorts = t.tail(N_CANDIDATES).iloc[::-1].assign(side="short", confidence=1 - pct)
    picks = pd.concat([longs, shorts])
    why = model.explain(picks)
    picks["reasons"] = [w if side == "long" else
                        [x for x in w if x.startswith("-")] + [x for x in w if x.startswith("+")]
                        for w, side in zip(why, picks["side"])]
    sign = picks["side"].map({"long": 1, "short": -1})
    picks["move"] = sign * (picks[I.EXIT_COL] / picks["c30"] - 1)
    up_share = float((t[I.EXIT_COL] > t["c30"]).mean())      # random-pick baseline for the day
    picks["baseline"] = picks["side"].map({"long": up_share, "short": 1 - up_share})
    return picks[["symbol", "side", "confidence", "c30", I.EXIT_COL, "move", "reasons",
                  "baseline"]], day
