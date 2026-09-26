"""Safety net for live paper trading: kill-switches, health alerts, decay monitor, and the
paper-vs-simulation check.

Kill-switches stop NEW trades (predictions are still made, saved and judged, so the models
keep learning); open positions keep their normal exits. Everything here is cheap (database
queries and small files) - it runs on the Mac.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

from stockpredictor.config import SHARED_MODELS_DIR

STALE_WEEKDAYS = 3              # daily prices older than this: no new trades
LIVE_PRICE_MAX_AGE_MIN = 10     # in market hours, live prices older than this: no new trades
INTRADAY_WEEK_LOSS = 0.03       # an intraday book down 3% this week pauses until Monday
LONGTERM_DRAWDOWN = 0.25        # long-term book 25% below its peak: no new buys
DECAY_DAYS_INTRADAY = 20        # judged days for the decay check
DECAY_WEEKS_LONGTERM = 8
DECAY_MARGIN = 0.03             # live accuracy this far below random = decayed
MODEL_MAX_AGE_DAYS = 3          # GitHub models older than this: warning
PAPER_SIM_MAX_GAP = 0.003       # paper vs simulated trade result differing by >0.3% of value


def _setting(conn, key):
    row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


# --- Kill-switches ------------------------------------------------------------------------

def stale_data(daily_last: date | pd.Timestamp | None, today: date) -> str | None:
    if daily_last is None:
        return "no market data"
    age = int(np.busday_count(pd.Timestamp(daily_last).date(), today))
    return f"market data is {age} weekdays old" if age > STALE_WEEKDAYS else None


def stale_live_prices(conn: sqlite3.Connection, now: datetime) -> str | None:
    ts = conn.execute("SELECT MAX(ts) FROM live_prices").fetchone()[0]
    if not ts:
        return "no live prices"
    age = (now.replace(tzinfo=None) - datetime.fromisoformat(ts)).total_seconds() / 60
    return f"live prices are {age:.0f} minutes old" if age > LIVE_PRICE_MAX_AGE_MIN else None


def broken_scores(scores: pd.Series) -> str | None:
    """Predictions that cannot be right: many missing, or all (nearly) the same."""
    if scores.empty:
        return "no predictions"
    if scores.isna().mean() > 0.05:
        return f"{scores.isna().mean():.0%} of predictions are missing"
    if float(scores.std()) < 1e-9:
        return "all predictions are identical"
    return None


def intraday_week_loss(conn: sqlite3.Connection, horizon: str, today: date,
                       capital: float) -> str | None:
    monday = today - pd.Timedelta(days=today.weekday())
    pnl = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM paper_trades WHERE horizon = ? AND "
                       "status = 'closed' AND substr(entry_time, 1, 10) >= ?",
                       (horizon, f"{monday:%Y-%m-%d}")).fetchone()[0]
    if capital and pnl <= -INTRADAY_WEEK_LOSS * capital:
        return f"this week's loss Rs {-pnl:,.0f} is over {INTRADAY_WEEK_LOSS:.0%} of capital"
    return None


def longterm_drawdown(conn: sqlite3.Connection, horizon: str = "longterm") -> str | None:
    eq = pd.read_sql("SELECT date, equity FROM paper_equity WHERE horizon = ? ORDER BY date",
                     conn, params=(horizon,))
    if len(eq) < 2:
        return None
    dd = eq["equity"].iloc[-1] / eq["equity"].cummax().iloc[-1] - 1
    return f"the book is {-dd:.0%} below its peak" if dd <= -LONGTERM_DRAWDOWN else None


def intraday_kill(conn, now: datetime, daily_last, horizon: str, capital: float,
                  prices: dict | None = None, n_stocks: int = 0) -> str | None:
    """Why no new intraday trades may be opened now (None = trading allowed). `prices`: the
    live prices just fetched for the picks (else the saved live-price table is checked)."""
    if prices is not None:
        live = None if len(prices) >= 0.5 * max(1, n_stocks) else \
            f"live prices for only {len(prices)} of {n_stocks} stocks"
    else:
        live = stale_live_prices(conn, now)
    return (stale_data(daily_last, now.date()) or live
            or intraday_week_loss(conn, horizon, now.date(), capital))


# --- Decay monitor -------------------------------------------------------------------------

def decay(conn: sqlite3.Connection, horizon: str) -> str | None:
    """Live picks clearly worse than random picks over the recent judged period."""
    p = pd.read_sql("SELECT date, correct, base_rate, direction FROM predictions WHERE "
                    "horizon = ? AND correct IS NOT NULL", conn, params=(horizon,))
    if p.empty:
        return None
    days = sorted(p["date"].unique())
    n = DECAY_DAYS_INTRADAY if horizon.startswith("intraday") else DECAY_WEEKS_LONGTERM * 5
    if len(days) < n:
        return None
    r = p[p["date"].isin(days[-n:])]
    base = r["base_rate"].to_numpy(float)        # already the random rate for that direction
    edge = r["correct"].mean() - np.nanmean(base)
    if edge < -DECAY_MARGIN:
        return (f"{horizon}: live picks {r['correct'].mean():.0%} right vs {np.nanmean(base):.0%} "
                f"for random picks over the last {len(days[-n:])} judged days")
    return None


# --- Paper vs simulation --------------------------------------------------------------------

def paper_vs_simulation(conn: sqlite3.Connection, day: str, summaries: pd.DataFrame,
                        rules, horizon: str) -> dict | None:
    """Re-run the day's closed paper trades through the backtest simulator (same entry,
    quantity, stop and target, the day's intraday summary). A big gap means live paper
    trading and the history tests disagree - a bug or an unrealistic assumption."""
    from stockpredictor.backtest import intraday as B
    from stockpredictor.data import intraday as I

    t = pd.read_sql("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'closed' AND "
                    "substr(entry_time, 1, 10) = ?", conn, params=(horizon, day))
    s = summaries[summaries["date"] == pd.Timestamp(day)].set_index("symbol")
    if t.empty or s.empty:
        return None
    exit_col = I.EXIT_COL if horizon == "intraday" else "px_1515"
    minutes = I.EXIT_MINUTES if horizon == "intraday" else I.WINDOW_MINUTES
    gaps = []
    for r in t.itertuples():
        if r.symbol not in s.index:
            continue
        row = s.loc[r.symbol].copy()
        row["c30"] = r.entry_price                  # the paper trade's own entry
        exit_, _ = B.replay(row, r.side, rules, exit_col, minutes)
        sim, _ = B.trade_pnl(r.side, r.qty, r.entry_price, exit_)
        gaps.append(abs(sim - r.pnl) / (r.qty * r.entry_price))
    if not gaps:
        return None
    return {"day": day, "trades": len(gaps), "avg_gap": float(np.mean(gaps)),
            "max_gap": float(np.max(gaps))}


# --- Health for the dashboard banner ----------------------------------------------------------

def health(conn: sqlite3.Connection, now: datetime, daily_last=None) -> list[dict]:
    """[{"level": "error"|"warning", "text"}] - shown at the top of every dashboard page."""
    items: list[dict] = []
    add = lambda lvl, txt: items.append({"level": lvl, "text": txt})  # noqa: E731
    if daily_last is not None and (w := stale_data(daily_last, now.date())):
        add("error", f"Stale data: {w} - no new trades until it is updated")
    q = SHARED_MODELS_DIR / "data_quality.json"
    if q.exists():
        try:
            for i in json.loads(q.read_text()).get("issues", []):
                add(i["level"], f"Data quality ({i['check']}): {i['detail']}")
        except ValueError:
            pass
    meta = SHARED_MODELS_DIR / "longterm" / "meta.json"
    if meta.exists():
        try:
            trained = datetime.fromisoformat(json.loads(meta.read_text())["trained_at"])
            if (now.replace(tzinfo=None) - trained).days > MODEL_MAX_AGE_DAYS:
                add("warning", f"GitHub models are {(now.replace(tzinfo=None) - trained).days} "
                               "days old - are the evening training runs working?")
        except (ValueError, KeyError):
            pass
    gh = _setting(conn, "gh_last_train")
    if gh and gh.startswith("failure"):
        add("error", f"Last GitHub training run failed ({gh.split('|', 1)[-1]}) - see GitHub Actions")
    for h in ("longterm", "intraday", "intraday_close"):
        if (msg := decay(conn, h)):
            add("warning", f"Possible model decay: {msg}")
    for key, label in (("kill_intraday", "Intraday trading paused"),
                       ("kill_intraday_close", "Intraday (until close) trading paused"),
                       ("kill_longterm", "Long-term: no new buys")):
        v = _setting(conn, key)
        if v and v.split("|", 1)[0] == f"{now:%Y-%m-%d}":
            add("error", f"{label}: {v.split('|', 1)[1]}")
    ps = _setting(conn, "paper_sim_check")
    if ps:
        try:
            r = json.loads(ps)
            if r.get("avg_gap", 0) > PAPER_SIM_MAX_GAP:
                add("warning", f"Paper trades on {r['day']} differ from the simulator by "
                               f"{r['avg_gap']:.2%} on average - live and backtest disagree")
        except ValueError:
            pass
    return items
