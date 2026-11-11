"""Experiment E-008 (GitHub, research.yml): the mid-day review vs holding.

Same picks both ways: each day the 'until close' model's out-of-sample scores (walk-forward:
each month scored by a model trained only on earlier days) give the best 10 buys and 10
sells at 9:45. Prices: Yahoo 5-minute candles (last ~60 days; Angel One not used).
  HOLD:   learned stop-loss / target, otherwise square-off at 15:15.
  REVIEW: the same, plus every 30 minutes from 10:15 (paper.intraday.review): cut trades that
          lost or moved < 0.3%, add the freed buying power to the 1-2 winners up >= 0.5%
          (<= 40% of the book in one stock; each top-up stopped at its own entry).
₹10,000 book with 5x intraday margin, real costs (0.37% round trip).
Fixed rule: REVIEW is kept if its daily P&L beats HOLD with paired t >= 2 over >= 30 days;
if it is worse with t <= -2 it is switched off; otherwise inconclusive (keep collecting).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor import store
from stockpredictor.backtest import intraday as B
from stockpredictor.costs import DEFAULT_INTRADAY_COSTS as COSTS
from stockpredictor.data import intraday as I
from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T
from stockpredictor.paper import engine as E
from stockpredictor.paper import intraday as PI

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
ctx = E.MarketContext.load(d, years=3)
feats = T.intraday_feats(ctx, d, MI.CLOSE)
scores = MI.walk_forward(feats, params=MI.current_params(MI.CLOSE), last_days=80)
print(f"out-of-sample scores: {scores['date'].nunique()} days", flush=True)

syms = sorted(set(store.tradable(ctx.universe)))
bars = I.yahoo_bars(syms, period="60d", interval="5m")
print(f"Yahoo 5-minute candles: {len(bars)} stocks", flush=True)
px = {}
for s, b in bars.items():
    b = b.assign(at=b["ts"] + pd.Timedelta(minutes=5))           # price at the bar's end
    px[s] = b.set_index("at")["close"]

learned = (PI.learned_rules(MI.CLOSE.horizon) or {}).get("rules") or {}
SL = B.snap(float(learned.get("stop_loss", 1.0))) / 100
TP = B.snap(float(learned.get("target", 2.0))) / 100
print(f"stop {SL:.2%}, target {TP:.2%}", flush=True)
POWER = 10_000 * PI.LEVERAGE
times = pd.date_range("2000-01-01 09:50", "2000-01-01 15:15", freq="5min").strftime("%H:%M")
REVIEWS = set(PI.review_times(MI.CLOSE.horizon))


def day_pnl(day: pd.Timestamp, picks: pd.DataFrame, review: bool) -> float | None:
    def price(sym, hhmm):
        s = px.get(sym)
        return None if s is None else s.get(pd.Timestamp(f"{day:%Y-%m-%d} {hhmm}"))

    slot = POWER / len(picks)
    pos = []                                  # [sym, sign, qty, entry, stop, target, entry_time]
    for r in picks.itertuples():
        p = price(r.symbol, "09:45")
        if p is None or p <= 0:
            continue
        sign = 1 if r.side == "long" else -1
        q = int(slot / p)
        if q > 0:
            pos.append([r.symbol, sign, q, p, p * (1 - sign * SL), p * (1 + sign * TP), "09:45"])
    if not pos:
        return None
    pnl = -sum(COSTS.cost("buy" if x[1] > 0 else "sell", x[2] * x[3]) for x in pos)

    def close(x, p):
        return x[1] * (p - x[3]) * x[2] - COSTS.cost("sell" if x[1] > 0 else "buy", x[2] * p)

    for hhmm in times:
        still = []
        for x in pos:
            p = price(x[0], hhmm)
            if p is None:
                still.append(x)
                continue
            hit = (x[1] * (p - x[4]) < 0) or (x[1] * (p - x[5]) >= 0)
            if hit or hhmm == "15:15":
                pnl += close(x, p)
            else:
                still.append(x)
        pos = still
        if review and hhmm in REVIEWS and pos:
            keep = []
            for x in pos:
                p = price(x[0], hhmm)
                move = x[1] * (p / x[3] - 1) if p else 0.0
                age = (pd.Timestamp(f"2000-01-01 {hhmm}") - pd.Timestamp(f"2000-01-01 {x[6]}")).seconds / 60
                if p and age >= PI.REVIEW_EVERY - 1 and move < PI.SIDEWAYS:
                    pnl += close(x, p)
                else:
                    keep.append((move, x, p))
            pos = [x for _, x, _ in keep]
            held = {}
            for x in pos:
                held[x[0]] = held.get(x[0], 0) + x[2] * x[3]
            free = POWER - sum(held.values())
            best = []
            for move, x, p in sorted(keep, key=lambda k: -k[0]):
                if p and move >= PI.STRONG and x[0] not in [b[1][0] for b in best]:
                    best.append((move, x, p))
                if len(best) == PI.TOP_WINNERS:
                    break
            for _, x, p in best:
                room = min(free / len(best), PI.MAX_IN_ONE * POWER - held[x[0]])
                q = int(room / p)
                if q > 0:
                    pnl -= COSTS.cost("buy" if x[1] > 0 else "sell", q * p)
                    pos.append([x[0], x[1], q, p, p, x[5], hhmm])
    return pnl


rows = []
first_bar_day = min(s.index.min() for s in px.values()).normalize()
for day, g in scores.groupby("date"):
    if pd.Timestamp(day) < first_bar_day:
        continue
    g = g.sort_values("score", ascending=False)
    picks = pd.concat([g.head(PI.N_TRADED).assign(side="long"),
                       g.tail(PI.N_TRADED).iloc[::-1].assign(side="short")])
    a, b = day_pnl(pd.Timestamp(day), picks, False), day_pnl(pd.Timestamp(day), picks, True)
    if a is not None and b is not None:
        rows.append((pd.Timestamp(day), a, b))
        print(f"{pd.Timestamp(day):%Y-%m-%d}  hold {a:+8.0f}  review {b:+8.0f}", flush=True)

r = pd.DataFrame(rows, columns=["day", "hold", "review"])
if r.empty:
    sys.exit("no overlapping days between scores and Yahoo candles")
diff = r["review"] - r["hold"]
t = diff.mean() / diff.std() * np.sqrt(len(diff)) if len(diff) > 2 and diff.std() else 0.0
print(f"\n{len(r)} days on ₹10,000: hold avg {r['hold'].mean():+.0f}/day (total {r['hold'].sum():+.0f}), "
      f"review avg {r['review'].mean():+.0f}/day (total {r['review'].sum():+.0f})", flush=True)
print(f"review - hold: {diff.mean():+.0f}/day, paired t = {t:.2f}, better on {(diff > 0).mean():.0%} "
      f"of days -> " + ("KEEP" if t >= 2 and len(r) >= 30 else "SWITCH OFF" if t <= -2
                        else "INCONCLUSIVE"), flush=True)
