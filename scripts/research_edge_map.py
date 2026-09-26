"""Where can short-term trading beat its costs? (GitHub, research.yml)

For the intraday model's picks (walk-forward: each month scored by a model trained only on
earlier days), ranked #1..#10 on the buy and the sell side, the average move BEFORE costs
from the 9:45 entry to: 12:30, the 15:15 close exit, the next day's close (2-day swing) and
the close 4 trading days later (5-day swing). Compared with realistic round-trip costs:
intraday (MIS) ~0.2% for a Rs 20-50k position; delivery (swing, buys only - shorts cannot be
held overnight in the cash market) ~0.3% (STT 0.1% each side, stamp, slippage).
Also the long-term (weekly) model's top picks, as the reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.costs import DEFAULT_COSTS, DEFAULT_INTRADAY_COSTS
from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T
from stockpredictor.paper.engine import MarketContext

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
ctx = MarketContext.load(d)
closes = ctx.daily.pivot(index="date", columns="symbol", values="close").sort_index()


def round_trip(costs, value: float = 25_000, delivery: bool = False) -> float:
    return (costs.cost("buy", value) + costs.cost("sell", value)) / value


cost_id, cost_dl = round_trip(DEFAULT_INTRADAY_COSTS), round_trip(DEFAULT_COSTS, delivery=True)
print(f"round-trip costs on Rs 25,000: intraday {cost_id:.2%}, delivery (swing) {cost_dl:.2%}\n",
      flush=True)

for target in MI.TARGETS:
    feats = T.intraday_feats(ctx, d, target)
    scores = MI.walk_forward(feats, params=MI.current_params(target), last_days=240)
    m = scores.merge(feats[["symbol", "date", "c30", "px_1230", "px_1515"]], on=["symbol", "date"])
    # forward closes: next day (2-day swing) and 4 trading days later (5-day swing)
    idx = {dt: i for i, dt in enumerate(closes.index)}
    def fwd(row_dates, syms, k):
        out = np.full(len(row_dates), np.nan)
        for j, (dt, s) in enumerate(zip(row_dates, syms)):
            i = idx.get(dt)
            if i is not None and i + k < len(closes.index) and s in closes:
                out[j] = closes.iat[i + k, closes.columns.get_loc(s)]
        return out
    m["c_1d"] = fwd(m["date"], m["symbol"], 1)
    m["c_4d"] = fwd(m["date"], m["symbol"], 4)
    m["rank_long"] = m.groupby("date")["score"].rank(ascending=False, method="first")
    m["rank_short"] = m.groupby("date")["score"].rank(ascending=True, method="first")
    buckets = [("#1", 1, 1), ("#2-3", 2, 3), ("#4-5", 4, 5), ("#6-10", 6, 10)]
    print(f"== ranking by the '{target.label}' model, {m['date'].nunique()} days", flush=True)
    print(f"{'pick':14}{'to 12:30':>10}{'to close':>10}{'2-day':>10}{'5-day':>10}   (average move in "
          "the pick's favour, before costs; hit rate)", flush=True)
    for side, sign, col in (("buy", 1, "rank_long"), ("sell", -1, "rank_short")):
        for name, lo, hi in buckets:
            g = m[(m[col] >= lo) & (m[col] <= hi)]
            cells = []
            for exit_col in ("px_1230", "px_1515", "c_1d", "c_4d"):
                r = sign * (g[exit_col] / g["c30"] - 1)
                cells.append(f"{r.mean():+.2%}/{(r > 0).mean():.0%}")
            print(f"{side + ' ' + name:14}" + "".join(f"{c:>12}" for c in cells), flush=True)
    print(flush=True)

lab = T.longterm_labeled(ctx)
from stockpredictor.models import longterm as M  # noqa: E402

raw = M.walk_forward(lab, ctx.feats, lab["date"].max().year - 2)
w = raw.merge(lab[["symbol", "date", "fwd_ret", "fwd_excess"]], on=["symbol", "date"]).dropna()
w["r"] = w.groupby("date")["score"].rank(ascending=False)
top = w[w["r"] <= 5]
print(f"== long-term (weekly) model, top 5 picks, last 3 years: 1-week move {top['fwd_ret'].mean():+.2%} "
      f"(vs Nifty {top['fwd_excess'].mean():+.2%}), up {(top['fwd_ret'] > 0).mean():.0%} of the time; "
      f"delivery cost {cost_dl:.2%} per round trip (held ~several weeks on average)", flush=True)
