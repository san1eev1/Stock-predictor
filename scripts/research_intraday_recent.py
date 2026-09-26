"""Intraday: old vs new trading rules on recent days (GitHub, research.yml).

Scores are walk-forward (each month predicted by a model trained only on earlier days), so
no day is judged by a model that saw it. Old rules = what paper trading used until
25 Sep 2026 (10 buys + 10 sells, stop 1%, target 2%, trade every day). New rules = the
book's profit-tuned rules (trained-models/<book>/rules.json) with the live defaults (strong
days only, risk per trade 1%). Caution: the new rules were tuned on history that includes
these days, which flatters them somewhat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from stockpredictor.backtest import intraday as B
from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T
from stockpredictor.paper.engine import MarketContext

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
ctx = MarketContext.load(d)
OLD = B.IntradayRules(n_long=10, n_short=10, stop_loss=1.0, target=2.0, skip_quantile=0.0)
DAY = pd.Timestamp("2026-09-25")

for target in MI.TARGETS:
    feats = T.intraday_feats(ctx, d, target)
    minutes = 165 if target == MI.TRADE else 330
    scores = MI.walk_forward(feats, params=MI.current_params(target), last_days=120)
    summ = B.summary_columns(feats, target.exit_col)
    new = T.rules_for(target)
    print(f"\n== {target.label}: new rules {new}", flush=True)
    for name, rules in (("old rules", OLD), ("new rules", new)):
        sim = B.simulate(scores, summ, rules, 100_000, exit_col=target.exit_col,
                         exit_minutes=minutes)
        days, trades = sim["days"], sim["trades"]
        last60 = days[days["date"].isin(sorted(days["date"].unique())[-60:])]
        t60 = trades[trades["date"].isin(last60["date"])] if not trades.empty else trades
        y = days[days["date"] == DAY]
        ty = trades[trades["date"] == DAY] if not trades.empty else trades
        yday = (f"25 Sep: Rs {y['pnl'].sum():+.0f} "
                + ("(skipped: weak day)" if len(y) and bool(y["skipped"].iloc[0]) else
                   f"({len(ty)} trades, {ty['correct'].mean():.0%} right)" if len(ty) else ""))
        print(f"  {name:10} | {yday} | last 60 days: Rs {last60['pnl'].sum():+,.0f} total, "
              f"Rs {last60['pnl'].mean():+.0f}/day, traded on {int((~last60['skipped']).sum())} "
              f"days, {len(t60)} trades, {t60['correct'].mean() if len(t60) else float('nan'):.1%} "
              f"right, costs Rs {t60['cost'].sum() if len(t60) else 0:,.0f}", flush=True)
