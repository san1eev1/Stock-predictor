"""Long-term: merging models (ensembles). GitHub, research.yml.

  week   : the current model (predicts the next 5 trading days)
  month  : the same inputs, predicting the next 21 trading days (longer embargo: 45 days)
  recent : the week model with recent years weighted more (half-life 3 years)
Ensembles = the average of the models' per-day ranks. Each goes through the live trading
score (20-day average, 90% momentum) and the live rules (5 holdings, weekly, costs, 'half'
regime filter). Reported for 2015-2026 and the last 5 years, with the ranking quality (IC,
1-week excess return).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from stockpredictor import store
from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.models import longterm as M

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats, labeled = bt.load_all(d)
universe = bt.point_in_time_universe(feats, 100)
close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
risk_off = P.risk_off_days(indices)

week = M.walk_forward(labeled, feats, 2015)
recent = M.walk_forward(labeled, feats, 2015, params={**M.current_params(), "recency_half_life": 3.0})
month_labeled = M.add_labels(M.training_snapshots(feats), daily, indices, horizon=21)
M.EMBARGO_DAYS = 45                                   # 21-day labels must not reach the test year
month = M.walk_forward(month_labeled, feats, 2015)
M.EMBARGO_DAYS = 14


def ranks(s: pd.DataFrame) -> pd.Series:
    return s.set_index(["symbol", "date"])["score"].groupby(level="date").rank(pct=True)


def merge(*parts: pd.DataFrame) -> pd.DataFrame:
    r = pd.concat([ranks(p) for p in parts], axis=1).mean(axis=1)
    return r.rename("score").reset_index()


def report(name: str, raw: pd.DataFrame) -> None:
    raw = raw.merge(universe, on=["symbol", "date"])
    ic = M.information_coefficient(raw, labeled)
    eq = P.simulate(M.smooth_scores(raw, feats), close, P.Rules(), 100_000, "weekly",
                    risk_off=risk_off).equity
    out = []
    for label, e in (("2015-2026", eq), ("last 5 yrs", eq[eq.index >= eq.index.max() - pd.DateOffset(years=5)])):
        p = P.performance(e / e.iloc[0] * 100_000)
        out.append(f"{label}: {p['cagr']:6.1%}/yr Sharpe {p['sharpe']:.2f} worst {p['max_drawdown']:6.1%}")
    print(f"{name:26} | IC {ic.mean():.4f} | " + " | ".join(out), flush=True)


report("week (current)", week)
report("month", month)
report("recent-weighted", recent)
report("week + month", merge(week, month))
report("week + recent", merge(week, recent))
report("week + month + recent", merge(week, month, recent))
