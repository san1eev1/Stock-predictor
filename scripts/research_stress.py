"""Stress test: the long-term strategy (live rules) through real crises. GitHub, research.yml.

Walk-forward from 2008 (models trained only on earlier years), 5 holdings, weekly, costs,
'half' regime filter. For each crisis window: the strategy's return and worst fall next to
Nifty 50's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.models import longterm as M

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats, labeled = bt.load_all(d)
raw = M.walk_forward(labeled, feats, 2008)
raw = raw.merge(bt.point_in_time_universe(feats, 100), on=["symbol", "date"])
close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
eq = P.simulate(M.smooth_scores(raw, feats), close, P.Rules(), 100_000, "weekly",
                risk_off=P.risk_off_days(indices)).equity
nifty = indices[indices["symbol"] == "NIFTY50"].set_index("date")["close"].sort_index()

WINDOWS = {"2008 global financial crisis": ("2008-01-01", "2009-03-31"),
           "2011 euro-debt / India slowdown": ("2011-01-01", "2011-12-31"),
           "2015-16 China / commodities": ("2015-03-01", "2016-02-29"),
           "Nov 2016 demonetisation": ("2016-11-01", "2016-12-31"),
           "2018 IL&FS / midcap crash": ("2018-01-15", "2018-10-31"),
           "Feb-Mar 2020 COVID crash": ("2020-02-01", "2020-04-30"),
           "2022 rate hikes / war": ("2022-01-01", "2022-06-30"),
           "2024-25 correction": ("2024-09-25", "2025-03-31")}


def stats(series: pd.Series, a: str, b: str) -> str:
    s = series[(series.index >= a) & (series.index <= b)]
    if len(s) < 5:
        return "no data"
    ret = s.iloc[-1] / s.iloc[0] - 1
    worst = (s / s.cummax() - 1).min()
    return f"{ret:+7.1%} (worst fall {worst:6.1%})"


print(f"{'crisis':34}{'strategy':>32}{'Nifty 50':>32}", flush=True)
for name, (a, b) in WINDOWS.items():
    print(f"{name:34}{stats(eq, a, b):>32}{stats(nifty, a, b):>32}", flush=True)
p = P.performance(eq)
n = P.performance(nifty[nifty.index >= eq.index.min()] / nifty[nifty.index >= eq.index.min()].iloc[0])
print(f"\nwhole period {eq.index.min():%Y}-{eq.index.max():%Y}: strategy {p['cagr']:.1%}/yr, worst "
      f"fall {p['max_drawdown']:.1%}; Nifty {n['cagr']:.1%}/yr, worst fall {n['max_drawdown']:.1%}",
      flush=True)
