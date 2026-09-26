"""Two hypotheses for the long-term book (GitHub, research.yml).

H1 crash guard: sudden VIX spikes / sharp weekly Nifty falls (portfolio.crash_days) - cut
   exposure (exit / no new buys / half-size). Must improve the full period AND the last 5
   years, and cut the COVID-2020 and 2024-25 losses.
H2 intraday sell signal: never hold stocks the 'until close' intraday model ranks as a
   top-5 sell that day (their next days average +0.75..1.2% in the sell's favour). Only
   testable where intraday history exists (~2 years).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.models import intraday as MI
from stockpredictor.models import longterm as M
from stockpredictor.models import trainer as T
from stockpredictor.paper.engine import MarketContext

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats, labeled = bt.load_all(d)
raw = M.walk_forward(labeled, feats, 2015)
raw = raw.merge(bt.point_in_time_universe(feats, 100), on=["symbol", "date"])
base = M.smooth_scores(raw, feats)
close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
risk_off, crash = P.risk_off_days(indices), P.crash_days(indices)
print(f"crash guard on for {len(crash & set(close.index)) / len(close.index):.0%} of days", flush=True)
WINDOWS = {"COVID 2020": ("2020-02-01", "2020-04-30"), "2024-25": ("2024-09-25", "2025-03-31")}


def line(name: str, eq: pd.Series, start=None) -> None:
    if start is not None:
        eq = eq[eq.index >= start]
    parts = []
    for label, e in (("full", eq), ("last 5y", eq[eq.index >= eq.index.max() - pd.DateOffset(years=5)])):
        p = P.performance(e / e.iloc[0] * 100_000)
        parts.append(f"{label} {p['cagr']:6.1%}/yr Sharpe {p['sharpe']:.2f} worst {p['max_drawdown']:6.1%}")
    for w, (a, b) in WINDOWS.items():
        s = eq[(eq.index >= a) & (eq.index <= b)]
        if len(s) > 5:
            parts.append(f"{w} {s.iloc[-1] / s.iloc[0] - 1:+6.1%}")
    print(f"{name:30} | " + " | ".join(parts), flush=True)


def sim(scores, rules):
    return P.simulate(scores, close, rules, 100_000, "weekly", risk_off=risk_off,
                      crash_days=crash).equity


print("\n== H1 crash guard (2015-2026)", flush=True)
line("current", sim(base, P.Rules()))
for mode in ("exit", "no_buys", "half"):
    line(f"crash guard: {mode}", sim(base, P.Rules(crash=mode)))

print("\n== H2 intraday sell signal as a filter", flush=True)
ctx = MarketContext.load(d)
fi = T.intraday_feats(ctx, d, MI.CLOSE)
si = MI.walk_forward(fi, params=MI.current_params(MI.CLOSE), last_days=500)
si["rank_sell"] = si.groupby("date")["score"].rank(ascending=True, method="first")
flag = si.loc[si["rank_sell"] <= 5, ["symbol", "date"]]
start = si["date"].min()
filtered = base.merge(flag.assign(f=1), on=["symbol", "date"], how="left")
filtered["score"] = np.where(filtered["f"] == 1, filtered["score"].min() - 1, filtered["score"])
filtered = filtered[["symbol", "date", "score"]]
print(f"period {start:%Y-%m-%d} to {si['date'].max():%Y-%m-%d}, {len(flag)} sell flags", flush=True)
b = base[base["date"] >= start]
f = filtered[filtered["date"] >= start]
line("current", sim(b, P.Rules()), start)
line("with sell-signal filter", sim(f, P.Rules()), start)
