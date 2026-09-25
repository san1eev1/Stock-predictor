"""Long-term research, run on GitHub (research.yml) so the Mac stays cool.

Tests, on the walk-forward scores (each year predicted by a model trained only on earlier
years), with the live trading rules (5 holdings, weekly, costs):
  1. market regime filter: what to do while Nifty 50 is below its 200-day average
  2. low-volatility factor: prefer calmer stocks (blend in the rank of -63-day volatility)
Every variant is reported for 2015-2026 and for the last 5 years separately (a change must
help in both to be trusted).
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
raw = M.walk_forward(labeled, feats, 2015)
raw = raw.merge(bt.point_in_time_universe(feats, 100), on=["symbol", "date"])
close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
u = store.load_universe(d)
sectors = dict(zip(u["symbol"], u["industry"]))
risk_off = P.risk_off_days(indices)
base = M.smooth_scores(raw, feats)                       # live trading score (90% momentum)
print(f"walk-forward scores {raw['date'].min():%Y-%m-%d} to {raw['date'].max():%Y-%m-%d}; "
      f"Nifty below its 200-day average on {len(risk_off & set(close.index)) / len(close):.0%} "
      "of days", flush=True)


def low_vol(scores: pd.DataFrame, weight: float) -> pd.DataFrame:
    s = scores.merge(feats[["symbol", "date", "vol_63"]], on=["symbol", "date"], how="left")
    r_score = s.groupby("date")["score"].rank(pct=True)
    r_calm = (-s["vol_63"]).groupby(s["date"]).rank(pct=True)
    s["score"] = (1 - weight) * r_score + weight * r_calm.fillna(0.5)
    return s[["symbol", "date", "score"]]


def run(name: str, scores: pd.DataFrame, rules: P.Rules) -> None:
    res = P.simulate(scores, close, rules, 100_000, "weekly", sectors=sectors, risk_off=risk_off)
    eq = res.equity
    rows = [("2015-2026", eq), ("last 5 yrs", eq[eq.index >= eq.index.max() - pd.DateOffset(years=5)])]
    out = []
    for label, e in rows:
        p = P.performance(e / e.iloc[0] * 100_000)
        out.append(f"{label}: {p['cagr']:6.1%}/yr Sharpe {p['sharpe']:.2f} worst {p['max_drawdown']:6.1%}")
    print(f"{name:34} | " + " | ".join(out), flush=True)


rules = P.Rules()
nifty = indices[indices["symbol"] == "NIFTY50"].set_index("date")["close"].sort_index()
for label, e in (("2015-2026", nifty[nifty.index >= raw["date"].min()]),
                 ("last 5 yrs", nifty[nifty.index >= nifty.index.max() - pd.DateOffset(years=5)])):
    p = P.performance(e / e.iloc[0] * 100_000)
    print(f"{'Nifty 50 (' + label + ')':34} | {p['cagr']:6.1%}/yr Sharpe {p['sharpe']:.2f} "
          f"worst {p['max_drawdown']:6.1%}", flush=True)
run("current (no filter)", base, rules)
for regime in ("no_buys", "exit", "half"):
    run(f"regime: {regime}", base, P.Rules(**{**rules.__dict__, "regime": regime}))
for w in (0.15, 0.3, 0.5):
    run(f"low-vol weight {w}", low_vol(base, w), rules)
    for regime in ("no_buys", "exit"):
        run(f"low-vol {w} + regime {regime}", low_vol(base, w),
            P.Rules(**{**rules.__dict__, "regime": regime}))
