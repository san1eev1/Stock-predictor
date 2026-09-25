"""Book / research ideas, tested on GitHub (research.yml) - kept only if they hold up.

  A. Volatility targeting (Moreira & Muir 2017; Barroso & Santa-Clara 2015; Carver,
     "Systematic Trading"): invest less when the portfolio's own recent volatility is high.
     Standard academic overlay on the daily returns: weight = min(1, target / vol of the last
     63 days, known the day before), 0.3% cost per unit of weight change.
  B. Clenow momentum ("Stocks on the Move"): rank by the 90-day exponential trend slope x R^2
     (steady trends beat jumpy ones), only stocks above their 100-day average.
  C. Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): how likely each Sharpe ratio is
     real rather than the luck of trying many variants on the same history.

Everything uses the live rules (5 holdings, weekly, costs, 'half' regime filter) and is
reported for 2015-2026 and for the last 5 years.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from stockpredictor import store
from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.models import longterm as M

TRIALS = 60            # variants tried on this history so far (for the deflated Sharpe)

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats, labeled = bt.load_all(d)
raw = M.walk_forward(labeled, feats, 2015)
raw = raw.merge(bt.point_in_time_universe(feats, 100), on=["symbol", "date"])
close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
u = store.load_universe(d)
sectors = dict(zip(u["symbol"], u["industry"]))
risk_off = P.risk_off_days(indices)
base = M.smooth_scores(raw, feats)
rules = P.Rules()
results: dict[str, pd.Series] = {}


def clenow_scores(window: int = 90, ma: int = 100) -> pd.DataFrame:
    lp = np.log(close)
    t = pd.Series(np.arange(len(lp)), index=lp.index, dtype=float)
    cov = lp.rolling(window).cov(t)
    var_t = t.rolling(window).var()
    slope = cov.div(var_t, axis=0)
    r2 = lp.rolling(window).corr(t) ** 2
    ann = np.exp(slope * 252) - 1
    score = (ann * r2).where(close > close.rolling(ma).mean())       # only uptrends
    s = score.stack().rename("score").reset_index()
    s.columns = ["date", "symbol", "score"]
    return s.merge(raw[["symbol", "date"]], on=["symbol", "date"])   # same days / universe


def blend(a: pd.DataFrame, b: pd.DataFrame, w: float) -> pd.DataFrame:
    m = a.merge(b, on=["symbol", "date"], how="left", suffixes=("_a", "_b"))
    ra = m.groupby("date")["score_a"].rank(pct=True)
    rb = m.groupby("date")["score_b"].rank(pct=True).fillna(0)
    return m.assign(score=(1 - w) * ra + w * rb)[["symbol", "date", "score"]]


def vol_target(eq: pd.Series, target: float) -> pd.Series:
    r = eq.pct_change().fillna(0)
    vol = r.rolling(63, min_periods=20).std().shift(1) * math.sqrt(252)
    w = (target / vol).clip(upper=1.0).fillna(1.0)
    net = w * r - 0.003 * w.diff().abs().fillna(0)
    return (1 + net).cumprod() * eq.iloc[0]


def report(name: str, eq: pd.Series) -> None:
    results[name] = eq
    out = []
    for label, e in (("2015-2026", eq), ("last 5 yrs", eq[eq.index >= eq.index.max() - pd.DateOffset(years=5)])):
        p = P.performance(e / e.iloc[0] * 100_000)
        out.append(f"{label}: {p['cagr']:6.1%}/yr Sharpe {p['sharpe']:.2f} worst {p['max_drawdown']:6.1%}")
    print(f"{name:38} | " + " | ".join(out), flush=True)


def sim(scores: pd.DataFrame) -> pd.Series:
    return P.simulate(scores, close, rules, 100_000, "weekly", sectors=sectors,
                      risk_off=risk_off).equity


def deflated_sharpe(eq: pd.Series, sr_std: float, trials: int = TRIALS) -> float:
    """Probability that the true Sharpe ratio is above what the best of `trials` random
    strategies would show (Bailey & Lopez de Prado). Daily returns; > 0.95 = convincing."""
    r = eq.pct_change().dropna()
    n, sr = len(r), r.mean() / r.std()
    skew = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    nd, g = NormalDist(), 0.5772156649
    sr0 = sr_std * ((1 - g) * nd.inv_cdf(1 - 1 / trials) + g * nd.inv_cdf(1 - 1 / (trials * math.e)))
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return nd.cdf(z)


nifty = indices[indices["symbol"] == "NIFTY50"].set_index("date")["close"].sort_index()
nifty = nifty[nifty.index >= raw["date"].min()]
report("Nifty 50", nifty)
current = sim(base)
report("current strategy (half regime filter)", current)
for t in (0.20, 0.25, 0.30):
    report(f"A. + volatility target {t:.0%}", vol_target(current, t))
cl = clenow_scores()
report("B. Clenow momentum alone", sim(cl))
for w in (0.3, 0.5):
    report(f"B. current + Clenow {w:.0%}", sim(blend(base, cl, w)))
    report(f"B+A. current + Clenow {w:.0%} + vol 25%", vol_target(sim(blend(base, cl, w)), 0.25))

daily_sr = [e.pct_change().dropna().mean() / e.pct_change().dropna().std() for e in results.values()]
sr_std = float(np.std(daily_sr))
print(f"\nC. Deflated Sharpe ratio (chance the edge is real after ~{TRIALS} variants were tried "
      "on this history; above 95% is convincing):", flush=True)
for name, eq in results.items():
    print(f"  {name:38} {deflated_sharpe(eq, sr_std):6.1%}", flush=True)
