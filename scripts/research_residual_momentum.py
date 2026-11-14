"""Experiment E-010 (GitHub, research.yml): two long-term features from the literature.
1. Residual momentum - Blitz, Huij & Martens (2011, J. Empirical Finance): rank on the
   stock's past returns after removing its market (beta x Nifty) part, scaled by residual
   volatility; ~half the risk of plain momentum. Windows 12-1 and 6-1 months.
2. Overnight vs intraday momentum - Lou, Polk & Skouras (2019, JFE): momentum profits
   come overnight (close -> next open) while intraday (open -> close) partly reverses;
   features: past 21/63/252-day sums of overnight and of intraday returns.
Protocol (fixed): identical data, walk-forward 2015-2026, same LightGBM, same backtest.
ACCEPT a feature set only if weekly IC improves with paired t >= 2 AND CAGR and Sharpe after
costs are not worse.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.models import longterm as M

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats, labeled = bt.load_all(d)
close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
opn = daily.pivot(index="date", columns="symbol", values="open").reindex(close.index)
ret = close.pct_change(fill_method=None)
mkt = indices[indices["symbol"] == "NIFTY50"].set_index("date")["close"].reindex(close.index).pct_change()

# 1. residual momentum (rolling 252-day beta, residuals, sums skipping the last month)
cov = ret.rolling(252, min_periods=126).cov(mkt)
beta = cov.div(mkt.rolling(252, min_periods=126).var(), axis=0)
res = ret - beta.mul(mkt, axis=0)
rsd = res.rolling(252, min_periods=126).std()
resid = {}
for name, w in (("res_mom_12_1", 252), ("res_mom_6_1", 126)):
    s = res.shift(21).rolling(w - 21, min_periods=(w - 21) // 2).sum()
    resid[name] = s / (rsd * np.sqrt(w - 21))
# 2. overnight vs intraday momentum
on = opn / close.shift(1) - 1
intra = close / opn - 1
split = {}
for w in (21, 63, 252):
    split[f"overnight_{w}"] = on.rolling(w, min_periods=w // 2).sum()
    split[f"intraday_{w}"] = intra.rolling(w, min_periods=w // 2).sum()


def long_form(frames: dict) -> pd.DataFrame:
    out = None
    for k, v in frames.items():
        s = v.stack(future_stack=True).rename(k).reset_index()
        s.columns = ["date", "symbol", k]
        out = s if out is None else out.merge(s, on=["date", "symbol"], how="outer")
    return out


sets = {"+ residual momentum": long_form(resid), "+ overnight/intraday": long_form(split)}
closep = close
risk_off = P.risk_off_days(indices)
fwd = labeled.set_index(["symbol", "date"])["target"]


def run(lab, fts, name):
    raw = M.walk_forward(lab, fts, 2015)
    raw = raw.merge(bt.point_in_time_universe(fts, 100), on=["symbol", "date"])
    wk = raw.merge(fwd.rename("y").reset_index(), on=["symbol", "date"]).dropna(subset=["y"])
    ic = wk.groupby("date").apply(lambda x: x["score"].corr(x["y"], method="spearman"))
    eq = P.simulate(M.smooth_scores(raw, fts), closep, P.Rules(), 100_000, "weekly",
                    risk_off=risk_off).equity
    p = P.performance(eq)
    print(f"{name:22} IC {ic.mean():.4f} (t {ic.mean() / ic.std() * np.sqrt(len(ic)):.1f}) | "
          f"CAGR {p['cagr']:.1%} Sharpe {p['sharpe']:.2f} worst {p['max_drawdown']:.1%}", flush=True)
    return ic, p


base_ic, base_p = run(labeled, feats, "baseline")
for name, extra in sets.items():
    ic, p = run(labeled.merge(extra, on=["symbol", "date"], how="left"),
                feats.merge(extra, on=["symbol", "date"], how="left"), name)
    diff = (ic - base_ic).dropna()
    t = diff.mean() / diff.std() * np.sqrt(len(diff))
    ok = t >= 2 and p["cagr"] >= base_p["cagr"] and p["sharpe"] >= base_p["sharpe"]
    print(f"{name}: paired IC {diff.mean():+.4f}, t = {t:.2f} -> {'ACCEPT' if ok else 'REJECT'}",
          flush=True)
