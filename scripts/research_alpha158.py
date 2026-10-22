"""Experiment E-001 (GitHub, research.yml): Qlib Alpha158 feature families we don't have.

Hypothesis: price-volume correlation (CORR/CORD), the timing of the high vs the low (IMXD),
range quantiles (QTLU/QTLD), stochastic position (RSV) and up-minus-down day counts (CNTD)
add information to the long-term model. Source: microsoft/qlib@5435523,
qlib/contrib/data/loader.py (Alpha158DL.get_feature_config), MIT licence.

Protocol (fixed before running): identical data, walk-forward years 2015-2026 (embargoed),
same LightGBM params, same backtest rules/costs/universe. Accept only if weekly IC improves
with paired t >= 2 AND the 5-holding backtest CAGR and Sharpe after costs don't get worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.features import longterm as F
from stockpredictor.models import longterm as M

WINDOWS = (20, 60)


def _rolling_corr(a: pd.Series, b: pd.Series, sym: pd.Series, w: int) -> pd.Series:
    parts = [a[idx].rolling(w, min_periods=w // 2).corr(b[idx])
             for idx in sym.groupby(sym).groups.values()]
    return pd.concat(parts).reindex(a.index)


def alpha158_extra(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["symbol", "date"]).copy()
    g = d.groupby("symbol", group_keys=False)
    logv = np.log(d["volume"].fillna(0) + 1)
    ret = g["close"].pct_change(fill_method=None)
    dv = np.log((d["volume"] + 1) / (g["volume"].shift(1) + 1))
    out = d[["symbol", "date"]].copy()
    for w in WINDOWS:
        roll = lambda s, f: s.groupby(d["symbol"]).transform(  # noqa: E731
            lambda x: f(x.rolling(w, min_periods=w // 2)))
        out[f"a158_corr{w}"] = _rolling_corr(d["close"], logv, d["symbol"], w)
        out[f"a158_cord{w}"] = _rolling_corr(ret, dv, d["symbol"], w)
        hi = roll(d["high"], lambda r: r.max())
        lo = roll(d["low"], lambda r: r.min())
        out[f"a158_rsv{w}"] = (d["close"] - lo) / (hi - lo + 1e-12)
        out[f"a158_qtlu{w}"] = roll(d["close"], lambda r: r.quantile(0.8)) / d["close"]
        out[f"a158_qtld{w}"] = roll(d["close"], lambda r: r.quantile(0.2)) / d["close"]
        imax = roll(d["high"], lambda r: r.apply(np.argmax, raw=True))
        imin = roll(d["low"], lambda r: r.apply(np.argmin, raw=True))
        out[f"a158_imxd{w}"] = (imax - imin) / w
        up = (ret > 0).astype(float)
        dn = (ret < 0).astype(float)
        out[f"a158_cntd{w}"] = roll(up, lambda r: r.mean()) - roll(dn, lambda r: r.mean())
    return out


d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats, labeled = bt.load_all(d)
extra = alpha158_extra(daily)
feats2 = feats.merge(extra, on=["symbol", "date"], how="left")
labeled2 = labeled.merge(extra, on=["symbol", "date"], how="left")
print(f"added {extra.shape[1] - 2} features; baseline {len(F.feature_columns(feats))} features",
      flush=True)

close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
risk_off = P.risk_off_days(indices)
fwd = labeled.set_index(["symbol", "date"])["target"]


def run(lab, fts, name):
    raw = M.walk_forward(lab, fts, 2015)
    raw = raw.merge(bt.point_in_time_universe(fts, 100), on=["symbol", "date"])
    wk = raw.merge(fwd.rename("y").reset_index(), on=["symbol", "date"]).dropna(subset=["y"])
    ic = wk.groupby("date").apply(lambda x: x["score"].corr(x["y"], method="spearman"))
    eq = P.simulate(M.smooth_scores(raw, fts), close, P.Rules(), 100_000, "weekly",
                    risk_off=risk_off).equity
    p = P.performance(eq)
    print(f"{name:12} IC {ic.mean():.4f} (t {ic.mean() / ic.std() * np.sqrt(len(ic)):.1f}) | "
          f"CAGR {p['cagr']:.1%} Sharpe {p['sharpe']:.2f} worst {p['max_drawdown']:.1%}",
          flush=True)
    return ic


a = run(labeled, feats, "baseline")
b = run(labeled2, feats2, "+alpha158")
diff = (b - a).dropna()
t = diff.mean() / diff.std() * np.sqrt(len(diff))
print(f"paired IC difference {diff.mean():+.4f}, t = {t:.2f} over {len(diff)} weeks "
      f"-> {'ACCEPT' if t >= 2 else 'REJECT'} (also needs CAGR/Sharpe not worse)", flush=True)
