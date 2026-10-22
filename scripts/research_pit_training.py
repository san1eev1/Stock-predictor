"""Experiments E-003 and E-004 (GitHub, research.yml). Everything is EVALUATED on the
point-in-time (survivorship-free) universe from bhavcopy, the honest test since E-000.

E-003 Train on point-in-time history? Production trains on today's Nifty 250 only (~250
  stocks, survivors). Training on every stock that was in a day's top 250 (incl. those
  that later fell out) gives more, unbiased rows; the model is data-limited (E-002).
  A: train on survivors, B: train on point-in-time; both score the point-in-time universe.
  Accept B if weekly IC paired t >= 2 AND CAGR and Sharpe after costs are not worse.

E-004 Trailing stop (QuantConnect LEAN, Algorithm.Framework/Risk/
  TrailingStopRiskManagementModel.py, Apache-2.0, re-implemented in our simulator): sell a
  holding that falls X% below its highest close since purchase. Weakness addressed:
  -51% worst fall. Accept if worst fall improves by >= 5 points with CAGR not worse by
  more than 1 point/yr AND Sharpe not worse, for a trailing level chosen on 2015-2020
  and confirmed on 2021-2026 (no tuning on the confirmation years).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor import store
from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.data import bhav
from stockpredictor.data.clean import clean_daily
from stockpredictor.features import longterm as F
from stockpredictor.models import longterm as M

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
indices = store.load_indices(d)
risk_off = P.risk_off_days(indices)
survivors = store.load_daily(d)
b = bhav.load(d).dropna(subset=["close"]).assign(adj_close=lambda x: x["close"], source="bhav")
b["value_med60"] = b.sort_values("date").groupby("symbol")["value"].transform(
    lambda x: x.rolling(60, min_periods=20).median())
rank = b.groupby("date")["value_med60"].rank(ascending=False, method="first")
keep = set(b.loc[rank <= 250, "symbol"])
pit, _ = clean_daily(b[b["symbol"].isin(keep)][["symbol", "date", "open", "high", "low", "close",
                                               "adj_close", "volume", "source"]])
print(f"survivors {survivors['symbol'].nunique()} stocks, point-in-time {len(keep)} stocks", flush=True)

feats_pit = F.build_features(pit, indices, None)
lab_pit = M.add_labels(F.weekly_snapshots(feats_pit), pit, indices)
feats_srv = F.build_features(survivors, indices, store.load_universe(d))
lab_srv = M.add_labels(F.weekly_snapshots(feats_srv), survivors, indices)
cols = sorted(set(M._model_columns(lab_pit)) & set(M._model_columns(lab_srv)))
univ = bt.point_in_time_universe(feats_pit, 100)
close = pit.pivot(index="date", columns="symbol", values="close").sort_index()


def walk(train_lab: pd.DataFrame) -> pd.DataFrame:
    """Yearly walk-forward: train on `train_lab` before each year, score point-in-time feats."""
    out = []
    for year in range(2015, feats_pit["date"].dt.year.max() + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        tr = train_lab[train_lab["date"] < start - pd.Timedelta(days=M.EMBARGO_DAYS)].dropna(
            subset=["target"])
        te = feats_pit[feats_pit["date"].dt.year == year]
        if len(tr) < 1000 or te.empty:
            continue
        m = M._fit(tr, cols)
        out.append(te[["symbol", "date"]].assign(score=m.predict(te[cols])))
    return pd.concat(out).merge(univ, on=["symbol", "date"])


def report(name: str, raw: pd.DataFrame, rules: P.Rules = P.Rules()) -> tuple[pd.Series, dict]:
    ic = M.information_coefficient(raw, lab_pit)
    eq = P.simulate(M.smooth_scores(raw, feats_pit), close, rules, 100_000, "weekly",
                    risk_off=risk_off).equity
    p = P.performance(eq)
    print(f"{name:34} IC {ic.mean():.4f} | CAGR {p['cagr']:6.1%} Sharpe {p['sharpe']:.2f} "
          f"worst {p['max_drawdown']:6.1%}", flush=True)
    return ic, {"eq": eq, **p}


print("\n== E-003 training history (evaluated point-in-time)", flush=True)
ra, rb = walk(lab_srv), walk(lab_pit)
ica, pa = report("A: trained on survivors", ra)
icb, pb = report("B: trained on point-in-time", rb)
diff = (icb - ica).dropna()
t = diff.mean() / diff.std() * np.sqrt(len(diff))
ok = t >= 2 and pb["cagr"] >= pa["cagr"] and pb["sharpe"] >= pa["sharpe"]
print(f"paired IC t = {t:.2f} -> {'ACCEPT' if ok else 'REJECT'}", flush=True)


def trailing(eq_rules: P.Rules, trail: float, raw: pd.DataFrame) -> pd.Series:
    """Trailing stop on closes: sell when close < (1 - trail) x highest close since entry."""
    return P.simulate(M.smooth_scores(raw, feats_pit), close, eq_rules, 100_000, "weekly",
                      risk_off=risk_off, trail_stop=trail).equity


print("\n== E-004 trailing stop (choose on 2015-2020, confirm on 2021-2026)", flush=True)
raw = rb if ok else ra
base = report("no trailing stop", raw)[1]["eq"]


def part(e: pd.Series, a: str, b_: str) -> dict:
    e = e[(e.index >= a) & (e.index <= b_)]
    return P.performance(e / e.iloc[0])


best, best_score = None, None
for trail in (0.10, 0.15, 0.20, 0.25):
    e = trailing(P.Rules(), trail, raw)
    tune, conf = part(e, "2015-01-01", "2020-12-31"), part(e, "2021-01-01", "2026-12-31")
    print(f"trail {trail:.0%}: 2015-20 CAGR {tune['cagr']:6.1%} Sharpe {tune['sharpe']:.2f} "
          f"worst {tune['max_drawdown']:6.1%} | 2021-26 CAGR {conf['cagr']:6.1%} "
          f"Sharpe {conf['sharpe']:.2f} worst {conf['max_drawdown']:6.1%}", flush=True)
    if best_score is None or tune["sharpe"] > best_score:
        best, best_score, best_conf = trail, tune["sharpe"], conf
b_conf = part(base, "2021-01-01", "2026-12-31")
ok4 = (best_conf["max_drawdown"] - b_conf["max_drawdown"] >= 0.05
       and best_conf["cagr"] >= b_conf["cagr"] - 0.01 and best_conf["sharpe"] >= b_conf["sharpe"])
print(f"chosen on 2015-20: {best:.0%}; confirmation 2021-26 vs no stop: worst "
      f"{best_conf['max_drawdown']:.1%} vs {b_conf['max_drawdown']:.1%}, CAGR "
      f"{best_conf['cagr']:.1%} vs {b_conf['cagr']:.1%} -> {'ACCEPT' if ok4 else 'REJECT'}",
      flush=True)
