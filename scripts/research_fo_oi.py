"""Experiment E-011a (GitHub, research.yml): F&O positioning features for the long-term model.
Source idea: option / futures open interest carries informed positioning (Pan & Poteshman
2006; option-open-interest literature); Indian traders' 'long/short build-up' reading.
Features (data/fobhav.features, known after each close): futures OI change 1/5/20 days,
OI change x price direction (build-up), put-call OI ratio and its 5-day change, futures
basis, option activity. Only F&O stocks get values (others NaN).
Fixed rule: ACCEPT only if (1) walk-forward 2015-2026 CAGR and Sharpe after costs are not
worse AND (2) CPCV (15 paths) IC is better on >= 80% of paths with paired t >= 2."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.data import fobhav as FB
from stockpredictor.models import longterm as M

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
fo = FB.load(d)
if fo.empty:
    sys.exit("no F&O bhavcopy yet (fobhav-update in the data workflow)")
print(f"F&O history: {fo['date'].min():%Y-%m-%d} .. {fo['date'].max():%Y-%m-%d}, "
      f"{fo['symbol'].nunique()} stocks, {fo['date'].nunique()} days", flush=True)
daily, indices, feats, labeled = bt.load_all(d)
close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
extra = FB.features(fo, close)
cols_new = [c for c in extra.columns if c not in ("symbol", "date")]
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
    print(f"{name:16} IC {ic.mean():.4f} (t {ic.mean() / ic.std() * np.sqrt(len(ic)):.1f}) | "
          f"CAGR {p['cagr']:.1%} Sharpe {p['sharpe']:.2f} worst {p['max_drawdown']:.1%}", flush=True)
    return ic, p


base_ic, base_p = run(labeled, feats, "baseline")
lab2 = labeled.merge(extra, on=["symbol", "date"], how="left")
new_ic, new_p = run(lab2, feats.merge(extra, on=["symbol", "date"], how="left"), "+ F&O OI")
diff = (new_ic - base_ic).dropna()
t_wf = diff.mean() / diff.std() * np.sqrt(len(diff))
wf_ok = new_p["cagr"] >= base_p["cagr"] and new_p["sharpe"] >= base_p["sharpe"]
print(f"walk-forward paired IC {diff.mean():+.4f}, t = {t_wf:.2f}; backtest not worse: {wf_ok}", flush=True)

# CPCV on the years with F&O data (2012+)
lab = lab2.dropna(subset=["target"])
lab = lab[lab["date"] >= max(pd.Timestamp("2012-03-01"), fo["date"].min() + pd.Timedelta(days=40))]
base_cols = M._model_columns(labeled)
new_cols = base_cols + cols_new
dates = np.array(sorted(lab["date"].unique()))
blocks = np.array_split(dates, 6)
emb = pd.Timedelta(days=M.EMBARGO_DAYS)


def ic(df, pred):
    x = df.assign(p=pred)
    return float(x.groupby("date").apply(lambda g: g["p"].corr(g["target"], method="spearman")).mean())


rows = []
for test_ids in itertools.combinations(range(6), 2):
    near = np.zeros(len(lab), bool)
    for i in test_ids:
        near |= ((lab["date"] >= blocks[i][0] - emb) & (lab["date"] <= blocks[i][-1] + emb)).to_numpy()
    tr, te = lab[~near], lab[lab["date"].isin(np.concatenate([blocks[i] for i in test_ids]))]
    a = ic(te, M._fit(tr, base_cols).predict(te[base_cols]))
    b = ic(te, M._fit(tr, new_cols).predict(te[new_cols]))
    rows.append((a, b))
    print(f"path {test_ids}: baseline {a:.4f} | + F&O OI {b:.4f}", flush=True)
r = pd.DataFrame(rows, columns=["base", "new"])
dd = r["new"] - r["base"]
t = dd.mean() / dd.std() * np.sqrt(len(dd))
share = (dd > 0).mean()
ok = wf_ok and share >= 0.8 and t >= 2
print(f"CPCV: baseline {r['base'].mean():.4f} vs new {r['new'].mean():.4f}; better on {share:.0%} "
      f"of paths, paired t = {t:.2f} -> {'ACCEPT' if ok else 'REJECT'}", flush=True)
