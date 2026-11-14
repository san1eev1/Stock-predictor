"""Experiment E-010b (GitHub, research.yml): overnight vs intraday momentum features
(Lou, Polk & Skouras 2019) through the CPCV harness (15 paths, 14-day purge), because E-010's
walk-forward gain (IC +0.0034, CAGR 22.5% -> 25.6%, Sharpe 0.88 -> 0.99) had t = 1.64 only.
Fixed rule: ACCEPT if IC is better on >= 80% of the 15 paths AND paired t >= 2."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import run as bt
from stockpredictor.models import longterm as M

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats, labeled = bt.load_all(d)
close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
opn = daily.pivot(index="date", columns="symbol", values="open").reindex(close.index)
on, intra = opn / close.shift(1) - 1, close / opn - 1
parts = []
for w in (21, 63, 252):
    for name, x in ((f"overnight_{w}", on), (f"intraday_{w}", intra)):
        s = x.rolling(w, min_periods=w // 2).sum().stack(future_stack=True).rename(name)
        parts.append(s)
extra = pd.concat(parts, axis=1).reset_index().rename(columns={"level_0": "date", "level_1": "symbol"})
extra.columns = ["date", "symbol", *extra.columns[2:]]
lab = labeled.merge(extra, on=["symbol", "date"], how="left").dropna(subset=["target"])
lab = lab[lab["date"] >= "2008-01-01"]
base_cols = M._model_columns(labeled)
new_cols = base_cols + [c for c in extra.columns if c not in ("date", "symbol")]
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
        lo, hi = blocks[i][0], blocks[i][-1]
        near |= ((lab["date"] >= lo - emb) & (lab["date"] <= hi + emb)).to_numpy()
    tr, te = lab[~near], lab[lab["date"].isin(np.concatenate([blocks[i] for i in test_ids]))]
    a = ic(te, M._fit(tr, base_cols).predict(te[base_cols]))
    b = ic(te, M._fit(tr, new_cols).predict(te[new_cols]))
    rows.append((a, b))
    print(f"path {test_ids}: baseline {a:.4f} | + overnight/intraday {b:.4f}", flush=True)
r = pd.DataFrame(rows, columns=["base", "new"])
diff = r["new"] - r["base"]
t = diff.mean() / diff.std() * np.sqrt(len(diff))
share = (diff > 0).mean()
print(f"baseline {r['base'].mean():.4f} vs new {r['new'].mean():.4f}; better on {share:.0%} of "
      f"paths, paired t = {t:.2f} -> {'ACCEPT' if share >= 0.8 and t >= 2 else 'REJECT'}", flush=True)
