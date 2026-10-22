"""Experiment E-002 (GitHub, research.yml): CPCV + DoubleEnsemble for the long-term model.

1. Combinatorial purged CV (skfolio/López de Prado idea, own code): the weekly data is cut
   into N_GROUPS time blocks; every pair of blocks is a test set; training rows within
   EMBARGO_DAYS of a test block are purged. Gives a *distribution* of out-of-sample IC
   instead of one walk-forward path, so model comparisons are less noisy.
2. DoubleEnsemble-lite (idea from microsoft/qlib@5435523 contrib/model/double_ensemble.py,
   MIT, re-implemented): K sub-models; each next one down-weights the samples the ensemble
   so far fits worst (likely noise) and drops the weakest 20% of features by gain.

Fixed acceptance rule: DoubleEnsemble is adopted only if its IC beats the baseline on at
least 80% of CPCV paths AND the paired t over paths >= 2. Mixing past and future blocks
is fine for comparing two models on equal footing; it is not a profit estimate.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import run as bt
from stockpredictor.models import longterm as M

N_GROUPS, K_TEST = 6, 2
K_MODELS = 3

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
_, _, _, labeled = bt.load_all(d)
lab = labeled.dropna(subset=["target"]).copy()
lab = lab[lab["date"] >= "2008-01-01"]
cols = M._model_columns(lab)
dates = np.array(sorted(lab["date"].unique()))
blocks = np.array_split(dates, N_GROUPS)
emb = pd.Timedelta(days=M.EMBARGO_DAYS)


def ic(df: pd.DataFrame, pred: np.ndarray) -> float:
    x = df.assign(p=pred)
    return float(x.groupby("date").apply(lambda g: g["p"].corr(g["target"], method="spearman")).mean())


def fit_baseline(train):
    return lambda df: M._fit(train, cols).predict(df[cols])


def fit_double(train):
    models, feats = [], list(cols)
    tr = train.copy()
    tr["fb_weight"] = 1.0
    pred_sum = np.zeros(len(tr))
    for k in range(K_MODELS):
        m = M._fit(tr, feats)
        models.append((m, list(feats)))
        pred_sum += m.predict(tr[feats])
        loss = (pred_sum / (k + 1) - tr["target"].to_numpy()) ** 2
        bins = pd.qcut(pd.Series(loss).rank(method="first"), 5, labels=False).to_numpy()
        tr["fb_weight"] = 1.0 / (1.0 + bins * 0.25)          # worst-fit fifth: weight 0.5
        try:
            gain = pd.Series(np.mean([b.feature_importance("gain") for b in m.boosters], axis=0),
                             index=m.used or feats)
            keep = gain.sort_values(ascending=False).index[:max(10, int(len(gain) * 0.8))]
            feats = [c for c in feats if c in set(keep)]
        except Exception:
            pass
    return lambda df: np.mean([m.predict(df[f]) for m, f in models], axis=0)


rows = []
for test_ids in itertools.combinations(range(N_GROUPS), K_TEST):
    test_dates = np.concatenate([blocks[i] for i in test_ids])
    lo_hi = [(blocks[i][0], blocks[i][-1]) for i in test_ids]
    near = np.zeros(len(lab), bool)
    for lo, hi in lo_hi:
        near |= (lab["date"] >= lo - emb).to_numpy() & (lab["date"] <= hi + emb).to_numpy()
    train, test = lab[~near], lab[lab["date"].isin(test_dates)]
    a = ic(test, fit_baseline(train)(test))
    b = ic(test, fit_double(train)(test))
    rows.append((test_ids, a, b))
    print(f"path {test_ids}: baseline IC {a:.4f} | double-ensemble IC {b:.4f}", flush=True)

r = pd.DataFrame(rows, columns=["test", "base", "double"])
diff = r["double"] - r["base"]
t = diff.mean() / diff.std() * np.sqrt(len(diff)) if diff.std() else 0.0
share = (diff > 0).mean()
print(f"\nbaseline IC {r['base'].mean():.4f} ± {r['base'].std():.4f} | "
      f"double {r['double'].mean():.4f} ± {r['double'].std():.4f}", flush=True)
print(f"double better on {share:.0%} of {len(r)} paths, paired t = {t:.2f} -> "
      f"{'ACCEPT' if share >= 0.8 and t >= 2 else 'REJECT'}", flush=True)
