"""Experiment E-011b (GitHub, research.yml): yesterday's F&O positioning for the intraday
'until close' model. Same features as E-011a, taken from the previous trading day (known
before the open). Walk-forward (monthly, out of sample) on the same intraday history.
Fixed rule: ACCEPT if daily IC improves with paired t >= 2 and more days improve than not."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.data import fobhav as FB
from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T
from stockpredictor.paper import engine as E

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
fo = FB.load(d)
if fo.empty:
    sys.exit("no F&O bhavcopy yet")
ctx = E.MarketContext.load(d)
feats = T.intraday_feats(ctx, d, MI.CLOSE)
close = ctx.daily.pivot(index="date", columns="symbol", values="close").sort_index()
extra = FB.features(fo, close)
days = sorted(feats["date"].unique())
nxt = dict(zip(days[:-1], days[1:]))                 # yesterday's values -> today's row
extra["date"] = extra["date"].map(nxt)
extra = extra.dropna(subset=["date"])
feats2 = feats.merge(extra, on=["symbol", "date"], how="left")
print(f"intraday days {len(days)}, rows with F&O values "
      f"{feats2['fo_oi_chg_1'].notna().mean():.0%}", flush=True)
params = MI.current_params(MI.CLOSE)


def daily_ic(f):
    s = MI.walk_forward(f, params=params).merge(f[["symbol", "date", "target"]],
                                                on=["symbol", "date"])
    return s.groupby("date").apply(lambda g: g["score"].corr(g["target"], method="spearman"))


a, b = daily_ic(feats), daily_ic(feats2)
diff = (b - a).dropna()
t = diff.mean() / diff.std() * np.sqrt(len(diff))
print(f"baseline IC {a.mean():.4f} | + F&O OI {b.mean():.4f} | paired {diff.mean():+.4f}, "
      f"t = {t:.2f}, better on {(diff > 0).mean():.0%} of {len(diff)} days -> "
      f"{'ACCEPT' if t >= 2 and (diff > 0).mean() > 0.5 else 'REJECT'}", flush=True)
