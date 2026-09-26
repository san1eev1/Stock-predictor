"""Self-diagnosis: where do the models' predictions work, where do they fail, what limits
them? (GitHub, research.yml). All scores are walk-forward (never judged by a model that saw
the day). IC = rank correlation between score and the real outcome, per day (0 = random).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import run as bt
from stockpredictor.models import intraday as MI
from stockpredictor.models import longterm as M
from stockpredictor.models import trainer as T
from stockpredictor.paper.engine import MarketContext

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")


def ic_by(df: pd.DataFrame, group: str, score="score", outcome="outcome", min_n=10) -> pd.Series:
    def one(g):
        return g[score].rank().corr(g[outcome].rank()) if len(g) >= min_n else np.nan
    per_day = df.groupby(["date", group]).apply(one, include_groups=False).dropna()
    return per_day.groupby(level=group).agg(["mean", "count"])


def bins(x: pd.Series, n: int, labels=None) -> pd.Series:
    """Per-day n-tiles by rank (works for any group size)."""
    b = np.ceil(x.rank(pct=True) * n).clip(1, n).astype("Int64")
    return b if labels is None else b.map(dict(enumerate(labels, 1)))


def show(title: str, table) -> None:
    print(f"\n-- {title}", flush=True)
    print(table.round(4).to_string(), flush=True)


# ============================== LONG-TERM ==============================
print("=" * 30, "LONG-TERM (1-week model)", "=" * 30, flush=True)
daily, indices, feats, labeled = bt.load_all(d)
raw = M.walk_forward(labeled, feats, 2015)
m = raw.merge(labeled[["symbol", "date", "fwd_excess"]], on=["symbol", "date"]).dropna()
m = m.rename(columns={"fwd_excess": "outcome"})
f = feats.set_index(["symbol", "date"])
for c in ("vol_63", "turnover_log", "mom_12_1", "mkt_above_ma200", "vix_pct_252"):
    if c in f:
        m[c] = f[c].reindex(pd.MultiIndex.from_frame(m[["symbol", "date"]])).to_numpy()
m["year"] = m["date"].dt.year
show("IC by year", ic_by(m, "year"))
m["market"] = np.where(m["mkt_above_ma200"] == 1, "Nifty above 200-day avg", "below (falling)")
show("IC by market trend", ic_by(m, "market"))
m["vix"] = pd.qcut(m["vix_pct_252"], 3, labels=["calm", "normal", "nervous"])
show("IC by VIX level", ic_by(m, "vix"))
for col, name in (("vol_63", "volatility"), ("turnover_log", "liquidity"), ("mom_12_1", "12-month momentum")):
    m[name] = m.groupby("date")[col].transform(lambda x: bins(x, 3, ["low", "mid", "high"]))
    show(f"IC within {name} thirds (does it rank well among similar stocks?)", ic_by(m, name))
m["decile"] = m.groupby("date")["score"].transform(lambda x: bins(x, 10))
dec = m.groupby("decile")["outcome"].agg(["mean", lambda x: (x > 0).mean()])
dec.columns = ["avg 1-week excess return", "share beating Nifty"]
show("Where is the edge? (decile 10 = top picks)", dec)
# learning curve: training history length vs accuracy on the last 3 years
print("\n-- More data or better model? (IC on 2023-2026 by training history length)", flush=True)
last3 = labeled["date"].max().year - 2
for years in (3, 6, 10, 20):
    lab = labeled[labeled["date"] >= pd.Timestamp(year=last3 - years, month=1, day=1)]
    s = M.walk_forward(lab, feats, last3)
    ic = M.information_coefficient(s, labeled)
    print(f"   trained on the previous {years:2d} years: IC {ic.mean():.4f}", flush=True)
# feature importance and its stability across yearly models
imps = []
for year in range(2016, labeled["date"].max().year + 1):
    tr = labeled[labeled["date"] < pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=M.EMBARGO_DAYS)]
    mdl = M.LongTermModel.train(tr)
    imps.append(mdl.importance().rename(year))
imp = pd.concat(imps, axis=1).fillna(0)
imp = imp / imp.sum()
stab = imp.rank(ascending=False)
summary = pd.DataFrame({"avg share": imp.mean(axis=1), "best rank": stab.min(axis=1),
                        "worst rank": stab.max(axis=1)}).sort_values("avg share", ascending=False)
show("Most used inputs (share of the model's decisions) and how stable their rank is", summary.head(20))
show("Least used inputs", summary.tail(10))

# ============================== INTRADAY ==============================
ctx = MarketContext.load(d)
for target in MI.TARGETS:
    print("\n" + "=" * 30, f"INTRADAY ({target.label})", "=" * 30, flush=True)
    fi = T.intraday_feats(ctx, d, target)
    print(f"history: {fi['date'].nunique()} days, {fi.dropna(subset=['target'])['date'].nunique()} "
          f"usable (>= {T.MIN_STOCKS_PER_DAY} stocks), {fi['symbol'].nunique()} stocks", flush=True)
    s = MI.walk_forward(fi, params=MI.current_params(target), last_days=240)
    x = s.merge(fi, on=["symbol", "date"]).rename(columns={"target_ret": "outcome"})
    x = x.dropna(subset=["outcome"])
    show("IC by month", ic_by(x.assign(month=x["date"].dt.to_period("M").astype(str)), "month", min_n=50))
    x["gap_size"] = pd.cut(x["gap"].abs(), [-1, 0.005, 0.015, 0.03, 1],
                           labels=["<0.5%", "0.5-1.5%", "1.5-3%", ">3%"])
    show("IC among stocks by gap size", ic_by(x, "gap_size"))
    if "day_oddness" in x:
        x["morning"] = np.where(x["day_oddness"] > 3, "unusual", "normal")
        show("IC on normal vs unusual mornings", ic_by(x, "morning", min_n=50))
    x["weekday"] = x["date"].dt.day_name()
    show("IC by weekday", ic_by(x, "weekday", min_n=50))
    for col, name in (("atr_pct", "volatility"), ("vol30_adv", "early volume")):
        if col in x:
            x[name] = x.groupby("date")[col].transform(lambda v: bins(v, 3, ["low", "mid", "high"]))
            show(f"IC within {name} thirds", ic_by(x, name))
    x["decile"] = x.groupby("date")["score"].transform(lambda v: bins(v, 10))
    show("Where is the edge? (decile 10 = top buys, 1 = top sells): average move",
         x.groupby("decile")["outcome"].agg(["mean", lambda v: (v > 0).mean()]))
    print("\n-- More data or better model? (IC on the last 120 days by training history)", flush=True)
    days = sorted(fi.dropna(subset=["target"])["date"].unique())
    for n in (60, 120, 250, 500, len(days)):
        keep = set(days[-(120 + n):])
        sc = MI.walk_forward(fi[fi["date"].isin(keep)], params=MI.current_params(target), last_days=120)
        mm = sc.merge(fi[["symbol", "date", "target_ret"]], on=["symbol", "date"])
        ic = mm.groupby("date").apply(lambda g: g["score"].rank().corr(g["target_ret"].rank()),
                                      include_groups=False).mean()
        print(f"   trained on up to {n:4d} earlier days: IC {ic:.4f}", flush=True)
    model = MI.IntradayModel.train(fi)
    imp = model.importance()
    show("Most used inputs", (imp / imp.sum()).head(15))
