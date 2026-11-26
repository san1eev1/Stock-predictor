"""Experiment E-015 (GitHub, research.yml): do the morning candles help the intraday picks?

The candle book (model shortlist re-ranked by five 9:15-9:45 candle checks, every trade cut
as soon as it slips below entry) lost 22% live with 9% winning trades. The literature on
candlestick patterns is mostly negative after costs (Marshall, Young & Rose 2006; Duvinage
et al. 2013; Thai and South African replications); the one strong opening-range result is
for US "stocks in play" with 5-minute ranges (Zarattini & Barbon 2024).

On the 'until close' model's walk-forward scores over every intraday day:
  1. each candle check alone: does agreeing with it predict 9:45 -> 15:15 (IC, hit rate)?
  2. re-rank weight w in {0, .25, .5, .75, 1}: top 3 a side from the 25 + 25 shortlist
  3. exits: hold to 15:15 | 1% stop | cut at entry (the candle book's rule), using each
     stock's high / low after 9:45 (stop assumed hit at its level)
Net of 0.37% (Angel-like) and 0.19% (0.03% broker) round trips. Fixed rule: change the
live candle book only for a variant with net t >= 2 that beats w = 0.5 / cut-at-entry with
paired t >= 2."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T
from stockpredictor.paper import engine as E
from stockpredictor.paper import intraday as PI

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
ctx = E.MarketContext.load(d)
feats = T.intraday_feats(ctx, d, MI.CLOSE)
scores = MI.walk_forward(feats, params=MI.current_params(MI.CLOSE))
need = ["symbol", "date", "open", "h30", "l30", "c30", "vwap30", "orb15", "up1_share",
        "high_after", "low_after", "px_1515"]
f = scores.merge(feats[[c for c in need if c in feats]], on=["symbol", "date"])
f["r"] = f["px_1515"] / f["c30"] - 1
f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "r", "c30"])
f = f[f["r"].abs() < 0.25]
print(f"out-of-sample days {f['date'].nunique()}, rows {len(f):,}; columns "
      f"{[c for c in need if c not in f]} missing", flush=True)


def stats(x) -> str:
    x = pd.Series(x).dropna()
    t = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan
    return f"{x.mean() * 1e4:+6.1f} bp  t={t:+5.2f}  win {np.mean(x > 0):.0%}  n={len(x)}"


# 1. each check alone (long side; the short side mirrors it)
checks = {
    "30-min candle closed up": f["c30"] > f["open"],
    "closed in top half of range": (f["c30"] - f["l30"]) / (f["h30"] - f["l30"]).replace(0, np.nan) > 0.5,
    "above VWAP": f["c30"] > f["vwap30"],
    "broke first-15-min range up": f["orb15"] > 0 if "orb15" in f else pd.Series(False, f.index),
    "most 1-min candles green": f["up1_share"] > 0.5 if "up1_share" in f else pd.Series(False, f.index),
}
print("\n1. Each candle check: stocks agreeing minus disagreeing, 9:45 -> 15:15, per day", flush=True)
for name, m in checks.items():
    daily = f.assign(m=m).groupby("date").apply(
        lambda g: g.loc[g["m"], "r"].mean() - g.loc[~g["m"], "r"].mean())
    print(f"  {name:32} {stats(daily)}", flush=True)
pts = sum(m.astype(int) for m in checks.values())
ic = f.assign(p=pts).groupby("date").apply(lambda g: g["p"].corr(g["r"], method="spearman"))
print(f"  all five (0-5 points) IC: {stats(ic)}", flush=True)


# 2 + 3. re-rank weights and exits
def trade_ret(rows: pd.DataFrame, side: str, exit_: str) -> pd.Series:
    sign = 1 if side == "long" else -1
    r = sign * rows["r"]
    if exit_ == "hold":
        return r
    if exit_ == "stop1":
        hit = (rows["low_after"] / rows["c30"] - 1 <= -0.01) if side == "long" else \
              (rows["high_after"] / rows["c30"] - 1 >= 0.01)
        return r.where(~hit, -0.01)
    hit = (rows["low_after"] < rows["c30"]) if side == "long" else (rows["high_after"] > rows["c30"])
    return r.where(~hit, -0.0005)                      # cut at entry: a little slippage


def day_ret(g: pd.DataFrame, w: float, exit_: str, k: int = 3) -> float:
    g = g.sort_values("score")
    out = []
    for side, short in (("long", g.tail(PI.N_CANDIDATES).iloc[::-1]),
                        ("short", g.head(PI.N_CANDIDATES))):
        rk = PI.shortlist_by_candles(short, side) if w > 0 else short.assign(model_rank=range(1, len(short) + 1))
        if w > 0:
            mp = 1 - (rk["model_rank"] - 1) / max(1, len(rk) - 1)
            rk = rk.assign(final=(1 - w) * mp + w * rk["candle"] / len(PI.CANDLE_CHECKS)) \
                   .sort_values(["final", "model_rank"], ascending=[False, True])
        out.append(trade_ret(rk.head(k), side, exit_).mean())
    return float(np.mean(out))


res = {}
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    for exit_ in ("hold", "stop1", "cut"):
        res[(w, exit_)] = f.groupby("date").apply(lambda g: day_ret(g, w, exit_))
live = res[(0.5, "cut")]
for cost_name, c in (("0.37%", 0.0037), ("0.19%", 0.0019)):
    print(f"\n2-3. Top 3 a side, per position, NET of {cost_name} round trip:", flush=True)
    for key, g in res.items():
        net = g - c
        diff = (net - (live - c)).dropna()
        tp = diff.mean() / diff.std() * np.sqrt(len(diff)) if diff.std() > 0 else np.nan
        tn = net.mean() / net.std() * np.sqrt(net.count()) if net.std() > 0 else np.nan
        print(f"  weight {key[0]:.2f} exit {key[1]:5}: {stats(net)}  vs live rule t={tp:+5.2f}"
              f"{'  <- ADOPT' if tn >= 2 and tp >= 2 else ''}", flush=True)
