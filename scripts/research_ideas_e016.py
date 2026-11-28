"""Experiment E-016 (GitHub, research.yml): five ideas for the intraday close book.

All on the 'until close' model's walk-forward scores over every intraday day, 3 a side
(E-014's best), held to 15:15, net of 0.33% (₹50k books at Angel-like rates) and 0.19%
(0.03% broker) round trips. Fixed rule: adopt if net t >= 2 AND paired t >= 2 vs the base
(model top/bottom 3).

  A. Candle REVERSAL (E-015 found the morning candle reverses): re-rank the 25 + 25 shortlist
     against the candle score, weights 0.25 / 0.5 (Heston, Korajczyk & Sadka 2010: intraday
     reversal from order imbalance and bid-ask bounce).
  B. Pure 30-minute reversal (no model): buy the 3 biggest 9:15-9:45 losers, sell the 3
     biggest gainers; and the model blended with it (rank average).
  C. Inverse-volatility sizing (Moreira & Muir 2017, volatility-managed portfolios): each
     position sized by 1 / ATR% instead of equally.
  D. Strong-signal days only: trade only days whose top-minus-bottom score spread is above
     its trailing 60-day median / 70th percentile (does the model know when it knows?).
  E. Market regime: base result split by India VIX tercile and by the Nifty's morning move.
"""
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
        "px_1515", "r30", "atr_pct", "vix", "mkt_r30"]
f = scores.merge(feats[[c for c in need if c in feats]], on=["symbol", "date"])
f["r"] = f["px_1515"] / f["c30"] - 1
f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "r", "c30"])
f = f[f["r"].abs() < 0.25]
print(f"days {f['date'].nunique()}, rows {len(f):,}; missing {[c for c in need if c not in f]}",
      flush=True)
K = 3
COSTS = {"0.33%": 0.0033, "0.19%": 0.0019}


def stats(x) -> str:
    x = pd.Series(x).dropna()
    t = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan
    return f"{x.mean() * 1e4:+6.1f} bp  t={t:+5.2f}  win {np.mean(x > 0):.0%}  n={len(x)}"


def per_pos(longs, shorts, w_long=None, w_short=None) -> float:
    """Average return per position (longs + shorts), optional weights."""
    lr, sr = longs["r"].to_numpy(), -shorts["r"].to_numpy()
    if w_long is None:
        return float(np.mean(np.concatenate([lr, sr])))
    w = np.concatenate([w_long, w_short])
    return float(np.sum(np.concatenate([lr, sr]) * w) / np.sum(w))


def base(g):
    g = g.sort_values("score")
    return per_pos(g.tail(K), g.head(K))


def candle_rev(g, w):
    g = g.sort_values("score")
    out = []
    for side, short in (("long", g.tail(PI.N_CANDIDATES).iloc[::-1]), ("short", g.head(PI.N_CANDIDATES))):
        rk = PI.shortlist_by_candles(short, side)
        mp = 1 - (rk["model_rank"] - 1) / max(1, len(rk) - 1)
        rk = rk.assign(final=(1 - w) * mp + w * (1 - rk["candle"] / len(PI.CANDLE_CHECKS)))
        out.append(rk.sort_values("final", ascending=False).head(K))
    return per_pos(out[0], out[1])


def pure_rev(g):
    g = g.dropna(subset=["r30"]).sort_values("r30")
    return per_pos(g.head(K), g.tail(K))            # buy losers, sell gainers


def blend_rev(g):
    g = g.dropna(subset=["r30"])
    s = g["score"].rank(pct=True) + (-g["r30"]).rank(pct=True)
    g = g.assign(b=s).sort_values("b")
    return per_pos(g.tail(K), g.head(K))


def inv_vol(g):
    g = g.dropna(subset=["atr_pct"]).sort_values("score")
    lo, hi = g.head(K), g.tail(K)
    return per_pos(hi, lo, 1 / hi["atr_pct"].clip(lower=0.003).to_numpy(),
                   1 / lo["atr_pct"].clip(lower=0.003).to_numpy())


variants = {"base: model top/bottom 3": base,
            "A candle reversal w=0.25": lambda g: candle_rev(g, 0.25),
            "A candle reversal w=0.5": lambda g: candle_rev(g, 0.5),
            "B pure 30-min reversal": pure_rev,
            "B model + 30-min reversal": blend_rev,
            "C inverse-volatility sizing": inv_vol}
res = {name: f.groupby("date").apply(fn) for name, fn in variants.items()}
b = res["base: model top/bottom 3"]

# D. strong-signal days
spread = f.groupby("date").apply(lambda g: g["score"].nlargest(K).mean() - g["score"].nsmallest(K).mean())
for q in (0.5, 0.7):
    thr = spread.rolling(60, min_periods=20).quantile(q).shift()
    res[f"D strong days only (> {q:.0%} of last 60)"] = b[spread > thr]

for cname, c in COSTS.items():
    print(f"\nNET of {cname} round trip, per position per day:", flush=True)
    for name, r in res.items():
        net = r - c
        al = pd.concat([net, b - c], axis=1).dropna()
        diff = al.iloc[:, 0] - al.iloc[:, 1]
        tp = diff.mean() / diff.std() * np.sqrt(len(diff)) if diff.std() > 0 else np.nan
        tn = net.mean() / net.std() * np.sqrt(net.count()) if net.std() > 0 else np.nan
        tag = "  <- ADOPT" if tn >= 2 and tp >= 2 and not name.startswith("base") else ""
        print(f"  {name:38} {stats(net)}  vs base t={tp:+5.2f}{tag}", flush=True)

# D note: days traded
print(f"\nD: strong-day filter keeps {(spread > spread.rolling(60, min_periods=20).median().shift()).mean():.0%}"
      " of days at the median threshold", flush=True)

# E. regimes (gross, base)
day = f.groupby("date").agg(vix=("vix", "first"), mkt=("mkt_r30", "first"))
day["base"] = b
if day["vix"].notna().sum() > 100:
    day["vix_t"] = pd.qcut(day["vix"], 3, labels=["low VIX", "mid VIX", "high VIX"])
    print("\nE. Base, gross, by India VIX tercile:", flush=True)
    for k, g in day.groupby("vix_t", observed=True):
        print(f"  {k:10} {stats(g['base'])}", flush=True)
if day["mkt"].notna().sum() > 100:
    day["mkt_t"] = pd.qcut(day["mkt"], 3, labels=["market down at 9:45", "flat", "market up at 9:45"])
    print("E. Base, gross, by the market's first 30 minutes:", flush=True)
    for k, g in day.groupby("mkt_t", observed=True):
        print(f"  {k:22} {stats(g['base'])}", flush=True)
