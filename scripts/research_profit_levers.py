"""Experiment E-014 (GitHub, research.yml): what would make the intraday books profitable?

The live P&L attribution (research/known_failure_modes.md) showed stock picking about zero and
costs (0.37% a round trip) larger than the edge. This tests the levers a trading desk would
pull, on the 'until close' model's walk-forward (out-of-sample) scores over all intraday days:

  A. costs: Angel-like 0.37% | 0.03%-broker 0.19% | + limit orders 0.11% | stock futures 0.13%
  B. trades per side: 3, 5, 10
  C. sector-neutral pairs (best vs worst stock in the same sector) vs plain top / bottom
  D. exit time: 12:30 vs 15:15
  E. news since the last close (company + sector, learned relevance x tone): does it predict
     9:45 -> 15:15 on the days it exists?

Daily long-short return = mean(long moves) - mean(short moves), each position equal size,
minus the round-trip cost. Fixed rule: a variant is worth adopting if its NET daily return is
positive with t >= 2 over all days, and better than the current way (top/bottom 5, close exit)
with paired t >= 2."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.data import intraday as I
from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T
from stockpredictor.paper import engine as E

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
ctx = E.MarketContext.load(d)
feats = T.intraday_feats(ctx, d, MI.CLOSE)
params = MI.current_params(MI.CLOSE)
scores = MI.walk_forward(feats, params=params)
cols = ["symbol", "date", "c30", "px_1515", I.EXIT_COL]
f = scores.merge(feats[[c for c in cols if c in feats]], on=["symbol", "date"])
f = f.merge(ctx.universe[["symbol", "industry"]], on="symbol", how="left")
f["r_close"] = f["px_1515"] / f["c30"] - 1
f["r_1230"] = f[I.EXIT_COL] / f["c30"] - 1
f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "r_close", "r_1230"])
f = f[(f["r_close"].abs() < 0.25) & (f["r_1230"].abs() < 0.25)]
print(f"out-of-sample days {f['date'].nunique()}, rows {len(f):,}", flush=True)

COSTS = {"Angel-like 0.37%": 0.0037, "0.03% broker 0.19%": 0.0019,
         "+ limit orders 0.11%": 0.0011, "stock futures 0.13%": 0.0013}


def plain(g, k, col):
    g = g.sort_values("score")
    return g[col].tail(k).mean() - g[col].head(k).mean()


def neutral(g, k, col):
    """k sector pairs with the widest score gap: long the best, short the worst in each."""
    pairs = []
    for _, s in g.groupby("industry"):
        if len(s) >= 2:
            s = s.sort_values("score")
            pairs.append((s["score"].iloc[-1] - s["score"].iloc[0],
                          s[col].iloc[-1] - s[col].iloc[0]))
    if not pairs:
        return np.nan
    pairs.sort(key=lambda p: -p[0])
    return float(np.mean([p[1] for p in pairs[:k]]))


def stats(x: pd.Series) -> str:
    x = x.dropna()
    t = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan
    return f"{x.mean() * 1e4:+6.1f} bp/day  t={t:+5.2f}  win {np.mean(x > 0):.0%}"


rows = {}
for name, fn in (("top/bottom", plain), ("sector pairs", neutral)):
    for k in (3, 5, 10):
        for exit_, col in (("15:15", "r_close"), ("12:30", "r_1230")):
            gross = f.groupby("date").apply(lambda g: fn(g, k, col))
            rows[(name, k, exit_)] = gross
base = rows[("top/bottom", 5, "15:15")]
print("\nGROSS (before costs), per position pair of long + short:", flush=True)
for key, g in rows.items():
    print(f"  {key[0]:12} k={key[1]:<2} exit {key[2]}: {stats(g)}", flush=True)
print("\nNET of round-trip costs (both legs pay):", flush=True)
for cname, c in COSTS.items():
    print(f"  costs {cname}", flush=True)
    for key, g in rows.items():
        net = g - 2 * c
        diff = (net - (base - 2 * COSTS["Angel-like 0.37%"])).dropna()
        tp = diff.mean() / diff.std() * np.sqrt(len(diff)) if diff.std() > 0 else np.nan
        ok = (net.mean() > 0 and net.mean() / net.std() * np.sqrt(net.count()) >= 2 and tp >= 2)
        print(f"    {key[0]:12} k={key[1]:<2} exit {key[2]}: {stats(net)}  vs now t={tp:+5.2f}"
              f"{'  <- ADOPT' if ok else ''}", flush=True)

# Recent two years only (regimes change)
recent = f["date"] >= f["date"].max() - pd.Timedelta(days=730)
print("\nLast two years, gross:", flush=True)
for key in (("top/bottom", 5, "15:15"), ("sector pairs", 5, "15:15"), ("top/bottom", 5, "12:30"),
            ("sector pairs", 10, "15:15")):
    g = rows[key]
    print(f"  {key}: {stats(g[g.index >= f['date'].max() - pd.Timedelta(days=730)])}", flush=True)

# E. News since the last close
try:
    from stockpredictor.paper import lab as LAB

    news = ctx.news
    days = sorted(f["date"].unique())
    out = []
    for i, day in enumerate(days[1:], 1):
        prev = pd.Timestamp(days[i - 1])
        since = (prev + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Kolkata").tz_convert("UTC")
        until = (pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=45)).tz_localize("Asia/Kolkata").tz_convert("UTC")
        if news.empty or news["published"].max() < since:
            continue
        ns = LAB.news_scores(news, ctx.universe, since, until)
        ns = ns[ns["n"] > 0]
        if ns.empty:
            continue
        g = f[f["date"] == day]
        m = g.merge(ns[["symbol", "score"]].rename(columns={"score": "news"}), on="symbol")
        if len(m) < 3:
            continue
        mkt = g["r_close"].median()
        pos, neg = m[m["news"] >= 0.3], m[m["news"] <= -0.3]
        out.append({"date": day, "n": len(m),
                    "ic": m["news"].corr(m["r_close"], method="spearman") if len(m) >= 5 else np.nan,
                    "pos_vs_mkt": (pos["r_close"] - mkt).mean() if len(pos) else np.nan,
                    "neg_vs_mkt": (neg["r_close"] - mkt).mean() if len(neg) else np.nan,
                    "model_agrees": (np.sign(m["news"]) == np.sign(m["score"] - g["score"].median())).mean()})
    o = pd.DataFrame(out)
    if len(o):
        print(f"\nNEWS since the last close ({len(o)} days with news, {o['n'].sum()} stock-days):", flush=True)
        print(f"  IC news vs 9:45->15:15 move: {stats(o['ic'])}", flush=True)
        print(f"  good news vs market: {stats(o['pos_vs_mkt'])}", flush=True)
        print(f"  bad news vs market:  {stats(o['neg_vs_mkt'])}", flush=True)
        print(f"  model already agrees with the news on {o['model_agrees'].mean():.0%} of them", flush=True)
    else:
        print("\nNEWS: no overlap between stored news and intraday days", flush=True)
except Exception as exc:
    print(f"news part failed: {exc}", flush=True)
