"""How much does survivorship bias flatter the long-term backtest? (GitHub, research.yml)

Survivor history (what we use today): the CURRENT Nifty 250 members, back to 2005.
Point-in-time history: every NSE stock of each day (bhavcopy, incl. companies that later
fell out of the index or were delisted); each day's universe = the 250 stocks with the
highest 60-day median traded value AT THAT TIME.

Same pipeline for both (features, walk-forward model since 2015, live trading rules:
5 holdings, weekly, costs, 'half' regime filter). The difference is the bias.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def backtest(daily: pd.DataFrame, universe: pd.DataFrame | None, label: str) -> None:
    feats = F.build_features(daily, indices, universe)
    labeled = M.add_labels(F.weekly_snapshots(feats), daily, indices)
    raw = M.walk_forward(labeled, feats, 2015)
    raw = raw.merge(bt.point_in_time_universe(feats, 100), on=["symbol", "date"])
    scores = M.smooth_scores(raw, feats)
    close = daily.pivot(index="date", columns="symbol", values="close").sort_index()
    eq = P.simulate(scores, close, P.Rules(), 100_000, "weekly", risk_off=risk_off).equity
    out = []
    for name, e in (("2015-2026", eq), ("last 5 yrs", eq[eq.index >= eq.index.max() - pd.DateOffset(years=5)])):
        p = P.performance(e / e.iloc[0] * 100_000)
        out.append(f"{name}: {p['cagr']:6.1%}/yr Sharpe {p['sharpe']:.2f} worst {p['max_drawdown']:6.1%}")
    ic = M.information_coefficient(raw, labeled)
    print(f"{label:44} | IC {ic.mean():.3f} | " + " | ".join(out), flush=True)


survivors = store.load_daily(d)
print(f"survivor history: {survivors['symbol'].nunique()} stocks", flush=True)
backtest(survivors, store.load_universe(d), "survivors only (today's Nifty 250)")

b = bhav.load(d)
if b.empty:
    sys.exit("no bhavcopy history yet (bhav-update runs in the 21:00 data workflow)")
b = b.dropna(subset=["close"]).assign(adj_close=lambda x: x["close"], source="bhav")
b["value_med60"] = b.sort_values("date").groupby("symbol")["value"].transform(
    lambda x: x.rolling(60, min_periods=20).median())
rank = b.groupby("date")["value_med60"].rank(ascending=False, method="first")
keep = set(b.loc[rank <= 250, "symbol"])                    # ever in the top 250
pit, _ = clean_daily(b[b["symbol"].isin(keep)][["symbol", "date", "open", "high", "low",
                                                  "close", "adj_close", "volume", "source"]])
gone = keep - set(survivors["symbol"])
print(f"point-in-time history: {len(keep)} stocks were in a day's top 250 since "
      f"{b['date'].min():%Y}; {len(gone)} of them are not in today's Nifty 250 "
      "(left the index, renamed, merged or delisted)", flush=True)
backtest(pit, None, "point-in-time (incl. stocks that left)")
