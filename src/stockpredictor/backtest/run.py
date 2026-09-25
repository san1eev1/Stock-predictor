"""End-to-end long-term backtest: features -> walk-forward scores -> portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor import store
from stockpredictor.backtest import portfolio as P
from stockpredictor.features import longterm as F
from stockpredictor.models import longterm as M


def load_all(store_dir: Path):
    daily, indices = store.load_daily(store_dir), store.load_indices(store_dir)
    universe = store.load_universe(store_dir)
    from stockpredictor.data import delivery as DL

    feats = F.build_features(daily, indices, universe, DL.load(store_dir))
    labeled = M.add_labels(F.weekly_snapshots(feats), daily, indices)
    return daily, indices, feats, labeled


def point_in_time_universe(feats: pd.DataFrame, size: int = 100) -> pd.DataFrame:
    """The `size` most-traded stocks on each date (by 20-day traded value): a stand-in for
    index membership at that time, which reduces survivorship bias in backtests."""
    rank = feats.groupby("date")["turnover_log"].rank(ascending=False, method="first")
    return feats.loc[rank <= size, ["symbol", "date"]]


def run(store_dir: Path, start_year: int = 2015, rules: P.Rules = P.Rules(),
        capital: float = 100_000, rebalances=("daily", "weekly"), cached=None,
        universe_size: int = 100) -> dict:
    daily, indices, feats, labeled, scores = cached or (*load_all(store_dir), None)
    if scores is None:
        scores = M.walk_forward(labeled, feats, start_year)
    # Trade only the most liquid `universe_size` stocks of each date (model still ranks all).
    scores = scores.merge(point_in_time_universe(feats, universe_size), on=["symbol", "date"])
    raw_scores = scores
    scores = M.smooth_scores(scores, feats)          # the steadier score paper trading uses
    close = daily.pivot(index="date", columns="symbol", values="close").sort_index()

    ic = M.information_coefficient(raw_scores, labeled)
    report = {
        "period": f"{scores['date'].min():%Y-%m-%d} to {scores['date'].max():%Y-%m-%d}",
        "rules": rules.__dict__, "capital": capital,
        "ic_mean": ic.mean(), "ic_positive_share": (ic > 0).mean(),
        "ic_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic) / (M.HORIZON / 5)),
        "strategies": {}, "benchmarks": {}, "yearly": {},
    }

    # Top-10 vs bottom-10 average forward excess return (weekly, overlapping).
    m = raw_scores.merge(labeled[["symbol", "date", "fwd_excess"]], on=["symbol", "date"]).dropna()
    m["r"] = m.groupby("date")["score"].rank(ascending=False)
    n = m.groupby("date")["r"].transform("max")
    report["top10_fwd_excess"] = m.loc[m["r"] <= 10, "fwd_excess"].mean()
    report["bottom10_fwd_excess"] = m.loc[m["r"] > n - 10, "fwd_excess"].mean()
    report["hit_rate_top10"] = (m.loc[m["r"] <= 10, "fwd_excess"] > 0).mean()
    report["base_rate_all"] = (m["fwd_excess"] > 0).mean()

    curves = {}
    for reb in rebalances:
        res = P.simulate(scores, close, rules, capital, reb)
        curves[f"model_{reb}"] = res.equity
        report["strategies"][reb] = {**P.performance(res.equity),
                                     **P.trade_stats(res.trades, res.equity),
                                     "total_costs": res.total_costs}
    sectors = dict(zip(*store.load_universe(store_dir)[["symbol", "industry"]].T.values)) \
        if (store_dir / "universe.csv").exists() else {}
    variants = {"weekly, max 2 per sector": P.Rules(**{**rules.__dict__, "max_per_sector": 2}),
                "weekly, volatility sizing": P.Rules(**{**rules.__dict__, "vol_sizing": True}),
                "weekly, both": P.Rules(**{**rules.__dict__, "max_per_sector": 2,
                                           "vol_sizing": True})}
    for name, r in variants.items():
        res = P.simulate(scores, close, r, capital, "weekly", sectors=sectors)
        curves[f"model_{name}"] = res.equity
        report["strategies"][name] = {**P.performance(res.equity),
                                      **P.trade_stats(res.trades, res.equity),
                                      "total_costs": res.total_costs}
    if rules.n_hold != 10:           # same strategy with 10 holdings, for comparison
        wide = P.Rules(**{**rules.__dict__, "n_hold": 10})
        res = P.simulate(scores, close, wide, capital, "weekly")
        curves["model_weekly_10_holdings"] = res.equity
        report["strategies"]["weekly, 10 holdings"] = {
            **P.performance(res.equity), **P.trade_stats(res.trades, res.equity),
            "total_costs": res.total_costs}
    # Baseline without machine learning: plain 12-1 month momentum, same rules.
    base = feats.loc[feats["date"] >= scores["date"].min(), ["symbol", "date", "mom_12_1"]] \
        .merge(point_in_time_universe(feats, universe_size), on=["symbol", "date"])
    res = P.simulate(base.rename(columns={"mom_12_1": "score"}).dropna(), close, rules,
                     capital, "weekly")
    curves["momentum_only"] = res.equity
    report["strategies"]["momentum only (weekly)"] = {
        **P.performance(res.equity), **P.trade_stats(res.trades, res.equity),
        "total_costs": res.total_costs}

    dates = next(iter(curves.values())).index
    nifty = indices[indices["symbol"] == F.MARKET_INDEX].set_index("date")["close"]
    nifty = nifty.reindex(dates).ffill().bfill()   # index can miss a day the stocks traded
    bench = {"nifty50": nifty / nifty.iloc[0] * capital,
             "equal_weight_nifty100": P.equal_weight_benchmark(
                 close.where(_membership(feats, close, universe_size)), dates) * capital}
    for name, curve in bench.items():
        report["benchmarks"][name] = P.performance(curve)
    for name, curve in {**curves, **bench}.items():
        report["yearly"][name] = {str(k): v for k, v in P.yearly_returns(curve).items()}
    report["_curves"] = {k: v for k, v in {**curves, **bench}.items()}
    return report


def _membership(feats, close, size) -> pd.DataFrame:
    """Boolean date x symbol mask of the point-in-time universe."""
    u = point_in_time_universe(feats, size).assign(v=True)
    return u.pivot(index="date", columns="symbol", values="v").reindex(
        index=close.index, columns=close.columns).fillna(False).astype(bool)


def format_report(r: dict) -> str:
    pct = lambda x: f"{x * 100:6.1f}%"  # noqa: E731
    lines = [f"Long-term backtest {r['period']}  (capital Rs {r['capital']:,.0f}, costs included)",
             "",
             f"Prediction quality: IC {r['ic_mean']:.3f} (t={r['ic_t_stat']:.1f}), "
             f"positive in {r['ic_positive_share']:.0%} of weeks",
             f"Top-10 picks beat Nifty over the next week: {r['hit_rate_top10']:.0%} "
             f"(all stocks: {r['base_rate_all']:.0%})",
             f"Avg 1-week excess return: top-10 {pct(r['top10_fwd_excess'])}, "
             f"bottom-10 {pct(r['bottom10_fwd_excess'])}",
             "",
             f"{'':26}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}"]
    rows = {**{(k if k.startswith("momentum") else f"Model ({k})"): v
               for k, v in r["strategies"].items()},
            "Nifty 50": r["benchmarks"]["nifty50"],
            "Equal-weight Nifty 250": r["benchmarks"]["equal_weight_nifty100"]}
    for name, s in rows.items():
        lines.append(f"{name:26}{pct(s['cagr']):>8}{pct(s['volatility']):>8}"
                     f"{s['sharpe']:8.2f}{pct(s['max_drawdown']):>8}")
    lines.append("")
    for k, s in r["strategies"].items():
        lines.append(f"{k}: {s['trades_per_year']:.0f} trades/yr, win rate {s['win_rate']:.0%}, "
                     f"avg hold {s['avg_hold_days']:.0f} days, costs Rs {s['total_costs']:,.0f}, "
                     f"exits {s['exit_reasons']}")
    lines.append("")
    names = list(r["yearly"])
    lines.append("Year  " + "".join(f"{n[:14]:>15}" for n in names))
    for y in r["yearly"][names[0]]:
        lines.append(f"{y}  " + "".join(f"{pct(r['yearly'][n].get(y, np.nan)):>15}" for n in names))
    return "\n".join(lines)


def save(r: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    curves = r.pop("_curves", {})
    path.write_text(json.dumps(r, indent=2, default=float))
    if curves:
        pd.DataFrame(curves).to_csv(path.with_suffix(".curves.csv"), float_format="%.2f")
