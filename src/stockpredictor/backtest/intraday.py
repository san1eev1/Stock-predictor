"""Intraday rules, trade replay and backtest.

Each trading day at 9:45: go long the top-ranked stocks and short the
bottom-ranked ones, equal money each, with a stop-loss and target in % of the
9:45 price; anything still open is squared off at 15:15. Stop/target hits are
replayed from the first-hit minutes in the intraday summaries (5-minute
resolution; if both happen in the same bar, the stop-loss is assumed first).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.costs import DEFAULT_INTRADAY_COSTS, IntradayCosts
from stockpredictor.data import intraday as I


@dataclass(frozen=True)
class IntradayRules:
    n_long: int = 10
    n_short: int = 10
    stop_loss: float = 1.0      # % from the 9:45 price (one of intraday.LEVELS)
    target: float = 2.0         # % (one of LEVELS), 0 = no target
    skip_quantile: float = 0.0  # skip days whose signal strength is below this
                                # quantile of the previous 60 days (0 = never skip)


def snap(pct: float) -> float:
    """Nearest stored level, so any setting can be replayed from the summaries."""
    return min(I.LEVELS, key=lambda lv: abs(lv - pct)) if pct else 0.0


def pick(scores: pd.DataFrame, rules: IntradayRules) -> pd.DataFrame:
    """scores: one day's symbol/score rows -> picks with side and rank."""
    s = scores.sort_values("score", ascending=False).reset_index(drop=True)
    longs = s.head(rules.n_long).assign(side="long")
    shorts = s.tail(rules.n_short).iloc[::-1].assign(side="short")
    out = pd.concat([longs, shorts], ignore_index=True)
    out["rank"] = list(range(1, len(longs) + 1)) + list(range(1, len(shorts) + 1))
    return out


def replay(row, side: str, rules: IntradayRules) -> tuple[float, str]:
    """Exit price and reason for one trade, from its intraday summary row."""
    c30, sl, tp = row["c30"], snap(rules.stop_loss), snap(rules.target)
    adverse, favourable = ("d", "u") if side == "long" else ("u", "d")
    t_sl = row[I.level_col(adverse, sl)] if sl else np.nan
    t_tp = row[I.level_col(favourable, tp)] if tp else np.nan
    sign = 1 if side == "long" else -1
    if not np.isnan(t_sl) and (np.isnan(t_tp) or t_sl <= t_tp):
        return c30 * (1 - sign * sl / 100), "stop-loss"
    if not np.isnan(t_tp):
        return c30 * (1 + sign * tp / 100), "target"
    return row["px_1515"], "15:15 square-off"


def trade_pnl(side: str, qty: int, entry: float, exit_: float,
              costs: IntradayCosts = DEFAULT_INTRADAY_COSTS) -> tuple[float, float]:
    """(P&L after costs, costs) for a round trip."""
    buy_px, sell_px = (entry, exit_) if side == "long" else (exit_, entry)
    c = costs.cost("buy", qty * buy_px) + costs.cost("sell", qty * sell_px)
    gross = (exit_ - entry) * qty * (1 if side == "long" else -1)
    return gross - c, c


def signal_strength(day_scores: pd.Series, rules: IntradayRules) -> float:
    s = day_scores.sort_values()
    return s.tail(rules.n_long).mean() - s.head(rules.n_short).mean()


def simulate(scores: pd.DataFrame, summ: pd.DataFrame, rules: IntradayRules = IntradayRules(),
             capital: float = 100_000, costs: IntradayCosts = DEFAULT_INTRADAY_COSTS) -> dict:
    data = scores.merge(summ, on=["symbol", "date"])
    slot = capital / (rules.n_long + rules.n_short)
    strength = data.groupby("date")["score"].apply(lambda s: signal_strength(s, rules))
    threshold = strength.shift(1).rolling(60, min_periods=20).quantile(rules.skip_quantile) \
        if rules.skip_quantile > 0 else pd.Series(-np.inf, index=strength.index)

    trades, days = [], []
    for d, day in data.groupby("date"):
        up_share = (day["px_1515"] > day["c30"]).mean()
        if strength[d] < threshold.get(d, -np.inf):
            days.append({"date": d, "pnl": 0.0, "skipped": True})
            continue
        pnl_day = 0.0
        for _, p in pick(day, rules).iterrows():
            qty = int(slot / p["c30"])
            if qty <= 0:
                continue
            exit_, reason = replay(p, p["side"], rules)
            pnl, c = trade_pnl(p["side"], qty, p["c30"], exit_, costs)
            pnl_day += pnl
            correct = (p["px_1515"] > p["c30"]) if p["side"] == "long" else (p["px_1515"] < p["c30"])
            trades.append({"date": d, "symbol": p["symbol"], "side": p["side"], "qty": qty,
                           "entry": p["c30"], "exit": exit_, "reason": reason, "pnl": pnl,
                           "cost": c, "correct": correct,
                           "baseline": up_share if p["side"] == "long" else 1 - up_share})
        days.append({"date": d, "pnl": pnl_day, "skipped": False})
    return {"trades": pd.DataFrame(trades), "days": pd.DataFrame(days)}


def metrics(sim: dict, capital: float = 100_000) -> dict:
    t, d = sim["trades"], sim["days"]
    if t.empty:
        return {"trading_days": 0}
    traded = d[~d["skipped"]]
    daily_ret = traded["pnl"] / capital
    equity = capital + d["pnl"].cumsum()
    return {
        "trading_days": len(traded), "skipped_days": int(d["skipped"].sum()),
        "total_pnl": float(d["pnl"].sum()), "return_pct": float(d["pnl"].sum() / capital),
        "avg_day_pnl": float(traded["pnl"].mean()),
        "win_days": float((traded["pnl"] > 0).mean()),
        "sharpe": float(daily_ret.mean() / daily_ret.std() * math.sqrt(252)) if daily_ret.std() else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "trade_win_rate": float((t["pnl"] > 0).mean()),
        "direction_accuracy": float(t["correct"].mean()),
        "direction_accuracy_long": float(t.loc[t["side"] == "long", "correct"].mean()),
        "direction_accuracy_short": float(t.loc[t["side"] == "short", "correct"].mean()),
        "random_baseline": float(t["baseline"].mean()),
        "total_costs": float(t["cost"].sum()),
        "exits": t["reason"].value_counts().to_dict(),
    }


def run(feats: pd.DataFrame, scores: pd.DataFrame, rules: IntradayRules = IntradayRules(),
        capital: float = 100_000) -> dict:
    """Model vs simple no-ML baselines, plus a small stop/target grid."""
    summ = feats[["symbol", "date", "c30", "px_1515", *I.LEVEL_COLS]]
    out = {"period": f"{scores['date'].min():%Y-%m-%d} to {scores['date'].max():%Y-%m-%d}",
           "rules": asdict(rules), "capital": capital, "strategies": {}, "grid": []}
    base = feats[feats["date"].isin(scores["date"].unique())]
    candidates = {
        "model": scores,
        "momentum (first 30 min)": base[["symbol", "date"]].assign(score=base["rel_r30"]),
        "reversal (first 30 min)": base[["symbol", "date"]].assign(score=-base["rel_r30"]),
    }
    for name, sc in candidates.items():
        out["strategies"][name] = metrics(simulate(sc, summ, rules, capital), capital)
    out["strategies"]["model, skip weak days"] = metrics(simulate(
        scores, summ, IntradayRules(**{**asdict(rules), "skip_quantile": 0.5}), capital), capital)
    # Fewer, larger positions: brokerage is capped per order, so costs fall as a share.
    for n in (3, 2, 1):
        r = IntradayRules(**{**asdict(rules), "n_long": n, "n_short": n})
        out["strategies"][f"model, {n} long + {n} short"] = metrics(
            simulate(scores, summ, r, capital), capital)
    for sl in (0.75, 1.0, 1.5):
        for tp in (0.0, 1.5, 2.0, 3.0):
            r = IntradayRules(**{**asdict(rules), "stop_loss": sl, "target": tp})
            m = metrics(simulate(scores, summ, r, capital), capital)
            out["grid"].append({"stop_loss": sl, "target": tp, "return_pct": m.get("return_pct"),
                                "sharpe": m.get("sharpe")})
    ic = scores.merge(feats[["symbol", "date", "target_ret"]], on=["symbol", "date"]) \
        .groupby("date").apply(lambda g: g["score"].rank().corr(g["target_ret"].rank()),
                               include_groups=False)
    out["ic_mean"], out["ic_positive_share"] = float(ic.mean()), float((ic > 0).mean())
    return out


def format_report(r: dict) -> str:
    pct = lambda x: f"{x * 100:6.1f}%"  # noqa: E731
    lines = [f"Intraday backtest {r['period']} (capital Rs {r['capital']:,.0f}, costs included)",
             f"Prediction quality: IC {r['ic_mean']:.3f}, positive on {r['ic_positive_share']:.0%} of days",
             "", f"{'':28}{'Days':>6}{'Return':>9}{'Sharpe':>8}{'Win days':>10}{'Dir. acc':>10}{'Random':>8}"]
    for name, m in r["strategies"].items():
        if not m.get("trading_days"):
            continue
        lines.append(f"{name:28}{m['trading_days']:6d}{pct(m['return_pct']):>9}{m['sharpe']:8.2f}"
                     f"{pct(m['win_days']):>10}{pct(m['direction_accuracy']):>10}"
                     f"{pct(m['random_baseline']):>8}")
    m = r["strategies"]["model"]
    if m.get("trading_days"):
        lines += ["", f"Model: costs Rs {m['total_costs']:,.0f}, max drawdown {pct(m['max_drawdown'])}, "
                      f"exits {m['exits']}",
                  "", "Stop-loss / target grid (model):  SL%  TP%  return  sharpe"]
        for g in r["grid"]:
            if g["return_pct"] is not None:
                lines.append(f"{'':34}{g['stop_loss']:4}{g['target']:5}{pct(g['return_pct']):>8}"
                             f"{g['sharpe']:8.2f}")
    return "\n".join(lines)


def save(r: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=2, default=float))
