"""'Take this trade?' filter for intraday picks (meta-labelling).

The ranking model says WHICH stocks look best; this second model estimates, for each of
the day's buy and sell candidates, the probability that the trade actually makes money
after costs (stop-loss / target / square-off). Only candidates above a probability
threshold are traded (rules.min_prob, chosen by profit in trainer.tune_rules), so trades
the costs would eat are skipped.

It learns from out-of-sample picks only: the ranking scores come from walk-forward models
(each day scored by a model trained on earlier days).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import intraday as B
from stockpredictor.costs import DEFAULT_INTRADAY_COSTS
from stockpredictor.data import intraday as I
from stockpredictor.features import intraday as FI

FILE = "take_model.txt"
N_CANDIDATES = 10
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 7,
          "min_data_in_leaf": 80, "feature_fraction": 0.7, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 7,
          "deterministic": True, "num_threads": 4}
ROUNDS = 200


def candidates(scores: pd.DataFrame, feats: pd.DataFrame, n: int = N_CANDIDATES) -> pd.DataFrame:
    """Each day's top-n buy and bottom-n sell candidates with their features, the side,
    the rank within the side and the score's percentile."""
    s = scores.copy()
    s["score_pct"] = s.groupby("date")["score"].rank(pct=True)
    s["rank_long"] = s.groupby("date")["score"].rank(ascending=False, method="first")
    s["rank_short"] = s.groupby("date")["score"].rank(ascending=True, method="first")
    longs = s[s["rank_long"] <= n].assign(side="long", side_rank=lambda d: d["rank_long"])
    shorts = s[s["rank_short"] <= n].assign(side="short", side_rank=lambda d: d["rank_short"])
    c = pd.concat([longs, shorts], ignore_index=True).drop(columns=["rank_long", "rank_short"])
    c = c.merge(feats, on=["symbol", "date"], how="left", suffixes=("", "_f"))
    c["is_long"] = (c["side"] == "long").astype(float)
    # the candidate's pull in its own direction (long: high score good; short: low)
    c["side_pct"] = np.where(c["is_long"] == 1, c["score_pct"], 1 - c["score_pct"])
    return c


def outcomes(c: pd.DataFrame, rules, exit_col: str, exit_minutes: int,
             capital: float = 100_000) -> pd.Series:
    """1 if the candidate's trade would have made money after costs, else 0."""
    slot = capital / max(1, rules.n_long + rules.n_short)
    long_x = B.trade_exits(c, "long", rules.stop_loss, rules.target, exit_col, exit_minutes)
    short_x = B.trade_exits(c, "short", rules.stop_loss, rules.target, exit_col, exit_minutes)
    ret = np.where(c["is_long"] == 1, long_x / c["c30"] - 1, 1 - short_x / c["c30"])
    cost = (DEFAULT_INTRADAY_COSTS.cost("buy", slot)
            + DEFAULT_INTRADAY_COSTS.cost("sell", slot)) / slot  # round trip, as a share
    y = pd.Series((ret - cost > 0).astype(float), index=c.index)
    return y.where(~np.isnan(ret))


def _columns(c: pd.DataFrame) -> list[str]:
    base = [x for x in FI.feature_columns(c) if x in c and x not in ("side",)]
    return sorted(set(base) | {"score_pct", "side_rank", "is_long", "side_pct"})


def fit(c: pd.DataFrame, y: pd.Series):
    import lightgbm as lgb

    ok = y.notna()
    cols = _columns(c)
    model = lgb.train(PARAMS, lgb.Dataset(c.loc[ok, cols].astype(np.float32), y[ok]), ROUNDS)
    return model, cols


def predict(model, cols: list[str], c: pd.DataFrame) -> np.ndarray:
    return model.predict(c.reindex(columns=cols).astype(np.float32))


def cross_fit(c: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Out-of-sample probabilities for every candidate: the first half of the days is
    predicted by a model fit on the second half and vice versa."""
    days = sorted(c["date"].unique())
    first = c["date"].isin(days[: len(days) // 2])
    prob = pd.Series(np.nan, index=c.index)
    for train, test in ((~first, first), (first, ~first)):
        if y[train].notna().sum() < 200 or not test.any():
            continue
        m, cols = fit(c[train], y[train])
        prob[test] = predict(m, cols, c[test])
    return prob


def with_probs(scores: pd.DataFrame, c: pd.DataFrame, prob: pd.Series) -> pd.DataFrame:
    """scores + prob_long / prob_short columns (NaN for stocks that were not candidates)."""
    p = c.assign(prob=prob.values)[["symbol", "date", "side", "prob"]]
    wide = p.pivot_table(index=["symbol", "date"], columns="side", values="prob").reset_index()
    wide = wide.rename(columns={"long": "prob_long", "short": "prob_short"})
    for col in ("prob_long", "prob_short"):
        if col not in wide:
            wide[col] = np.nan
    return scores.merge(wide[["symbol", "date", "prob_long", "prob_short"]],
                        on=["symbol", "date"], how="left")


def save(model, cols: list[str], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(str(folder / FILE))
    (folder / "take_model.json").write_text(json.dumps({"features": cols}))


def load(folder: Path):
    import lightgbm as lgb

    if not (folder / FILE).exists():
        return None
    cols = json.loads((folder / "take_model.json").read_text())["features"]
    return lgb.Booster(model_file=str(folder / FILE)), cols


def exit_minutes(exit_col: str) -> int:
    return I.EXIT_MINUTES if exit_col == I.EXIT_COL else I.WINDOW_MINUTES
