"""Shared LightGBM training: early stopping + a small seed ensemble.

1. Early stopping: hold out the most recent 15% of dates (after a gap so no
   label overlaps), grow trees until the held-out error stops improving, and
   remember how many trees were useful.
2. Refit on all data with that number of trees (+10% for the extra data).
3. Train `n_seeds` models with different random seeds and average them:
   steadier predictions for a few seconds of extra training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_ROUNDS = 1500
PATIENCE = 50
HOLDOUT = 0.15
ENGINE_KEYS = ("num_rounds", "early_stopping", "n_seeds", "rank_objective", "top_features")
RANK_BINS = 10          # relevance levels for the ranking objective


class Ensemble:
    """Average of one or more LightGBM boosters, with one interface for all callers."""

    def __init__(self, boosters: list, rounds: int | None = None, used: list[str] | None = None):
        self.boosters = boosters
        self.rounds = rounds
        self.used = used          # feature subset the boosters were trained on (None = all)

    def _select(self, X, all_cols: list[str] | None = None):
        if self.used is None or not hasattr(X, "columns"):
            return X
        return X[self.used]

    def predict(self, X, pred_contrib: bool = False):
        Xs = self._select(X)
        preds = [b.predict(Xs, pred_contrib=pred_contrib) for b in self.boosters]
        out = np.mean(preds, axis=0)
        if pred_contrib and self.used is not None and hasattr(X, "columns"):
            # Expand contributions back to all columns (unused features contribute 0).
            full = np.zeros((out.shape[0], X.shape[1] + 1))
            idx = [list(X.columns).index(c) for c in self.used]
            full[:, idx] = out[:, :-1]
            full[:, -1] = out[:, -1]
            return full
        return out

    def feature_importance(self, all_cols: list[str] | None = None) -> np.ndarray:
        imp = np.sum([b.feature_importance(importance_type="gain") for b in self.boosters], axis=0)
        if self.used is None or all_cols is None:
            return imp
        full = np.zeros(len(all_cols))
        for c, v in zip(self.used, imp):
            full[all_cols.index(c)] = v
        return full

    def save(self, path: Path) -> None:
        import json

        path.mkdir(parents=True, exist_ok=True)
        for old in path.glob("model_*.txt"):
            old.unlink()
        for i, b in enumerate(self.boosters):
            b.save_model(str(path / ("model.txt" if i == 0 else f"model_{i}.txt")))
        (path / "used_features.json").write_text(json.dumps(self.used))

    @classmethod
    def load(cls, path: Path) -> "Ensemble":
        import json

        import lightgbm as lgb

        files = [path / "model.txt", *sorted(path.glob("model_*.txt"))]
        used_file = path / "used_features.json"
        used = json.loads(used_file.read_text()) if used_file.exists() else None
        return cls([lgb.Booster(model_file=str(f)) for f in files], used=used)

    @property
    def booster_(self):   # backwards compatibility
        return self.boosters[0]


def _dataset(lgb, X, y, w, dates, rank: bool, reference=None):
    """Regression dataset, or a ranking dataset with one query group per date."""
    if not rank:
        return lgb.Dataset(X, y, weight=w, reference=reference, free_raw_data=False)
    labels = np.minimum((y * RANK_BINS).astype(int), RANK_BINS - 1)
    _, groups = np.unique(dates, return_counts=True)       # dates are sorted
    # Sample weights (recency / feedback) are not used with the ranking objective.
    return lgb.Dataset(X, labels, group=groups, reference=reference, free_raw_data=False)


def fit(train: pd.DataFrame, cols: list[str], params: dict, weights: np.ndarray | None = None,
        gap_days: int = 0, default_rounds: int = 400) -> Ensemble:
    """Train with early stopping and a seed ensemble.

    rank_objective=True learns the order of stocks within each date directly
    (LambdaRank, optimising the top of the list) instead of regressing a score.
    """
    import lightgbm as lgb

    params = dict(params)
    top_k = int(params.pop("top_features", 0) or 0)
    if top_k and top_k < len(cols):
        cols = select_features(train, cols, params, weights, top_k)
    rounds = int(params.pop("num_rounds", default_rounds))
    early = params.pop("early_stopping", True)
    seeds = int(params.pop("n_seeds", 1))
    rank = bool(params.pop("rank_objective", False))
    params.setdefault("verbose", -1)
    params.setdefault("force_col_wise", True)
    if rank:
        params.update(objective="lambdarank", metric="ndcg", eval_at=[10],
                      label_gain=list(range(RANK_BINS)))
    order = np.argsort(train["date"].to_numpy(), kind="stable")
    train = train.iloc[order]
    X = train[cols].astype(np.float32).to_numpy()
    y = train["target"].to_numpy()
    w = (np.ones(len(train)) if weights is None else np.asarray(weights)[order])
    d = train["date"].to_numpy()

    dates = np.unique(d)
    if early and len(dates) >= 40:
        cut = dates[int(len(dates) * (1 - HOLDOUT))]
        fit_mask = d < cut - np.timedelta64(gap_days, "D")
        val_mask = d >= cut
        if fit_mask.sum() > 1000 and val_mask.sum() > 200:
            dfit = _dataset(lgb, X[fit_mask], y[fit_mask], w[fit_mask], d[fit_mask], rank)
            dval = _dataset(lgb, X[val_mask], y[val_mask], w[val_mask], d[val_mask], rank,
                            reference=dfit)
            probe = lgb.train(params, dfit, num_boost_round=MAX_ROUNDS, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
            rounds = max(PATIENCE, int((probe.best_iteration or rounds) * 1.1))

    data = _dataset(lgb, X, y, w, d, rank)
    base_seed = params.pop("seed", 42)
    boosters = [lgb.train({**params, "seed": base_seed + i}, data, num_boost_round=rounds)
                for i in range(max(1, seeds))]
    return Ensemble(boosters, rounds, cols)


def select_features(train: pd.DataFrame, cols: list[str], params: dict,
                    weights: np.ndarray | None, k: int) -> list[str]:
    """Keep the k most useful features, judged by a quick model on the training data only."""
    import lightgbm as lgb

    p = {k_: v for k_, v in params.items() if k_ not in ENGINE_KEYS}
    p.update(verbose=-1, force_col_wise=True, learning_rate=0.1)
    data = lgb.Dataset(train[cols].astype(np.float32).to_numpy(), train["target"].to_numpy(),
                       weight=weights)
    gain = lgb.train(p, data, num_boost_round=100).feature_importance(importance_type="gain")
    keep = set(np.argsort(gain)[::-1][:k])
    return [c for i, c in enumerate(cols) if i in keep]
