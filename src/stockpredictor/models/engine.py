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
ENGINE_KEYS = ("num_rounds", "early_stopping", "n_seeds")


class Ensemble:
    """Average of one or more LightGBM boosters, with one interface for all callers."""

    def __init__(self, boosters: list, rounds: int | None = None):
        self.boosters = boosters
        self.rounds = rounds

    def predict(self, X, pred_contrib: bool = False):
        preds = [b.predict(X, pred_contrib=pred_contrib) for b in self.boosters]
        return np.mean(preds, axis=0)

    def feature_importance(self) -> np.ndarray:
        return np.sum([b.feature_importance(importance_type="gain") for b in self.boosters], axis=0)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for old in path.glob("model_*.txt"):
            old.unlink()
        for i, b in enumerate(self.boosters):
            b.save_model(str(path / ("model.txt" if i == 0 else f"model_{i}.txt")))

    @classmethod
    def load(cls, path: Path) -> "Ensemble":
        import lightgbm as lgb

        files = [path / "model.txt", *sorted(path.glob("model_*.txt"))]
        return cls([lgb.Booster(model_file=str(f)) for f in files])

    @property
    def booster_(self):   # backwards compatibility
        return self.boosters[0]


def fit(train: pd.DataFrame, cols: list[str], params: dict, weights: np.ndarray | None = None,
        gap_days: int = 0, default_rounds: int = 400) -> Ensemble:
    import lightgbm as lgb

    params = dict(params)
    rounds = int(params.pop("num_rounds", default_rounds))
    early = params.pop("early_stopping", True)
    seeds = int(params.pop("n_seeds", 1))
    params.setdefault("verbose", -1)
    params.setdefault("force_col_wise", True)
    X = train[cols].astype(np.float32)
    y = train["target"].to_numpy()
    w = np.ones(len(train)) if weights is None else weights

    dates = np.sort(train["date"].unique())
    if early and len(dates) >= 40:
        cut = dates[int(len(dates) * (1 - HOLDOUT))]
        fit_mask = (train["date"] < cut - pd.Timedelta(days=gap_days)).to_numpy()
        val_mask = (train["date"] >= cut).to_numpy()
        if fit_mask.sum() > 1000 and val_mask.sum() > 200:
            dfit = lgb.Dataset(X[fit_mask], y[fit_mask], weight=w[fit_mask])
            dval = lgb.Dataset(X[val_mask], y[val_mask], weight=w[val_mask], reference=dfit)
            probe = lgb.train(params, dfit, num_boost_round=MAX_ROUNDS, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
            rounds = max(PATIENCE, int((probe.best_iteration or rounds) * 1.1))

    data = lgb.Dataset(X, y, weight=w, free_raw_data=False)
    base_seed = params.pop("seed", 42)
    boosters = [lgb.train({**params, "seed": base_seed + i}, data, num_boost_round=rounds)
                for i in range(max(1, seeds))]
    return Ensemble(boosters, rounds)
