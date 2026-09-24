"""Intraday ranking model: which stocks will do best / worst from 9:45 to 15:15."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.config import PROJECT_ROOT
from stockpredictor.features import intraday as FI
from stockpredictor.models.longterm import _BoosterWrapper

MODEL_DIR = PROJECT_ROOT / "models" / "intraday"
MIN_TRAIN_DAYS = 40
NUM_ROUNDS = 300
PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
    bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0,
    seed=7, deterministic=True, num_threads=4, verbose=-1,
)


def current_params() -> dict:
    """Tuned settings if the weekly tuner saved any, else the defaults."""
    path = MODEL_DIR / "params.json"
    if path.exists():
        return json.loads(path.read_text())
    return {**PARAMS, "num_rounds": NUM_ROUNDS}


def _fit(train: pd.DataFrame, cols: list[str], params: dict | None = None) -> _BoosterWrapper:
    import lightgbm as lgb

    data = lgb.Dataset(train[cols], train["target"], free_raw_data=True)
    params = dict(params or current_params())
    rounds = params.pop("num_rounds", NUM_ROUNDS)
    return _BoosterWrapper(lgb.train(params, data, num_boost_round=rounds))


@dataclass
class IntradayModel:
    model: object
    features: list[str]
    trained_at: str
    train_to: str
    train_days: int
    metrics: dict = field(default_factory=dict)

    @classmethod
    def train(cls, feats: pd.DataFrame, params: dict | None = None) -> "IntradayModel":
        train = feats.dropna(subset=["target"])
        cols = FI.feature_columns(train)
        return cls(_fit(train, cols, params), cols, datetime.now().isoformat(timespec="seconds"),
                   f"{train['date'].max():%Y-%m-%d}", int(train["date"].nunique()))

    def score(self, feats: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(feats[self.features]), index=feats.index)

    def explain(self, feats: pd.DataFrame, top: int = 3) -> list[list[str]]:
        contrib = self.model.predict(feats[self.features], pred_contrib=True)[:, :-1]
        out = []
        for row in contrib:
            order = np.argsort(row)
            out.append([f"+ {self.features[i]}" for i in order[::-1][:top] if row[i] > 0]
                       + [f"- {self.features[i]}" for i in order[:top] if row[i] < 0])
        return out

    def importance(self) -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save(self, path: Path = MODEL_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path / "model.txt"))
        (path / "meta.json").write_text(json.dumps({
            "features": self.features, "trained_at": self.trained_at, "train_to": self.train_to,
            "train_days": self.train_days, "metrics": self.metrics}, indent=2))

    @classmethod
    def load(cls, path: Path = MODEL_DIR) -> "IntradayModel":
        import lightgbm as lgb

        meta = json.loads((path / "meta.json").read_text())
        return cls(_BoosterWrapper(lgb.Booster(model_file=str(path / "model.txt"))),
                   meta["features"], meta["trained_at"], meta["train_to"],
                   meta.get("train_days", 0), meta.get("metrics", {}))


def walk_forward(feats: pd.DataFrame, min_train_days: int = MIN_TRAIN_DAYS,
                 params: dict | None = None, last_days: int | None = None) -> pd.DataFrame:
    """Out-of-sample scores: each month is scored by a model trained only on earlier days."""
    feats = feats.dropna(subset=["target"])
    days = sorted(feats["date"].unique())
    if len(days) <= min_train_days:
        return pd.DataFrame(columns=["symbol", "date", "score"])
    cols = FI.feature_columns(feats)
    test_days = days[min_train_days:] if last_days is None else days[max(min_train_days, len(days) - last_days):]
    months = pd.Series(test_days).dt.to_period("M").unique()
    out = []
    for m in months:
        start = m.start_time
        train = feats[feats["date"] < start]
        test = feats[(feats["date"].dt.to_period("M") == m) & (feats["date"] >= test_days[0])]
        if train["date"].nunique() < min_train_days or test.empty:
            continue
        model = _fit(train, cols, params)
        out.append(test[["symbol", "date"]].assign(score=model.predict(test[cols])))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "date", "score"])
