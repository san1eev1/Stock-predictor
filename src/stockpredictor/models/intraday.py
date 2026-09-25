"""Intraday ranking models: which stocks will do best / worst from 9:45 to 12:30 (traded),
and from 9:45 to the 15:30 close (a second model that keeps learning until the close)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.config import LOCAL_MODELS_DIR
from stockpredictor.data import intraday as I
from stockpredictor.features import intraday as FI
from stockpredictor.models import engine
from stockpredictor.models.engine import Ensemble

MODEL_DIR = LOCAL_MODELS_DIR / "intraday"     # trained on the Mac (Angel One data)


@dataclass(frozen=True)
class Target:
    """What an intraday model predicts: the 9:45 -> exit move."""
    horizon: str          # predictions.horizon in the database
    exit_col: str         # intraday summary column with the exit price
    model_dir: Path
    label: str


TRADE = Target("intraday", I.EXIT_COL, MODEL_DIR, "until 12:30 (traded)")
CLOSE = Target("intraday_close", "close", LOCAL_MODELS_DIR / "intraday_close",
               "until the 15:30 close (learning)")
TARGETS = (TRADE, CLOSE)
# Held while model files are written or read (background training runs in another thread).
MODEL_LOCK = threading.RLock()
MIN_TRAIN_DAYS = 40
NUM_ROUNDS = 300
PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
    bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0,
    seed=7, deterministic=True, num_threads=4, verbose=-1,
)


def current_params(target: Target = TRADE) -> dict:
    """Tuned settings if the tuner saved any, else the defaults."""
    path = target.model_dir / "params.json"
    if path.exists():
        return json.loads(path.read_text())
    return {**PARAMS, "num_rounds": NUM_ROUNDS}


def sample_weights(train: pd.DataFrame, params: dict) -> np.ndarray:
    """Biggest risers AND fallers count more (tail_weight; the buy and sell picks come from
    both ends), and judged paper picks carry their feedback weight (fb_weight)."""
    w = np.ones(len(train))
    tail = params.get("tail_weight") or 0
    if tail > 0:
        w *= 1 + tail * np.abs(train["target"].to_numpy() - 0.5) * 2
    if "fb_weight" in train:
        w *= train["fb_weight"].fillna(1.0).to_numpy()
    return w


def _fit(train: pd.DataFrame, cols: list[str], params: dict | None = None) -> Ensemble:
    params = {"early_stopping": True, "n_seeds": 3, **(params or current_params())}
    weights = sample_weights(train, params)
    params.pop("tail_weight", None)
    return engine.fit(train, cols, params, weights, gap_days=1, default_rounds=NUM_ROUNDS)


@dataclass
class IntradayModel:
    model: object
    features: list[str]
    trained_at: str
    train_to: str
    train_days: int
    metrics: dict = field(default_factory=dict)

    @classmethod
    def train(cls, feats: pd.DataFrame, params: dict | None = None,
              target: Target = TRADE) -> "IntradayModel":
        train = feats.dropna(subset=["target"])
        cols = FI.feature_columns(train)
        return cls(_fit(train, cols, params or current_params(target)), cols, datetime.now().isoformat(timespec="seconds"),
                   f"{train['date'].max():%Y-%m-%d}", int(train["date"].nunique()))

    def score(self, feats: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(feats[self.features].astype(np.float32)),
                         index=feats.index)

    def explain(self, feats: pd.DataFrame, top: int = 3) -> list[list[str]]:
        contrib = self.model.predict(feats[self.features].astype(np.float32),
                                     pred_contrib=True)[:, :-1]
        out = []
        for row in contrib:
            order = np.argsort(row)
            out.append([f"+ {self.features[i]}" for i in order[::-1][:top] if row[i] > 0]
                       + [f"- {self.features[i]}" for i in order[:top] if row[i] < 0])
        return out

    def importance(self) -> pd.Series:
        imp = self.model.feature_importance(self.features)
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save(self, path: Path = MODEL_DIR) -> None:
        with MODEL_LOCK:
            path.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            (path / "meta.json").write_text(json.dumps({
                "features": self.features, "trained_at": self.trained_at,
                "train_to": self.train_to, "train_days": self.train_days,
                "metrics": self.metrics}, indent=2))

    @classmethod
    def load(cls, path: Path = MODEL_DIR) -> "IntradayModel":
        with MODEL_LOCK:
            meta = json.loads((path / "meta.json").read_text())
            return cls(Ensemble.load(path),
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
        model = _fit(train, cols, {**(params or current_params()), "n_seeds": 1})
        out.append(test[["symbol", "date"]].assign(score=model.predict(test[cols].astype(np.float32))))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "date", "score"])
