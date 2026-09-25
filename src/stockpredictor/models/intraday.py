"""Intraday ranking models: which stocks will do best / worst from 9:45 to 12:30, and from
9:45 until the close (15:15 square-off). Both are paper-traded on their own Rs 1 lakh a day,
compete after every close, and can learn from each other: each model's tuning may blend in
the other model's ranking (peer_weight) when that is more accurate out-of-sample."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor import config
from stockpredictor.config import LOCAL_MODELS_DIR, SHARED_MODELS_DIR
from stockpredictor.data import intraday as I
from stockpredictor.features import intraday as FI
from stockpredictor.models import engine
from stockpredictor.models.engine import Ensemble

# Trained on GitHub (published with the other models) unless the Mac trains itself.
ROOT = LOCAL_MODELS_DIR if config.MAC_TRAINING else SHARED_MODELS_DIR
MODEL_DIR = ROOT / "intraday"


@dataclass(frozen=True)
class Target:
    """What an intraday model predicts: the 9:45 -> exit move."""
    horizon: str          # predictions.horizon in the database
    exit_col: str         # intraday summary column with the exit price
    model_dir: Path
    label: str


TRADE = Target("intraday", I.EXIT_COL, MODEL_DIR, "until 12:30")
CLOSE = Target("intraday_close", "px_1515", ROOT / "intraday_close",
               "until the close (15:15 square-off)")
TARGETS = (TRADE, CLOSE)


def model_path(target: Target) -> Path | None:
    """Where this target's trained model is: the published one, else a model the Mac trained
    earlier (used until the first one arrives from GitHub)."""
    for d in (target.model_dir, LOCAL_MODELS_DIR / target.model_dir.name):
        if (d / "model.txt").exists():
            return d
    return None


def peer_of(target: Target) -> Target:
    return CLOSE if target == TRADE else TRADE


class Blended:
    """A model's scores blended with its peer's ranking (learning from each other).
    Used exactly like an IntradayModel for picking; reasons come from the model itself."""

    def __init__(self, own: "IntradayModel", peer: "IntradayModel | None", weight: float):
        self.own, self.peer, self.weight = own, peer, weight
        self.train_to = own.train_to

    def score(self, feats: pd.DataFrame) -> pd.Series:
        s = self.own.score(feats)
        if self.peer is None or not self.weight:
            return s
        return blend(s, self.peer.score(feats), feats["date"], self.weight)

    def explain(self, feats: pd.DataFrame, top: int = 3) -> list[list[str]]:
        return self.own.explain(feats, top)


def blend(own: pd.Series, peer: pd.Series, dates: pd.Series, weight: float) -> pd.Series:
    """Per day: (1 - weight) x own rank + weight x peer rank (both 0..1)."""
    r_own = own.groupby(dates.values).rank(pct=True)
    r_peer = peer.groupby(dates.values).rank(pct=True)
    return (1 - weight) * r_own + weight * r_peer
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
    params.pop("peer_weight", None)            # used when scoring (Blended), not by LightGBM
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
