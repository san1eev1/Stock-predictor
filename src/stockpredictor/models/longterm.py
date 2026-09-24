"""Long-term ranking model: which Nifty 100 stocks will beat Nifty over ~3 months.

Target: each stock's cross-sectional percentile of its forward 63-trading-day
return in excess of Nifty 50. A LightGBM regressor learns to score stocks so
that higher scores mean better expected relative performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.features import longterm as F

from stockpredictor.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models" / "longterm"
HORIZON = 63            # trading days (~3 months)
EMBARGO_DAYS = 100      # calendar days between train labels and test start
NUM_ROUNDS = 400
PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
    bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
    seed=42, deterministic=True, num_threads=4, verbose=-1,
)


def add_labels(feats: pd.DataFrame, daily: pd.DataFrame, indices: pd.DataFrame,
               horizon: int = HORIZON) -> pd.DataFrame:
    """Attach forward excess return vs Nifty and its per-date percentile rank."""
    px = daily[["symbol", "date", "close"]].sort_values(["symbol", "date"]).copy()
    px["fwd_ret"] = px.groupby("symbol")["close"].shift(-horizon) / px["close"] - 1
    nifty = indices[indices["symbol"] == F.MARKET_INDEX][["date", "close"]].sort_values("date")
    nifty["nifty_fwd"] = nifty["close"].shift(-horizon) / nifty["close"] - 1
    out = feats.merge(px[["symbol", "date", "fwd_ret"]], on=["symbol", "date"], how="left")
    out = out.merge(nifty[["date", "nifty_fwd"]], on="date", how="left")
    out["fwd_excess"] = out["fwd_ret"] - out["nifty_fwd"]
    out["target"] = out.groupby("date")["fwd_excess"].rank(pct=True)
    return out.drop(columns=["nifty_fwd"])


def _fit(train: pd.DataFrame, cols: list[str]) -> "_BoosterWrapper":
    import lightgbm as lgb

    data = lgb.Dataset(train[cols], train["target"], free_raw_data=True)
    return _BoosterWrapper(lgb.train(PARAMS, data, num_boost_round=NUM_ROUNDS))


@dataclass
class LongTermModel:
    model: object
    features: list[str]
    trained_at: str
    train_to: str
    metrics: dict = field(default_factory=dict)

    @classmethod
    def train(cls, labeled_weekly: pd.DataFrame) -> "LongTermModel":
        train = labeled_weekly.dropna(subset=["target"])
        cols = _model_columns(train)
        return cls(model=_fit(train, cols), features=cols,
                   trained_at=datetime.now().isoformat(timespec="seconds"),
                   train_to=f"{train['date'].max():%Y-%m-%d}")

    def score(self, feats: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(feats[self.features]), index=feats.index)

    def explain(self, feats: pd.DataFrame, top: int = 3) -> list[list[str]]:
        """Top positive and negative feature contributions per row, as readable text."""
        contrib = self.model.predict(feats[self.features], pred_contrib=True)[:, :-1]
        reasons = []
        for row in contrib:
            order = np.argsort(row)
            ups = [f"+ {self.features[i]}" for i in order[::-1][:top] if row[i] > 0]
            downs = [f"- {self.features[i]}" for i in order[:top] if row[i] < 0]
            reasons.append(ups + downs)
        return reasons

    def importance(self) -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path / "model.txt"))
        (path / "meta.json").write_text(json.dumps({
            "features": self.features, "trained_at": self.trained_at,
            "train_to": self.train_to, "metrics": self.metrics}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LongTermModel":
        import lightgbm as lgb

        meta = json.loads((path / "meta.json").read_text())
        return cls(model=_BoosterWrapper(lgb.Booster(model_file=str(path / "model.txt"))),
                   features=meta["features"], trained_at=meta["trained_at"],
                   train_to=meta["train_to"], metrics=meta.get("metrics", {}))


class _BoosterWrapper:
    """Small wrapper so trained and loaded models share one interface."""

    def __init__(self, booster):
        self.booster_ = booster

    def predict(self, X, pred_contrib=False):
        return self.booster_.predict(X, pred_contrib=pred_contrib)


def _model_columns(df: pd.DataFrame) -> list[str]:
    skip = F.FEATURE_COLUMNS_EXCLUDE | {"fwd_ret", "fwd_excess", "target"}
    return [c for c in df.columns if c not in skip]


def walk_forward(labeled_weekly: pd.DataFrame, feats_daily: pd.DataFrame,
                 start_year: int, end_year: int | None = None) -> pd.DataFrame:
    """Out-of-sample scores: for each year, train only on data that ended
    before that year (with an embargo so no label overlaps the test period)."""
    end_year = end_year or feats_daily["date"].dt.year.max()
    cols = _model_columns(labeled_weekly)
    out = []
    for year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        train = labeled_weekly[(labeled_weekly["date"] < test_start - pd.Timedelta(days=EMBARGO_DAYS))
                               ].dropna(subset=["target"])
        test = feats_daily[feats_daily["date"].dt.year == year]
        if len(train) < 5000 or test.empty:
            continue
        model = _fit(train, cols)
        out.append(test[["symbol", "date"]].assign(score=model.predict(test[cols])))
    return pd.concat(out, ignore_index=True)


def information_coefficient(scores: pd.DataFrame, labeled: pd.DataFrame) -> pd.Series:
    """Per-date Spearman correlation between scores and realised forward excess return."""
    m = scores.merge(labeled[["symbol", "date", "fwd_excess"]], on=["symbol", "date"]).dropna()
    return m.groupby("date").apply(
        lambda g: g["score"].rank().corr(g["fwd_excess"].rank()) if len(g) >= 10 else np.nan,
        include_groups=False).dropna()
