"""Long-term ranking model: which Nifty 250 stocks will beat Nifty over ~3 months.

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
from stockpredictor.features import technical
from stockpredictor.models import engine
from stockpredictor.models.engine import Ensemble

from stockpredictor.config import SHARED_MODELS_DIR

MODEL_DIR = SHARED_MODELS_DIR / "longterm"   # trained on GitHub (see config.py)
HORIZON = 5             # trading days: predict the next week
EMBARGO_DAYS = 14       # calendar days between train labels and test start (> horizon)
MIN_TRAIN_ROWS = 5000
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


def blend(scores: pd.Series, feats: pd.DataFrame, mom_weight: float) -> pd.Series:
    """Mix model scores with plain 12-month momentum, as per-date percentile ranks.
    mom_weight 0 = model only, 1 = momentum only (the self-tuner picks the weight)."""
    if not mom_weight:
        return scores
    date = feats["date"]
    model_rank = scores.groupby(date).rank(pct=True)
    mom_rank = feats["mom_12_1"].groupby(date).rank(pct=True).fillna(0.5)
    return (1 - mom_weight) * model_rank + mom_weight * mom_rank


# Settings that shape the training data rather than LightGBM itself (tuned weekly).
TRAINING_DEFAULTS = {"mom_weight": 0.0, "recency_half_life": 0.0, "tail_weight": 0.0,
                     "early_stopping": True, "n_seeds": 3, "wf_seeds": 1}
SNAPSHOT_STEP = 5       # use every 5th trading day, counted back from the newest day


# How the paper portfolio turns weekly predictions into trades. Trading on the raw weekly
# signal churns too much (costs ate everything in the backtest); averaging each stock's
# score over 20 days and blending in 12-month momentum kept the edge after costs.
TRADE_SMOOTH_DAYS = 20
TRADE_MOM_WEIGHT = 0.5


def trading_scores(model: "LongTermModel", feats: pd.DataFrame, date: pd.Timestamp,
                   symbols, smooth_days: int = TRADE_SMOOTH_DAYS,
                   mom_weight: float = TRADE_MOM_WEIGHT) -> pd.Series:
    """Steadier score used for paper trading: 20-day average of the weekly prediction,
    blended with momentum. Index = symbol."""
    days = sorted(d for d in feats["date"].unique() if d <= date)[-smooth_days:]
    rows = feats[feats["date"].isin(days) & feats["symbol"].isin(symbols)]
    raw = pd.Series(model.model.predict(rows[model.features].astype(np.float32)), index=rows.index)
    avg = raw.groupby(rows["symbol"]).mean()
    today = feats[(feats["date"] == date) & feats["symbol"].isin(avg.index)].set_index("symbol")
    avg = avg.reindex(today.index)
    frame = today.reset_index()[["symbol", "date", "mom_12_1"]]
    blended = blend(pd.Series(avg.values, index=frame.index), frame, mom_weight)
    return pd.Series(blended.values, index=today.index)


def smooth_scores(scores: pd.DataFrame, feats: pd.DataFrame, smooth_days: int = TRADE_SMOOTH_DAYS,
                  mom_weight: float = TRADE_MOM_WEIGHT) -> pd.DataFrame:
    """Backtest version of trading_scores for a whole symbol/date/score table."""
    s = scores.sort_values(["symbol", "date"]).copy()
    s["score"] = s.groupby("symbol")["score"].transform(
        lambda x: x.rolling(smooth_days, min_periods=1).mean())
    s = s.merge(feats[["symbol", "date", "mom_12_1"]], on=["symbol", "date"], how="left")
    s["score"] = blend(s["score"], s, mom_weight).values
    return s[["symbol", "date", "score"]]


def current_params() -> dict:
    """Tuned settings if the weekly tuner saved any, else the defaults."""
    path = MODEL_DIR / "params.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get("horizon") != HORIZON:      # settings tuned for another horizon don't apply
        saved = {}
    return {**TRAINING_DEFAULTS, **PARAMS, "num_rounds": NUM_ROUNDS, **saved, "horizon": HORIZON}


def training_snapshots(feats: pd.DataFrame, step: int = SNAPSHOT_STEP) -> pd.DataFrame:
    """Every `step`-th trading day counted back from the newest one. The grid moves each
    day, so every daily retrain includes the newest day whose 3-month outcome is known."""
    days = np.array(sorted(feats["date"].unique()))
    keep = days[(len(days) - 1 - np.arange(len(days))) % step == 0]
    return feats[feats["date"].isin(keep)].reset_index(drop=True)


def sample_weights(train: pd.DataFrame, params: dict) -> np.ndarray:
    """Recent years count more (half-life in years), the extreme winners/losers count more
    (they decide the picks), and paper-trading feedback rows carry their own weight."""
    w = np.ones(len(train))
    hl = params.get("recency_half_life") or 0
    if hl > 0:
        age = (train["date"].max() - train["date"]).dt.days.to_numpy() / 365.25
        w *= 0.5 ** (age / hl)
    tail = params.get("tail_weight") or 0
    if tail > 0:
        w *= 1 + tail * (train["target"].to_numpy() - 0.5).__abs__() * 2
    if "fb_weight" in train:
        w *= train["fb_weight"].fillna(1.0).to_numpy()
    return w


def _fit(train: pd.DataFrame, cols: list[str], params: dict | None = None) -> Ensemble:
    params = {**TRAINING_DEFAULTS, **(params or current_params())}
    weights = sample_weights(train, params)
    for key in ("mom_weight", "recency_half_life", "tail_weight", "wf_seeds", "horizon",
                "chart_groups"):
        params.pop(key, None)
    return engine.fit(train, cols, params, weights, gap_days=EMBARGO_DAYS,
                      default_rounds=NUM_ROUNDS)


@dataclass
class LongTermModel:
    model: object
    features: list[str]
    trained_at: str
    train_to: str
    metrics: dict = field(default_factory=dict)

    @classmethod
    def train(cls, labeled_weekly: pd.DataFrame, params: dict | None = None) -> "LongTermModel":
        train = labeled_weekly.dropna(subset=["target"])
        params = params or current_params()
        cols = _model_columns(train, params)
        return cls(model=_fit(train, cols, params), features=cols,
                   trained_at=datetime.now().isoformat(timespec="seconds"),
                   train_to=f"{train['date'].max():%Y-%m-%d}",
                   metrics={"mom_weight": params.get("mom_weight", 0.0)})

    def score(self, feats: pd.DataFrame) -> pd.Series:
        raw = pd.Series(self.model.predict(feats[self.features].astype(np.float32)),
                        index=feats.index)
        return blend(raw, feats, self.metrics.get("mom_weight", 0.0))

    def explain(self, feats: pd.DataFrame, top: int = 3) -> list[list[str]]:
        """Top positive and negative feature contributions per row, as readable text."""
        contrib = self.model.predict(feats[self.features].astype(np.float32),
                                     pred_contrib=True)[:, :-1]
        reasons = []
        for row in contrib:
            order = np.argsort(row)
            ups = [f"+ {self.features[i]}" for i in order[::-1][:top] if row[i] > 0]
            downs = [f"- {self.features[i]}" for i in order[:top] if row[i] < 0]
            reasons.append(ups + downs)
        return reasons

    def importance(self) -> pd.Series:
        imp = self.model.feature_importance(self.features)
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        (path / "meta.json").write_text(json.dumps({
            "features": self.features, "trained_at": self.trained_at,
            "train_to": self.train_to, "metrics": self.metrics, "horizon": HORIZON}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LongTermModel":
        meta = json.loads((path / "meta.json").read_text())
        return cls(model=Ensemble.load(path),
                   features=meta["features"], trained_at=meta["trained_at"],
                   train_to=meta["train_to"], metrics=meta.get("metrics", {}))


def _model_columns(df: pd.DataFrame, params: dict | None = None) -> list[str]:
    """Model inputs; chart-signal groups not switched on (params['chart_groups']) are left out."""
    skip = F.FEATURE_COLUMNS_EXCLUDE | {"fwd_ret", "fwd_excess", "target", "fb_weight"} \
        | technical.excluded_columns((params or current_params()).get("chart_groups"))
    return [c for c in df.columns if c not in skip]


def walk_forward(labeled_weekly: pd.DataFrame, feats_daily: pd.DataFrame,
                 start_year: int, end_year: int | None = None,
                 params: dict | None = None) -> pd.DataFrame:
    """Out-of-sample scores: for each year, train only on data that ended
    before that year (with an embargo so no label overlaps the test period)."""
    end_year = end_year or feats_daily["date"].dt.year.max()
    cols = _model_columns(labeled_weekly, params)
    out = []
    for year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        train = labeled_weekly[(labeled_weekly["date"] < test_start - pd.Timedelta(days=EMBARGO_DAYS))
                               ].dropna(subset=["target"])
        test = feats_daily[feats_daily["date"].dt.year == year]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        p = params or current_params()
        model = _fit(train, cols, {**p, "n_seeds": p.get("wf_seeds", 1)})
        raw = pd.Series(model.predict(test[cols].astype(np.float32)), index=test.index)
        score = blend(raw, test, (params or current_params()).get("mom_weight", 0.0))
        out.append(test[["symbol", "date"]].assign(score=score.values))
    if not out:
        return pd.DataFrame(columns=["symbol", "date", "score"])
    return pd.concat(out, ignore_index=True)


def information_coefficient(scores: pd.DataFrame, labeled: pd.DataFrame) -> pd.Series:
    """Per-date Spearman correlation between scores and realised forward excess return."""
    m = scores.merge(labeled[["symbol", "date", "fwd_excess"]], on=["symbol", "date"]).dropna()
    return m.groupby("date").apply(
        lambda g: g["score"].rank().corr(g["fwd_excess"].rank()) if len(g) >= 10 else np.nan,
        include_groups=False).dropna()
