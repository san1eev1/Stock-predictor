import numpy as np
import pandas as pd
import pytest

from stockpredictor.backtest import intraday as B
from stockpredictor.costs import IntradayCosts
from stockpredictor.data import intraday as I
from stockpredictor.features import intraday as FI
from stockpredictor.features import longterm as F
from stockpredictor.models import intraday as MI
from test_features_intraday import fake_summaries
from test_features_longterm import synthetic


def row(**hits):
    r = {"c30": 100.0, "px_1230": 100.5, **{c: np.nan for c in I.LEVEL_COLS}}
    r.update(hits)
    return pd.Series(r)


def test_replay_stop_target_and_squareoff():
    rules = B.IntradayRules(stop_loss=1.0, target=2.0)
    assert B.replay(row(), "long", rules) == (100.5, "12:30 square-off")
    assert B.replay(row(d100=30, u200=60), "long", rules) == (99.0, "stop-loss")
    assert B.replay(row(d100=90, u200=60), "long", rules) == (102.0, "target")
    assert B.replay(row(d100=30, u200=30), "long", rules)[1] == "stop-loss"   # same bar: stop first
    # Short: adverse move is up.
    assert B.replay(row(u100=10), "short", rules) == (101.0, "stop-loss")
    assert B.replay(row(d200=10), "short", rules) == (98.0, "target")
    assert B.replay(row(u200=5), "long", B.IntradayRules(target=0))[1] == "12:30 square-off"
    # A stop-loss hit after the 12:30 square-off (minute 165) doesn't count.
    assert B.replay(row(d100=200), "long", rules) == (100.5, "12:30 square-off")
    assert B.replay(row(d100=165), "long", rules)[1] == "stop-loss"


def test_trade_pnl_short_and_costs():
    pnl, c = B.trade_pnl("short", 100, 100.0, 98.0, IntradayCosts())
    assert c > 0 and pnl == pytest.approx(200 - c)
    assert B.snap(1.1) == 1.0 and B.snap(0) == 0.0


def test_pick_sides():
    s = pd.DataFrame({"symbol": list("ABCDEF"), "score": [6, 5, 4, 3, 2, 1]})
    p = B.pick(s, B.IntradayRules(n_long=2, n_short=2))
    assert p[p["side"] == "long"]["symbol"].tolist() == ["A", "B"]
    assert p[p["side"] == "short"]["symbol"].tolist() == ["F", "E"]


@pytest.fixture(scope="module")
def feats():
    daily, indices, universe = synthetic(n_days=500, symbols=[f"S{i}" for i in range(15)])
    lt = F.build_features(daily, indices, universe)
    return FI.build(fake_summaries(daily, n_days=150), daily, lt)


def test_walk_forward_and_backtest(feats):
    scores = MI.walk_forward(feats, min_train_days=40)
    assert not scores.empty
    first_test = scores["date"].min()
    assert first_test > sorted(feats["date"].unique())[39]
    r = B.run(feats, scores, B.IntradayRules(n_long=3, n_short=3))
    m = r["strategies"]["model"]
    assert m["trading_days"] == scores["date"].nunique()
    assert 0 <= m["direction_accuracy"] <= 1 and m["total_costs"] > 0
    assert len(r["grid"]) == 12
    assert "Intraday backtest" in B.format_report(r)


def test_model_save_load(feats, tmp_path):
    m = MI.IntradayModel.train(feats)
    day = feats[feats["date"] == feats["date"].max()]
    m.save(tmp_path)
    m2 = MI.IntradayModel.load(tmp_path)
    assert np.allclose(m.score(day), m2.score(day))
    assert len(m2.explain(day.head(2))) == 2
