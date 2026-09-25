import json

import pandas as pd
import pytest

from stockpredictor import db
from stockpredictor.data import news as N
from stockpredictor.features import longterm as F
from stockpredictor.models import intraday as MI
from stockpredictor.models import longterm as M
from stockpredictor.models import trainer as T
from stockpredictor.paper import engine as E
from test_features_longterm import synthetic


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MODEL_DIR", tmp_path / "lt")
    monkeypatch.setattr(MI, "MODEL_DIR", tmp_path / "id")
    daily, indices, universe = synthetic(n_days=1300, symbols=[f"S{i:02d}" for i in range(15)])
    ctx = E.MarketContext(daily, indices, universe, F.build_features(daily, indices, universe),
                          pd.DataFrame(columns=N.NEWS_COLS))
    db.init_db(tmp_path / "t.db")
    return ctx, db.connect(tmp_path / "t.db")


def test_candidates_include_current_and_vary():
    cur = M.current_params()
    c = T.candidates(cur, 4, seed=1)
    assert c[0] == cur and len(c) == 5
    assert len({json.dumps(x, sort_keys=True) for x in c}) == 5


def test_retrain_only_when_new_outcomes(env):
    ctx, conn = env
    assert T.retrain_longterm(ctx, conn) is not None
    assert T.retrain_longterm(ctx, conn) is None            # nothing new since last time
    assert conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0] == 1


def test_tune_logs_and_adopts_only_with_margin(env, monkeypatch):
    ctx, conn = env
    # 3 years: current, small gain, clear gain; then the 6-year check: winner, current
    scores = iter([0.02, 0.021, 0.05, 0.04, 0.03])
    monkeypatch.setattr(T, "evaluate_longterm",
                        lambda lab, p, years=3: {"ic": next(scores), "top10_hit": 0.5,
                                                 "top10_excess": 0.0})
    r = T.tune_longterm(ctx, conn, n_candidates=2, seed=3)
    assert r["adopted"] and r["ic"] == 0.05 and r["tested"] == 3
    assert (M.MODEL_DIR / "params.json").exists()
    assert json.loads((M.MODEL_DIR / "params.json").read_text()) == r["params"]
    kinds = [x[0] for x in conn.execute("SELECT train_from FROM model_runs")]
    assert kinds == ["tune", "retrain"]


def test_tune_rejects_winner_that_fails_longer_check(env, monkeypatch):
    ctx, conn = env
    scores = iter([0.02, 0.05, 0.01, 0.02, 0.03])   # wins on 3 years, loses on 6 years
    monkeypatch.setattr(T, "evaluate_longterm",
                        lambda lab, p, years=3: {"ic": next(scores), "top10_hit": 0.5,
                                                 "top10_excess": 0.0})
    r = T.tune_longterm(ctx, conn, n_candidates=2, seed=3)
    assert not r["adopted"] and r["long_check"] == {"best": 0.02, "current": 0.03}
    assert not (M.MODEL_DIR / "params.json").exists()


def test_tune_keeps_current_when_no_real_gain(env, monkeypatch):
    ctx, conn = env
    scores = iter([0.02, 0.022, 0.018])
    monkeypatch.setattr(T, "evaluate_longterm",
                        lambda lab, p, years=3: {"ic": next(scores), "top10_hit": 0.5,
                                                 "top10_excess": 0.0})
    r = T.tune_longterm(ctx, conn, n_candidates=2, seed=3)
    assert not r["adopted"] and not (M.MODEL_DIR / "params.json").exists()


def test_evaluate_longterm_real_walk_forward(env, monkeypatch):
    ctx, _ = env
    monkeypatch.setattr(M, "MIN_TRAIN_ROWS", 500)
    res = T.evaluate_longterm(T.longterm_labeled(ctx), M.current_params(), years=1)
    assert -1 <= res["ic"] <= 1 and res["weeks"] > 0
