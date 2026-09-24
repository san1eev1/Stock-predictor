import json

import numpy as np
import pandas as pd
import pytest

from stockpredictor import db
from stockpredictor.models import longterm as M
from stockpredictor.models import trainer as T
from stockpredictor.paper import engine as E
from test_paper import ctx_model  # noqa: F401 (fixture)


def test_training_snapshots_move_daily(ctx_model):
    ctx, _ = ctx_model
    days = sorted(ctx.feats["date"].unique())
    a = M.training_snapshots(ctx.feats)
    b = M.training_snapshots(ctx.feats[ctx.feats["date"] < days[-1]])
    assert a["date"].max() == days[-1] and b["date"].max() == days[-2]
    assert a["date"].nunique() == pytest.approx(len(days) / 5, abs=1)


def test_sample_weights():
    df = pd.DataFrame({"date": pd.to_datetime(["2016-01-01", "2026-01-01"]),
                       "target": [0.5, 1.0], "fb_weight": [np.nan, 2.0]})
    w = M.sample_weights(df, {"recency_half_life": 5, "tail_weight": 1})
    assert w[0] == pytest.approx(0.25, rel=0.01)          # 10 years old, half-life 5
    assert w[1] == pytest.approx(1 * 2 * 2)                # extreme target x2, feedback x2
    assert np.allclose(M.sample_weights(df.drop(columns="fb_weight"), {}), 1)


@pytest.fixture
def conn(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    E.ensure_account(c, 100_000)
    return c


def test_paper_feedback_adds_weighted_rows(conn, ctx_model):
    ctx, model = ctx_model
    day = sorted(ctx.feats["date"].unique())[-120]
    E.run_decision(conn, ctx, model, day)
    E.evaluate_predictions(conn, ctx, horizon_days=20)
    labeled = T.longterm_labeled(ctx)
    out = T.paper_feedback(ctx, conn, labeled)
    fb = out.dropna(subset=["fb_weight"])
    assert len(fb) == 20 and set(fb["fb_weight"]) <= {1.5, 2.0}
    assert not out.duplicated(["symbol", "date"]).any()
    assert fb["target"].notna().all()


def test_shadow_race_and_live_switch(conn, ctx_model, tmp_path, monkeypatch):
    ctx, model = ctx_model
    monkeypatch.setattr(M, "MODEL_DIR", tmp_path / "lt")
    for day in sorted(ctx.feats["date"].unique())[-160:-130]:
        E.run_decision(conn, ctx, model, day)
    n = conn.execute("SELECT COUNT(DISTINCT variant), COUNT(*) FROM shadow_predictions").fetchone()
    assert tuple(n) == (3, 3 * 20 * 30)
    E.evaluate_predictions(conn, ctx)
    race = E.strategy_race(conn)
    assert set(race["variant"]) == set(E.SHADOW_VARIANTS) and (race["days"] == 30).all()
    # Force a clear live winner and check the switch.
    conn.execute("UPDATE shadow_predictions SET correct = (variant = 'Momentum only')")
    conn.commit()
    sel = T.live_selection(conn)
    assert sel["switched"] and sel["best"] == "Momentum only"
    assert json.loads((M.MODEL_DIR / "params.json").read_text())["mom_weight"] == 1.0
    assert T.live_selection(conn)["switched"] is False       # already on the winner
