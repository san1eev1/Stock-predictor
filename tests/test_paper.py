import pandas as pd
import pytest

from stockpredictor import db
from stockpredictor.backtest.portfolio import Rules
from stockpredictor.data import news as N
from stockpredictor.features import longterm as F
from stockpredictor.models import longterm as M
from stockpredictor.paper import engine as E
from test_features_longterm import synthetic


@pytest.fixture(scope="module")
def ctx_model():
    daily, indices, _ = synthetic(n_days=700, symbols=[f"S{i:02d}" for i in range(25)])
    universe = pd.DataFrame({"symbol": daily["symbol"].unique(), "industry": "X", "active": 1})
    feats = F.build_features(daily, indices, universe)
    labeled = M.add_labels(F.weekly_snapshots(feats), daily, indices)
    model = M.LongTermModel.train(pd.concat([labeled] * 10))
    ctx = E.MarketContext(daily, indices, universe, feats, pd.DataFrame(columns=N.NEWS_COLS))
    return ctx, model


@pytest.fixture
def conn(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    E.ensure_account(c, 100_000)
    return c


def test_decisions_queue_then_fill_next_day(conn, ctx_model):
    ctx, model = ctx_model
    dates = sorted(ctx.feats["date"].unique())[-80:]
    rules = Rules(n_hold=5, exit_rank=10)
    r1 = E.run_decision(conn, ctx, model, dates[0], rules)
    assert r1["rebalance"] and len(r1["buys"]) == 5 and r1["fills"] == []
    assert E.holdings(conn) == {}                          # nothing filled yet
    r2 = E.run_decision(conn, ctx, model, dates[1], rules)
    assert len(r2["fills"]) == 5                           # filled at next day's close
    pos = E.holdings(conn)
    assert len(pos) == 5
    prices = ctx.closes_on(dates[1])
    assert all(p.entry_price == prices[s] for s, p in pos.items())
    v = E.value(conn, prices)
    assert v["equity"] < 100_000 and v["equity"] > 99_000   # only costs lost on fill day
    assert v["cash"] >= 0


def test_weekly_mode_rebalances_every_7_days(conn, ctx_model):
    ctx, model = ctx_model
    dates = sorted(ctx.feats["date"].unique())[-30:]
    flags = [E.run_decision(conn, ctx, model, d)["rebalance"] for d in dates[:8]]
    assert flags[0] and sum(flags) == 2


def test_predictions_saved_and_evaluated(conn, ctx_model):
    ctx, model = ctx_model
    dates = sorted(ctx.feats["date"].unique())
    E.run_decision(conn, ctx, model, dates[-40])
    n = conn.execute("SELECT COUNT(*), SUM(direction='up') FROM predictions").fetchone()
    assert tuple(n) == (20, 10)
    assert E.evaluate_predictions(conn, ctx, horizon_days=20) == 20
    acc = E.accuracy(conn)
    assert acc["matured"] == 20 and 0 <= acc["accuracy"] <= 1
    assert 0 <= acc["random_baseline"] <= 1


def test_negative_news_blocks_buy(conn, ctx_model):
    ctx, model = ctx_model
    date = sorted(ctx.feats["date"].unique())[-10]
    first = E.run_decision(conn, ctx, model, date)["buys"][0]
    conn.execute("DELETE FROM paper_orders")
    conn.execute("DELETE FROM app_settings")
    bad = pd.DataFrame([{"symbol": first, "published": pd.Timestamp(date, tz="UTC"),
                         "source": "x", "title": "fraud probe", "url": "u", "sentiment": -0.95}])
    ctx2 = E.MarketContext(ctx.daily, ctx.indices, ctx.universe, ctx.feats, bad)
    assert first not in E.run_decision(conn, ctx2, model, date)["buys"]
