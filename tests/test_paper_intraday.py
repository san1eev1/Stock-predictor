import pandas as pd
import pytest

from stockpredictor import db
from stockpredictor.backtest.intraday import IntradayRules
from stockpredictor.models import intraday as MI
from stockpredictor.paper import engine as E
from stockpredictor.paper import intraday as PI
from test_intraday_backtest import feats  # noqa: F401 (fixture)


@pytest.fixture
def setup(tmp_path, feats):
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    model = MI.IntradayModel.train(feats[feats["date"] < feats["date"].max()])
    today = feats[feats["date"] == feats["date"].max()]
    return conn, model, today


def test_picks_open_long_and_short(setup):
    conn, model, today = setup
    prices = dict(zip(today["symbol"], today["c30"]))
    r = PI.run_picks(conn, today, model, prices, today["date"].iloc[0],
                     IntradayRules(n_long=3, n_short=3), 1_000_000)
    assert not r["skipped"] and len(r["picks"]) == 6
    sides = [t["side"] for t in PI.open_trades(conn)]
    assert sides.count("long") == 3 and sides.count("short") == 3
    v = PI.value(conn, prices)
    assert 998_000 < v["equity"] < 1_000_000       # only entry costs so far
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE horizon='intraday'").fetchone()[0]
    assert n == 6


def test_stop_loss_target_squareoff_and_evaluation(setup):
    conn, model, today = setup
    d = today["date"].iloc[0]
    prices = dict(zip(today["symbol"], today["c30"]))
    PI.run_picks(conn, today, model, prices, d, IntradayRules(n_long=2, n_short=2,
                                                              stop_loss=1.0, target=2.0), 1_000_000)
    trades = PI.open_trades(conn)
    long_ = next(t for t in trades if t["side"] == "long")
    short = next(t for t in trades if t["side"] == "short")
    moved = {**prices, long_["symbol"]: long_["entry_price"] * 0.985,   # long hits stop
             short["symbol"]: short["entry_price"] * 0.975}             # short hits target
    log = PI.check_exits(conn, moved, "10:30")
    assert len(log) == 2 and any("stop-loss" in x for x in log) and any("target" in x for x in log)
    PI.square_off(conn, moved, "15:15")
    assert PI.open_trades(conn) == []
    assert PI.evaluate_day(conn, f"{d:%Y-%m-%d}", moved) == 4
    acc = E.accuracy(conn, "intraday")
    assert acc["matured"] == 4 and acc["closed_trades"] == 4
    v = PI.value(conn, moved)
    assert v["positions"] == 0 and v["equity"] == pytest.approx(v["cash"])


def test_skip_weak_day_and_negative_news(setup):
    conn, model, today = setup
    prices = dict(zip(today["symbol"], today["c30"]))
    strong = pd.Series([10.0] * 30)   # history of much stronger days
    r = PI.run_picks(conn, today, model, prices, today["date"].iloc[0],
                     IntradayRules(skip_quantile=0.5), 100_000, strengths=strong)
    assert r["skipped"] and PI.open_trades(conn) == []
    top = today.assign(s=model.score(today)).sort_values("s")["symbol"].iloc[-1]
    PI.run_picks(conn, today, model, prices, today["date"].iloc[0],
                 IntradayRules(n_long=2, n_short=2), 100_000, negative_news={top})
    assert top not in [t["symbol"] for t in PI.open_trades(conn) if t["side"] == "long"]
