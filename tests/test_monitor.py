from datetime import datetime

import pandas as pd
import pytest

from stockpredictor import db
from stockpredictor.live import monitor as MON
from stockpredictor.live.prices import IST
from stockpredictor.paper import engine as E
from stockpredictor.portfolio import real as R
from test_paper import ctx_model  # noqa: F401  (fixture)


class FakePrices:
    source = "fake"

    def __init__(self, prices, live=True):
        self.prices, self.live = prices, live

    def get(self, symbols):
        return {s: self.prices[s] for s in symbols if s in self.prices}

    def market_is_live(self):
        return self.live


def make(tmp_path, ctx, model, prices, when):
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    clock = {"now": when}
    mon = MON.Monitor(conn, tmp_path, FakePrices(prices), 100_000,
                      clock=lambda: clock["now"], sync=lambda d: None)
    mon._ctx, mon._model = ctx, model
    return mon, conn, clock


def ist(*a):
    return datetime(*a, tzinfo=IST)


def test_paper_stop_loss_and_real_alert(tmp_path, ctx_model):
    ctx, model = ctx_model
    mon, conn, _ = make(tmp_path, ctx, model, {"S01": 80.0, "S02": 100.0},
                        ist(2026, 9, 21, 10, 0))
    conn.execute("INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
                 "costs, status) VALUES ('longterm', 'S01', 'long', 10, '2026-09-01', 100, 5, 'open')")
    R.add_trade(conn, "longterm", "S02", "buy", 5, 130.0, "2026-09-01")
    mon.minute_job(mon.clock())
    assert E.holdings(conn) == {}
    closed = conn.execute("SELECT exit_price, exit_reason FROM paper_trades").fetchone()
    assert tuple(closed) == (80.0, "stop-loss")
    kinds = [r[0] for r in conn.execute("SELECT source FROM alerts WHERE kind = 'stop-loss'")]
    assert sorted(kinds) == ["paper-longterm", "portfolio-longterm"]
    mon.minute_job(mon.clock())   # same alerts are not repeated
    assert conn.execute("SELECT COUNT(*) FROM alerts WHERE kind = 'stop-loss'").fetchone()[0] == 2


def test_pending_orders_fill_after_opening_minutes(tmp_path, ctx_model):
    ctx, model = ctx_model
    mon, conn, clock = make(tmp_path, ctx, model, {"S03": 50.0}, ist(2026, 9, 21, 9, 16))
    E.queue_orders(conn, [], ["S03"], "2026-09-18 18:00")
    mon.minute_job(clock["now"])
    assert E.holdings(conn) == {}
    clock["now"] = ist(2026, 9, 21, 9, 21)
    mon.minute_job(clock["now"])
    assert E.holdings(conn)["S03"].entry_price == 50.0


def test_tick_routing(tmp_path, ctx_model, monkeypatch):
    ctx, model = ctx_model
    last = ctx.daily["date"].max()
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(2026, 9, 26, 12, 0))  # Saturday
    monkeypatch.setattr(E.MarketContext, "load", classmethod(lambda cls, d: ctx))
    model.trained_at = "2000-01-01T00:00:00"
    monkeypatch.setattr(MON.M.LongTermModel, "save", lambda self, p: None)
    assert mon.tick() == ["retrain"]

    # After close on a weekday whose data is not yet published: wait.
    clock["now"] = ist(last.year, last.month, last.day, 17, 30) + pd.Timedelta(days=1)
    if clock["now"].weekday() >= 5:
        clock["now"] += pd.Timedelta(days=7 - clock["now"].weekday())
    assert mon.tick() == []
    # Data for the day exists: the decision runs once.
    mon._last_quarter = None
    clock["now"] = ist(last.year, last.month, last.day, 17, 30)
    assert mon.tick() == ["after-close"]
    mon._last_quarter = None
    assert mon.tick() == []
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 20


def test_live_scores_saved(tmp_path, ctx_model):
    ctx, model = ctx_model
    last = ctx.feats[ctx.feats["date"] == ctx.feats["date"].max()]
    prices = dict(zip(last["symbol"], last["close"] * 1.01))
    when = ctx.daily["date"].max() + pd.Timedelta(days=3)
    mon, conn, _ = make(tmp_path, ctx, model, prices, ist(when.year, when.month, when.day, 11, 0))
    mon.live_scores(mon.clock())
    rows = conn.execute("SELECT COUNT(*), MIN(rank), MAX(rank) FROM live_scores").fetchone()
    assert tuple(rows) == (len(prices), 1, len(prices))
