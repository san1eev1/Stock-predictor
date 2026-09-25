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



@pytest.fixture(autouse=True)
def fast_tuning(monkeypatch):
    """Tuning rounds train many models; tests only check when they run."""
    calls = []
    monkeypatch.setattr(MON.Monitor, "_tune_round", lambda self, now, n, label: calls.append(label))
    monkeypatch.setattr(MON.store, "sync_models", lambda: None)       # no GitHub in tests
    monkeypatch.setattr(MON.store, "push_feedback", lambda judged: None)
    return calls

def make(tmp_path, ctx, model, prices, when):
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    clock = {"now": when}
    mon = MON.Monitor(conn, tmp_path, FakePrices(prices), 100_000,
                      clock=lambda: clock["now"], sync=lambda d: None)
    mon._ctx, mon._model = ctx, model
    mon._started = True          # startup routine has its own test
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
    assert mon.tick() == ["retrain", "tune"]     # weekend: weekly retrain + a tuning round

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
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 10              # top 10 buys only (no down)


def test_live_scores_saved(tmp_path, ctx_model):
    ctx, model = ctx_model
    last = ctx.feats[ctx.feats["date"] == ctx.feats["date"].max()]
    prices = dict(zip(last["symbol"], last["close"] * 1.01))
    when = ctx.daily["date"].max() + pd.Timedelta(days=3)
    mon, conn, _ = make(tmp_path, ctx, model, prices, ist(when.year, when.month, when.day, 11, 0))
    mon.live_scores(mon.clock())
    rows = conn.execute("SELECT COUNT(*), MIN(rank), MAX(rank) FROM live_scores").fetchone()
    assert tuple(rows) == (len(prices), 1, len(prices))


def test_startup_trains_decides_and_prints_scoreboard(tmp_path, ctx_model, monkeypatch, caplog):
    import logging
    ctx, model = ctx_model
    last = ctx.daily["date"].max()
    monkeypatch.setattr(E.MarketContext, "load", classmethod(lambda cls, d: ctx))
    monkeypatch.setattr(MON.T, "retrain_longterm", lambda ctx, conn: None)
    monkeypatch.setattr(MON.T, "retrain_intraday", lambda ctx, d, conn: None)
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(last.year, last.month, last.day, 20, 0))
    mon._started = False
    with caplog.at_level(logging.INFO):
        assert "startup" in mon.tick()
    assert conn.execute("SELECT COUNT(*) FROM predictions WHERE horizon='longterm'").fetchone()[0] == 10
    assert any("Accuracy now" in r.message for r in caplog.records)
    assert "startup" not in mon.tick()          # only once per run


def test_unknown_market_state_counts_as_open_and_rechecks(tmp_path, ctx_model):
    ctx, model = ctx_model
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(2026, 9, 25, 10, 0))
    answers = iter([None, False])
    mon.prices.market_is_live = lambda: next(answers)
    assert mon.trading_today(clock["now"]) is True             # unknown -> assume open
    clock["now"] = ist(2026, 9, 25, 10, 10)
    assert mon.trading_today(clock["now"]) is True             # not re-asked within 30 min
    clock["now"] = ist(2026, 9, 25, 10, 45)
    assert mon.trading_today(clock["now"]) is False            # re-asked: holiday


def test_scheduled_run(tmp_path, ctx_model, monkeypatch):
    ctx, model = ctx_model
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(2026, 9, 25, 18, 0))  # Friday
    calls = []
    monkeypatch.setattr(mon, "after_close_job", lambda now: calls.append("ac") or True)
    monkeypatch.setattr(mon, "weekly_retrain", lambda now: calls.append("wr") or True)
    MON._set(conn, "monitor_heartbeat", ist(2026, 9, 25, 17, 58).isoformat())
    assert mon.scheduled_run(clock["now"]) == []          # live monitor is running
    clock["now"] = ist(2026, 9, 25, 21, 30)
    assert mon.scheduled_run(clock["now"]) == ["after-close"]
    clock["now"] = ist(2026, 9, 26, 18, 0)                # Saturday
    assert mon.scheduled_run(clock["now"]) == ["retrain"] + ["tune"] * MON.SCHEDULED_TUNE_ROUNDS
    assert calls == ["ac", "wr"]


def test_schedule_plist():
    from stockpredictor import scheduler

    p = scheduler.build_plist("/x/python")
    assert p["ProgramArguments"][-4:] == ["/x/python", "-m", "stockpredictor", "auto"]
    assert {"Hour": 18, "Minute": 0} in p["StartCalendarInterval"]


def test_idle_tuning_hourly_only_when_market_closed(tmp_path, ctx_model, fast_tuning):
    ctx, model = ctx_model
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(2026, 9, 25, 11, 0))   # Friday
    assert not mon.market_idle(clock["now"])                  # market open: never tune
    clock["now"] = ist(2026, 9, 25, 18, 0)
    assert not mon.market_idle(clock["now"])                  # after-close decision not made yet
    MON._set(conn, "lt_after_close_day", "2026-09-25")
    assert mon.market_idle(clock["now"])
    assert mon.idle_tune(clock["now"]) is True
    clock["now"] = ist(2026, 9, 25, 18, 30)
    assert mon.idle_tune(clock["now"]) is False               # less than an hour since the last
    clock["now"] = ist(2026, 9, 25, 19, 1)
    assert mon.idle_tune(clock["now"]) is True
    assert fast_tuning == ["background", "background"]
    assert mon.market_idle(ist(2026, 9, 26, 3, 0))            # Saturday night


class _FakeBackground:
    def __init__(self):
        import threading
        self.model_changed = threading.Event()
        self.days = []

    def learn_from_today(self, day):
        self.days.append(day)


def test_live_learning_after_square_off(tmp_path, ctx_model, fast_tuning, monkeypatch):
    ctx, model = ctx_model
    monkeypatch.setattr(MON.Monitor, "weekly_retrain", lambda self, now: False)
    monkeypatch.setattr(MON.Monitor, "quarter_job", lambda self, now: None)
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(2026, 9, 25, 15, 10))
    mon.background = bg = _FakeBackground()
    mon._imodel = "old"
    mon._last_quarter = clock["now"]                            # 15-minute job not due
    assert "square-off" in mon.tick() and "live-learn" not in mon.tick()   # 12:30 exit done
    clock["now"] = ist(2026, 9, 25, 15, 25)
    assert "live-learn" not in mon.tick()                      # market still open: wait
    clock["now"] = ist(2026, 9, 25, 15, 32)
    assert "live-learn" in mon.tick() and bg.days == [clock["now"].date()]
    assert "live-learn" not in mon.tick()                       # once per day
    bg.model_changed.set()
    mon.tick()
    assert mon._imodel is None and not bg.model_changed.is_set()  # improved model reloaded
    clock["now"] = ist(2026, 9, 26, 3, 0)                        # background thread tunes,
    assert "tune" not in mon.tick() and fast_tuning == []        # so the monitor doesn't


def test_mac_trains_nothing_when_training_is_on_github(tmp_path, ctx_model, monkeypatch):
    ctx, model = ctx_model
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(2026, 9, 26, 12, 0))  # Saturday
    monkeypatch.setattr(E.MarketContext, "load", classmethod(lambda cls, d: ctx))
    monkeypatch.setattr(MON.config, "MAC_TRAINING", False)
    monkeypatch.setattr(MON.T, "retrain_intraday", lambda *a, **k: pytest.fail("trained"))
    monkeypatch.setattr(MON.T, "tune_intraday", lambda *a, **k: pytest.fail("tuned"))
    assert "retrain" not in mon.tick() and "tune" not in mon.tick()


def test_mac_trains_once_at_four(tmp_path, ctx_model, monkeypatch):
    ctx, model = ctx_model
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(2026, 9, 28, 15, 50))  # Monday
    monkeypatch.setattr(E.MarketContext, "load", classmethod(lambda cls, d: ctx))
    monkeypatch.setattr(MON.config, "MAC_TRAINING", False)
    monkeypatch.setattr(MON.config, "MAC_DAILY_TRAINING", True)
    calls = []
    monkeypatch.setattr(mon, "daily_mac_train", lambda c: calls.append(1))
    monkeypatch.setattr(mon, "angel_topup", lambda now: None)
    mon._started = True
    assert "mac-train" not in mon.tick()                  # 15:50: not yet
    clock["now"] = ist(2026, 9, 28, 16, 1)
    assert "mac-train" in mon.tick() and calls == [1]
    assert "mac-train" not in mon.tick() and calls == [1]  # once a day


def test_mac_starts_github_runs_on_time(tmp_path, ctx_model, monkeypatch):
    ctx, model = ctx_model
    mon, conn, clock = make(tmp_path, ctx, model, {}, ist(2026, 9, 28, 20, 59))  # Monday
    monkeypatch.setattr(E.MarketContext, "load", classmethod(lambda cls, d: ctx))
    monkeypatch.setattr(MON.config, "CLOUD_TRAINING", True)
    monkeypatch.setattr(MON.config, "MAC_DAILY_TRAINING", False)
    started = []
    monkeypatch.setattr(MON, "start_github_run", lambda w: started.append(w) or True)
    mon._started = True
    for key in ("gh_run_1630",):
        MON._set(conn, key, "2026-09-28")           # the data run was started earlier
    mon.tick()
    assert started == []                             # 20:59: too early for training
    clock["now"] = ist(2026, 9, 28, 21, 2)
    assert "github-train-models" in mon.tick() and started == ["train-models.yml"]
    mon.tick()
    assert started == ["train-models.yml"]           # once per slot
