from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stockpredictor import db
from stockpredictor.data import intraday as I
from stockpredictor.data import news as N
from stockpredictor.features import intraday as FI
from stockpredictor.features import longterm as F
from stockpredictor.live import monitor as MON
from stockpredictor.live.prices import IST
from stockpredictor.models import intraday as MI
from stockpredictor.paper import engine as E
from stockpredictor.paper import intraday as PI
from test_features_intraday import fake_summaries
from test_features_longterm import synthetic


class FakePrices:
    source = "fake"

    def __init__(self):
        self.prices = {}

    def get(self, symbols):
        return {s: self.prices[s] for s in symbols if s in self.prices}

    def market_is_live(self):
        return True

    def _angel_client(self):
        return None



@pytest.fixture(autouse=True)
def fast_tuning(monkeypatch):
    """Tuning rounds train many models; tests only check when they run."""
    calls = []
    monkeypatch.setattr(MON.Monitor, "_tune_round", lambda self, now, n, label: calls.append(label))
    monkeypatch.setattr(MON.store, "sync_models", lambda: None)       # no GitHub in tests
    monkeypatch.setattr(MON.store, "push_feedback", lambda judged: None)
    return calls

def test_intraday_day_cycle(tmp_path, monkeypatch):
    daily, indices, universe = synthetic(n_days=400, symbols=[f"S{i:02d}" for i in range(20)])
    universe = universe.assign(active=1, name=universe["symbol"])
    summ = fake_summaries(daily, n_days=80)
    last_day = pd.Timestamp(summ["date"].max())
    # Historical summaries in the store; "today" = the last day, served live.
    store_dir = tmp_path / "store"
    I.upsert_store(store_dir, summ[summ["date"] < summ["date"].max()].to_dict("records"))
    lt = F.build_features(daily, indices, universe)
    ctx = E.MarketContext(daily, indices, universe, lt, pd.DataFrame(columns=N.NEWS_COLS))
    feats = FI.build(summ, daily, lt)
    model = MI.IntradayModel.train(feats[feats["date"] < last_day])

    today = summ[summ["date"] == summ["date"].max()].set_index("symbol")
    f30 = {s: r[["open", "h30", "l30", "c30", "v30", "vwap30"]].to_dict() for s, r in today.iterrows()}
    monkeypatch.setattr(MON.intraday_bars, "first30", lambda client, syms, d: f30)

    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    prices = FakePrices()
    prices.prices = today["c30"].to_dict()
    clock = {"now": datetime(last_day.year, last_day.month, last_day.day, 9, 46, tzinfo=IST)}
    mon = MON.Monitor(conn, store_dir, prices, 100_000, clock=lambda: clock["now"],
                      sync=lambda d: None, capital_intraday=100_000)
    mon._ctx, mon._imodel = ctx, model
    mon._model = None
    mon._started = True
    monkeypatch.setattr(MON.Monitor, "quarter_job", lambda self, now: None)

    assert "intraday-picks" in mon.tick()
    assert len(PI.open_trades(conn)) == 10           # the 5 best buys + 5 best sells of 10 + 10
    assert "intraday-picks" not in mon.tick()          # only once per day

    clock["now"] = clock["now"].replace(hour=15, minute=16)
    prices.prices = today["px_1515"].to_dict()
    assert "square-off" in mon.tick()
    assert PI.open_trades(conn) == []
    acc = E.accuracy(conn, "intraday")
    assert acc["matured"] == 20 and acc["closed_trades"] == 10   # 10+10 judged, 5+5 traded
    assert conn.execute("SELECT COUNT(*) FROM alerts WHERE source = 'paper-intraday' "
                        "AND kind = 'decision'").fetchone()[0] == 1


def test_intraday_picks_retry_when_live_data_missing(tmp_path, monkeypatch):
    daily, indices, universe = synthetic(n_days=400, symbols=[f"S{i:02d}" for i in range(20)])
    universe = universe.assign(active=1, name=universe["symbol"])
    lt = F.build_features(daily, indices, universe)
    ctx = E.MarketContext(daily, indices, universe, lt, pd.DataFrame(columns=N.NEWS_COLS))
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    clock = {"now": datetime(2026, 9, 25, 11, 0, tzinfo=IST)}
    mon = MON.Monitor(conn, tmp_path, FakePrices(), 100_000, clock=lambda: clock["now"],
                      sync=lambda d: None)
    mon._ctx, mon._started = ctx, True
    mon._imodel = object()                       # any model: data check happens first
    calls = []
    monkeypatch.setattr(MON.intraday_bars, "first30", lambda c, s, d: calls.append(1) or {})
    mon.intraday_picks_job(clock["now"])
    assert MON._setting(conn, "id_last_picks") == ""          # not marked done
    mon.intraday_picks_job(clock["now"])                      # within 5 min: no retry
    clock["now"] = datetime(2026, 9, 25, 11, 6, tzinfo=IST)
    mon.intraday_picks_job(clock["now"])
    assert len(calls) == 2
