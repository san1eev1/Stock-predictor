from datetime import date

from stockpredictor import db
from stockpredictor.paper import fno as FO

ITEMS = [
    {"token": "1", "symbol": "ABC27OCT26FUT", "name": "ABC", "expiry": "27OCT2026", "strike": "-1",
     "lotsize": "100", "instrumenttype": "FUTSTK"},
    {"token": "2", "symbol": "ABC27OCT26500CE", "name": "ABC", "expiry": "27OCT2026",
     "strike": "50000", "lotsize": "100", "instrumenttype": "OPTSTK"},
    {"token": "3", "symbol": "ABC27OCT26520PE", "name": "ABC", "expiry": "27OCT2026",
     "strike": "52000", "lotsize": "100", "instrumenttype": "OPTSTK"},
    {"token": "4", "symbol": "ABC27OCT26500PE", "name": "ABC", "expiry": "27OCT2026",
     "strike": "50000", "lotsize": "100", "instrumenttype": "OPTSTK"},
]


class Client:
    def __init__(self, px):
        self.px = px

    def _require_login(self):
        return self

    def getMarketData(self, mode, req):
        return {"data": {"fetched": [{"symbolToken": t, "ltp": self.px[t]} for t in req["NFO"]]}}


def test_contracts_and_books(tmp_path, monkeypatch):
    today = date(2026, 10, 22)
    monkeypatch.setattr(FO, "instruments", lambda d: ITEMS)
    assert FO.future(ITEMS, "ABC", today).symbol == "ABC27OCT26FUT"
    assert FO.atm_option(ITEMS, "ABC", 503, "PE", today).strike == 500
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    opened = FO.open_books(conn, Client({"1": 500.0, "2": 12.0, "4": 9.0}),
                           [("ABC", "long")], {"ABC": 503.0}, today, "2026-10-22 09:46")
    assert len(opened) == 2           # 1 future (margin 10k of 100k) + call option
    fut = FO.value(conn, {"ABC27OCT26FUT": 510.0}, FO.FUT_BOOK)
    assert fut["pnl"] > 0             # long future gained Rs 10 x lots x 100
    FO.square_off(conn, {"ABC27OCT26FUT": 510.0, "ABC27OCT26500CE": 15.0}, "2026-10-22 15:15")
    for b in FO.BOOKS:
        v = FO.value(conn, {}, b)
        assert v["positions"] == 0 and v["pnl"] > 0


def test_trade_outcomes_feed_learning(tmp_path):
    from stockpredictor.models import trainer as T

    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    conn.execute("INSERT INTO fno_contracts VALUES ('ABC27OCT26FUT', '1', 'ABC', 100)")
    for h, sym, pnl in (("fno_futures", "ABC27OCT26FUT", -50.0), ("intraday_close", "ABC", 20.0),
                        ("intraday", "XYZ", 10.0)):
        conn.execute("INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
                     "costs, status, pnl) VALUES (?, ?, 'long', 1, '2026-10-22 09:46', 1, 0, "
                     "'closed', ?)", (h, sym, pnl))
    out = T.paper_trade_outcomes(conn).set_index(["horizon", "symbol"])["correct"]
    assert out[("intraday_close", "ABC")] == 0      # the futures loss counts for the stock
    assert out[("intraday", "XYZ")] == 1


def test_trial_threshold_rises(tmp_path):
    from stockpredictor.models import trainer as T

    assert T.trial_t_min("x", 3, tmp_path) == T.MIN_T_STAT
    assert T.trial_t_min("x", 500, tmp_path) > 3.4
    T.reset_trials("x", tmp_path)
    assert T.trial_t_min("x", 1, tmp_path) == T.MIN_T_STAT


def test_fiidii_parse_and_update(tmp_path, monkeypatch):
    from stockpredictor.data import fiidii

    js = [{"buyValue": "19232.31", "category": "DII", "date": "23-Oct-2026", "netValue": "3235.36",
           "sellValue": "15996.95"}, {"buyValue": "14445.54", "category": "FII/FPI",
                                      "date": "23-Oct-2026", "netValue": "887.65", "sellValue": "13557.89"}]
    monkeypatch.setattr(fiidii, "fetch", lambda: fiidii.parse(js))
    assert fiidii.update(tmp_path) == 2 and fiidii.update(tmp_path) == 0   # no duplicates
    assert set(fiidii.load(tmp_path)["category"]) == {"FII", "DII"}
