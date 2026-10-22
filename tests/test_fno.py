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
