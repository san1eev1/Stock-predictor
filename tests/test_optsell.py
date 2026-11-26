from datetime import date, datetime

from stockpredictor import db
from stockpredictor.paper import optsell as OS

FLY = next(s for s in OS.STRATEGIES if s.book == "os_fly_920")
CONDOR = next(s for s in OS.STRATEGIES if s.book == "os_condor")


def _items(expiry="02DEC2026"):
    rows = []
    for k in range(-15, 16):
        strike = 26000 + 50 * k
        for kind in ("CE", "PE"):
            rows.append({"token": f"{strike}{kind}", "symbol": f"NIFTY02DEC26{strike}{kind}",
                         "name": "NIFTY", "expiry": expiry, "strike": f"{strike * 100}",
                         "lotsize": "65"})
    return rows


def _db(tmp_path):
    db.init_db(tmp_path / "t.db")
    return db.connect(tmp_path / "t.db")


def test_iron_fly_stop_and_stop_to_cost(tmp_path):
    legs = OS.legs_for(_items(), 26012.0, date(2026, 11, 27), FLY)
    assert [(l.contract.strike, l.contract.kind, l.side) for l in legs] == [
        (26000, "CE", "short"), (26000, "PE", "short"), (26500, "CE", "long"), (25500, "PE", "long")]
    prices = {legs[0].contract.symbol: 150.0, legs[1].contract.symbol: 140.0,
              legs[2].contract.symbol: 20.0, legs[3].contract.symbol: 18.0}
    assert OS.max_loss_per_lot(legs, prices) == (500 - 252) * 65
    conn = _db(tmp_path)
    assert len(OS.enter(conn, FLY, legs, prices, "2026-11-27 09:20", 150_000)) == 4
    ce, pe = legs[0].contract.symbol, legs[1].contract.symbol
    up = {**prices, ce: 200.0}
    log = OS.check_stops(conn, FLY, up, "2026-11-27 10:05")
    assert any("stop-loss" in x for x in log) and any("wing" in x for x in log)
    left = {t["symbol"]: t for t in OS.open_legs(conn, FLY.book)}
    assert set(left) == {pe, legs[3].contract.symbol}
    assert left[pe]["stop_loss"] == left[pe]["entry_price"]
    OS.square_off(conn, FLY.book, {**up, pe: 60.0}, "2026-11-27 15:15")
    assert OS.open_legs(conn, FLY.book) == []


def test_condor_takes_profit_at_half_the_credit(tmp_path):
    legs = OS.legs_for(_items(), 26000.0, date(2026, 11, 27), CONDOR)
    assert [l.contract.strike for l in legs] == [26250, 25750, 26650, 25350]   # 1% / 2.5% away
    prices = {legs[0].contract.symbol: 60.0, legs[1].contract.symbol: 55.0,
              legs[2].contract.symbol: 10.0, legs[3].contract.symbol: 9.0}
    conn = _db(tmp_path)
    OS.enter(conn, CONDOR, legs, prices, "2026-11-27 09:30", 150_000)
    half = {legs[0].contract.symbol: 20.0, legs[1].contract.symbol: 18.0,
            legs[2].contract.symbol: 3.0, legs[3].contract.symbol: 2.0}
    log = OS.check_stops(conn, CONDOR, half, "2026-11-27 12:00")
    assert len(log) == 4 and all("profit target" in x for x in log)


def test_filters_skip_without_trading(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    monkeypatch.setattr(OS, "instruments", lambda d: _items())
    monkeypatch.setattr(OS, "index_spot", lambda c: 26000.0)
    monkeypatch.setattr(OS.FO, "ltp", lambda c, t: {s: 50.0 for s in t})
    out = OS.tick(conn, None, datetime(2026, 11, 27, 9, 31), 150_000, vix=12.0, realized=14.0)
    assert any("VIX not above realized" in x for x in out)        # VRP filter
    out = OS.tick(conn, None, datetime(2026, 11, 27, 9, 46), 150_000, vix=12.0, realized=14.0)
    assert any("not an expiry day" in x for x in out)              # expiry 2 Dec, not 27 Nov
    assert not OS.entered(conn, "os_condor_vrp", "2026-11-27")
