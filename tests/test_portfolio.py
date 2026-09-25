import pytest

from stockpredictor import db
from stockpredictor.portfolio import real as R


@pytest.fixture
def conn(tmp_path):
    db.init_db(tmp_path / "t.db")
    return db.connect(tmp_path / "t.db")


def test_average_cost_and_realized_pnl(conn):
    R.add_trade(conn, "longterm", "infy", "buy", 10, 100, "2026-01-01", charges=10)
    R.add_trade(conn, "longterm", "INFY", "buy", 10, 120, "2026-02-01", charges=10)
    R.add_trade(conn, "longterm", "INFY", "sell", 5, 150, "2026-03-01", charges=5)
    h = R.holdings(conn, "longterm").iloc[0]
    assert h["qty"] == 15
    assert h["avg_cost"] == pytest.approx(111.0)                 # (1000+10+1200+10)/20
    assert h["realized"] == pytest.approx(5 * 150 - 5 - 5 * 111)  # 190
    s = R.summary(conn, "longterm", {"INFY": 130.0})
    assert s.value == pytest.approx(1950)
    assert s.unrealized == pytest.approx(1950 - 15 * 111)
    assert s.table.iloc[0]["stop_loss"] == pytest.approx(111 * 0.85)


def test_cannot_oversell_and_horizons_separate(conn):
    R.add_trade(conn, "longterm", "TCS", "buy", 2, 3000, "2026-01-01")
    with pytest.raises(ValueError):
        R.add_trade(conn, "longterm", "TCS", "sell", 3, 3100, "2026-01-02")
    with pytest.raises(ValueError):
        R.add_trade(conn, "intraday", "TCS", "sell", 1, 3100, "2026-01-02")
    assert R.holdings(conn, "intraday").empty


def test_custom_stop_loss_and_delete(conn):
    tid = R.add_trade(conn, "longterm", "SBIN", "buy", 1, 800, "2026-01-01")
    R.set_stop_loss(conn, "longterm", None, 0.10)
    assert R.stop_loss_pct(conn, "longterm", "SBIN") == 0.10
    R.set_stop_loss(conn, "longterm", "SBIN", 0.05)
    assert R.stop_loss_pct(conn, "longterm", "SBIN") == 0.05
    R.delete_trade(conn, tid)
    assert R.trades(conn, "longterm").empty


def test_regime_filter_uses_only_the_previous_close():
    import pandas as pd

    from stockpredictor.backtest import portfolio as P

    days = pd.bdate_range("2025-01-01", periods=260)
    close = [100.0] * 250 + [50.0] * 10                   # crash on day 250
    idx = pd.DataFrame({"symbol": "NIFTY50", "date": days, "close": close})
    off = P.risk_off_days(idx)
    assert days[250] not in off and days[251] in off     # known only the day after
