import pandas as pd
import pytest

from stockpredictor import db
from stockpredictor.paper import splits as SP


@pytest.fixture
def conn(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    c.execute("INSERT INTO paper_accounts (horizon, capital, cash) VALUES ('longterm', 1000, 0)")
    yield c
    c.close()


ACTS = pd.DataFrame({"symbol": ["SNDK"], "date": ["2026-11-20"], "kind": ["split"], "value": [8.0]})


def _buy(conn, status="open", exit_price=None, exit_time=None, reason=None):
    conn.execute("INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
                 "costs, status, reason, stop_loss, exit_price, exit_time, exit_reason, pnl) "
                 "VALUES ('longterm','SNDK','long',0.25,'2026-11-16 16:00',800,0.2,?,'top rank',"
                 "720,?,?,?,?)", (status, exit_price, exit_time, reason,
                                  None if exit_price is None else (exit_price - 800) * 0.25))
    conn.commit()


def test_open_position_adjusted_once(conn):
    _buy(conn)
    assert SP.adjust_open(conn, ACTS, "2026-11-25", fractional=True)
    t = conn.execute("SELECT qty, entry_price, stop_loss FROM paper_trades").fetchone()
    assert (t["qty"], t["entry_price"], t["stop_loss"]) == (2.0, 100.0, 90.0)
    assert not SP.adjust_open(conn, ACTS, "2026-11-25", fractional=True)   # idempotent


def test_split_stop_out_repaired(conn):
    _buy(conn, "closed", 94.39, "2026-11-23 16:00", "stop-loss")
    conn.execute("UPDATE paper_accounts SET cash = cash + 0.25 * 94.39")
    log = SP.repair(conn, ACTS, fractional=True)
    assert log
    t = conn.execute("SELECT status, qty, entry_price FROM paper_trades").fetchone()
    assert t["status"] == "open" and t["qty"] == 2.0 and t["entry_price"] == 100.0
    cash = conn.execute("SELECT cash FROM paper_accounts").fetchone()[0]
    assert cash == pytest.approx(0.1, abs=0.01)                 # the sale's money is taken back


def test_real_crash_not_repaired(conn):
    _buy(conn, "closed", 30.0, "2026-11-23 16:00", "stop-loss")    # 30 x 8 = 240: a real -70%
    assert not SP.repair(conn, ACTS, fractional=True)
