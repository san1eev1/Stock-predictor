from stockpredictor import db, universe

SAMPLE_CSV = """Company Name,Industry,Symbol,Series,ISIN Code
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
Infosys Ltd.,Information Technology,INFY,EQ,INE009A01021
"""


def test_init_db_creates_tables(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    db.init_db(path)  # idempotent
    with db.connect(path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"stocks", "daily_prices", "intraday_prices", "news", "predictions",
            "paper_trades", "portfolio_trades", "model_runs", "app_settings"} <= tables


def test_parse_constituents():
    rows = universe.parse_constituents(SAMPLE_CSV)
    assert [r["symbol"] for r in rows] == ["RELIANCE", "INFY"]
    assert rows[1]["industry"] == "Information Technology"


def test_save_universe_marks_removed_stocks_inactive(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    with db.connect(path) as conn:
        universe.save_universe(conn, universe.parse_constituents(SAMPLE_CSV))
        universe.save_universe(conn, [{"symbol": "INFY", "name": "Infosys Ltd.",
                                       "industry": "IT", "isin": "INE009A01021"}])
        assert universe.active_symbols(conn) == ["INFY"]
        assert conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0] == 2


def test_old_paper_trades_table_is_rebuilt_keeping_rows(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.execute("""CREATE TABLE paper_trades (id INTEGER PRIMARY KEY AUTOINCREMENT,
            horizon TEXT NOT NULL CHECK (horizon IN ('intraday', 'longterm')), prediction_id INTEGER,
            symbol TEXT NOT NULL, side TEXT NOT NULL, qty INTEGER NOT NULL, entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL, exit_time TEXT, exit_price REAL, costs REAL, pnl REAL,
            status TEXT NOT NULL DEFAULT 'open')""")
        c.execute("INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price) "
                  "VALUES ('longterm', 'TCS', 'long', 1, '2026-01-01', 100)")
    db.init_db(path)
    with db.connect(path) as c:
        c.execute("INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price) "
                  "VALUES ('longterm_short', 'ITC', 'short', 1, '2026-01-01', 100)")
        assert c.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 2
        assert "stop_loss" in {r["name"] for r in c.execute("PRAGMA table_info(paper_trades)")}
