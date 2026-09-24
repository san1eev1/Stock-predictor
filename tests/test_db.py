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
