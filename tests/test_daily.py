import sqlite3
from datetime import date

import pandas as pd

from stockpredictor import db
from stockpredictor.data import daily


def frame(days, closes, splits=None):
    idx = pd.DatetimeIndex(pd.to_datetime(days)).tz_localize("Asia/Kolkata")
    return pd.DataFrame({
        "Open": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes],
        "Close": closes, "Adj Close": closes, "Volume": [1000] * len(closes),
        "Dividends": [0.0] * len(closes), "Stock Splits": splits or [0.0] * len(closes),
    }, index=idx)


class FakeDownloader:
    def __init__(self, frames):
        self.frames = frames  # list returned in order
        self.calls = []

    def __call__(self, ticker, start):
        self.calls.append((ticker, start))
        return self.frames.pop(0)


def make_db(tmp_path) -> sqlite3.Connection:
    path = tmp_path / "t.db"
    db.init_db(path)
    return db.connect(path)


def test_history_to_rows_skips_bad_candles():
    df = frame(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 102.0])
    df.loc[df.index[1], "Close"] = float("nan")
    df.loc[df.index[2], "High"] = 50.0  # high below close -> inconsistent
    rows = daily.history_to_rows("ABC", df)
    assert [r[1] for r in rows] == ["2024-01-01"]


def test_update_stock_first_download_and_incremental(tmp_path):
    conn = make_db(tmp_path)
    fake = FakeDownloader([
        frame(["2024-01-01", "2024-01-02"], [100.0, 101.0]),
        frame(["2024-01-02", "2024-01-03"], [101.5, 103.0]),  # overlap corrects partial day
    ])
    assert daily.update_stock(conn, "ABC", date(2024, 1, 1), fake) == 2
    daily.update_stock(conn, "ABC", date(2024, 1, 1), fake)
    closes = conn.execute("SELECT date, close FROM daily_prices ORDER BY date").fetchall()
    assert [tuple(r) for r in closes] == [
        ("2024-01-01", 100.0), ("2024-01-02", 101.5), ("2024-01-03", 103.0)]
    assert fake.calls[0] == ("ABC.NS", date(2024, 1, 1))
    assert fake.calls[1][1] == date(2024, 1, 2) - pd.Timedelta(days=daily.OVERLAP_DAYS)


def test_new_split_triggers_full_redownload(tmp_path):
    conn = make_db(tmp_path)
    fake = FakeDownloader([
        frame(["2024-01-01", "2024-01-02"], [200.0, 202.0]),
        frame(["2024-01-03"], [102.0], splits=[2.0]),                   # split seen
        frame(["2024-01-01", "2024-01-02", "2024-01-03"],               # adjusted history
              [100.0, 101.0, 102.0], splits=[0.0, 0.0, 2.0]),
    ])
    daily.update_stock(conn, "ABC", date(2024, 1, 1), fake)
    daily.update_stock(conn, "ABC", date(2024, 1, 1), fake)
    closes = [r[0] for r in conn.execute("SELECT close FROM daily_prices ORDER BY date")]
    assert closes == [100.0, 101.0, 102.0]
    assert conn.execute("SELECT kind, value FROM corporate_actions").fetchone()[:] == ("split", 2.0)
    assert len(fake.calls) == 3


def test_migration_adds_adj_close_to_phase1_database(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, "
                     "low REAL, close REAL, volume INTEGER, source TEXT, "
                     "PRIMARY KEY (symbol, date))")
    db.init_db(path)
    with db.connect(path) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(daily_prices)")}
    assert "adj_close" in cols


def test_extend_back_adds_only_older_rows(tmp_path):
    conn = make_db(tmp_path)
    fake = FakeDownloader([frame(["2024-06-03", "2024-06-04"], [100.0, 101.0])])
    daily.update_stock(conn, "ABC", date(2024, 6, 1), fake)
    older = FakeDownloader([frame(["2024-01-02", "2024-01-03", "2024-06-03"], [90.0, 91.0, 555.0])])
    older_calls = []
    assert daily.extend_back(conn, "ABC", date(2024, 1, 1),
                             downloader=lambda t, s, e: (older_calls.append((s, e)), older(t, s))[1]) == 2
    closes = [r[0] for r in conn.execute("SELECT close FROM daily_prices ORDER BY date")]
    assert closes == [90.0, 91.0, 100.0, 101.0]          # stored 2024-06-03 not overwritten
    assert older_calls[0] == (date(2024, 1, 1), date(2024, 6, 3))
    assert daily.extend_back(conn, "ABC", date(2024, 1, 1), downloader=older) == 0  # nothing older
