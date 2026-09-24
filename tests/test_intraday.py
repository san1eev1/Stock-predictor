from datetime import date, datetime

from stockpredictor import db
from stockpredictor.data import intraday, quality


class FakeClient:
    def __init__(self):
        self.calls = []

    def candles(self, token, interval, frm, to):
        self.calls.append((frm, to))
        ts = frm.replace(hour=9, minute=15).isoformat() + "+05:30"
        return [[ts, 100, 101, 99, 100.5, 5000]]


def test_chunk_ranges_cover_period_without_overlap():
    ranges = intraday.chunk_ranges(date(2024, 1, 1), date(2024, 3, 10), 30)
    assert ranges[0] == (datetime(2024, 1, 1, 9, 15), datetime(2024, 1, 30, 15, 30))
    assert ranges[1][0] == datetime(2024, 1, 31, 9, 15)
    assert ranges[-1][1] == datetime(2024, 3, 10, 15, 30)
    assert len(ranges) == 3


def test_update_symbol_resumes_from_last_stored_day(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    client = FakeClient()
    with db.connect(path) as conn:
        intraday.update_symbol(conn, client, "ABC", "1", "ONE_MINUTE",
                               date(2024, 1, 1), date(2024, 1, 20))
        intraday.update_symbol(conn, client, "ABC", "1", "ONE_MINUTE",
                               date(2024, 1, 1), date(2024, 1, 25))
        assert client.calls[-1][0] == datetime(2024, 1, 1, 9, 15)
        assert conn.execute("SELECT COUNT(*) FROM intraday_prices").fetchone()[0] == 1


def test_split_factor(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    with db.connect(path) as conn:
        conn.execute("INSERT INTO corporate_actions VALUES ('ABC', '2024-06-01', 'split', 2.0)")
        assert intraday.split_factor(conn, "ABC", date(2024, 5, 1)) == 0.5
        assert intraday.split_factor(conn, "ABC", date(2024, 7, 1)) == 1.0


def test_quality_report_flags_gaps(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    with db.connect(path) as conn:
        conn.executemany(
            "INSERT INTO daily_prices (symbol, date, open, high, low, close, volume) "
            "VALUES ('ABC', ?, 1, 1, 1, 1, 10)",
            [("2024-01-01",), ("2024-01-02",), ("2024-02-01",)])
        report = quality.daily_report(conn, ["ABC", "XYZ"])
    assert report[0]["gaps"] == 1 and report[0]["rows"] == 3
    assert report[1]["rows"] == 0
    assert "1 without data" in quality.format_report(report)
