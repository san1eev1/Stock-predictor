from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from stockpredictor.data import intraday as I


def day_bars(d="2026-09-21", path=None, vol=1000):
    """75 five-minute bars 9:15..15:25; `path` = close price per bar."""
    ts = pd.date_range(f"{d} 09:15", f"{d} 15:25", freq="5min")
    path = np.asarray(path if path is not None else np.full(len(ts), 100.0), dtype=float)
    return pd.DataFrame({"ts": ts, "open": path, "high": path + 0.1, "low": path - 0.1,
                         "close": path, "volume": vol})


def test_summary_first30_and_exit():
    path = np.full(75, 100.0)
    path[:6] = [99, 99.5, 100, 100.5, 101, 102]   # 9:15..9:40
    path[6:] = 102
    path[37] = 103                                  # bar 12:20-12:25
    path[38] = 104                                  # bar 12:25-12:30: the 12:30 exit price
    path[39] = 110                                  # after 12:30: not the trading exit
    path[72] = 105                                  # 15:15 bar (after exit)
    s = I.summarize_day(day_bars(path=path))
    assert s["open"] == 99 and s["c30"] == 102 and s["h30"] == pytest.approx(102.1)
    assert s["px_1230"] == 104 and s["px_1515"] == 102 and s["close"] == 102
    assert s["high_after"] == pytest.approx(110.1)  # the 15:15 spike is excluded


def test_first_hit_minutes():
    path = np.full(75, 100.0)
    path[8] = 99.0      # bar 9:55-10:00 -> first hit 15 min after 9:45
    path[20] = 101.6    # bar 10:55-11:00 -> 75 min after 9:45
    s = I.summarize_day(day_bars(path=path))
    assert s["d100"] == 15 and s["d075"] == 15 and np.isnan(s["d150"])
    assert s["u150"] == 75 and np.isnan(s["u200"])


def test_incomplete_days_skipped_and_multi_day():
    bars = pd.concat([day_bars("2026-09-21"), day_bars("2026-09-22").head(3)])
    rows = I.summarize(bars, "ABC", "yahoo")
    assert [r["date"] for r in rows] == ["2026-09-21"]


def test_storage_merge_prefers_angel(tmp_path):
    y = I.summarize(day_bars("2026-09-21"), "ABC", "yahoo")
    a = [{**y[0], "c30": 123.0, "source": "angelone"}]
    I.upsert_store(tmp_path / "store", y)
    I.upsert_store(tmp_path / "store", y)            # idempotent
    I.upsert_backfill(a, tmp_path / "bf.csv")
    df = I.load_summaries(tmp_path / "store", tmp_path / "bf.csv")
    assert len(df) == 1 and df["c30"].iloc[0] == 123.0


def test_chunk_ranges_cover_period():
    r = I.chunk_ranges(date(2024, 1, 1), date(2024, 3, 10), 30)
    assert r[0] == (datetime(2024, 1, 1, 9, 15), datetime(2024, 1, 30, 15, 30))
    assert r[-1][1] == datetime(2024, 3, 10, 15, 30) and len(r) == 3


def test_split_factor():
    ca = pd.DataFrame([{"symbol": "ABC", "date": "2024-06-01", "kind": "split", "value": 2.0}])
    assert I.split_factor(ca, "ABC", date(2024, 5, 1)) == 0.5
    assert I.split_factor(ca, "ABC", date(2024, 7, 1)) == 1.0
