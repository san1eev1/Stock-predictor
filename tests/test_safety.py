from datetime import date, datetime

import pandas as pd

from stockpredictor import db
from stockpredictor.live import safety as S


def conn_(tmp_path):
    db.init_db(tmp_path / "t.db")
    return db.connect(tmp_path / "t.db")


def test_kill_switches(tmp_path):
    c = conn_(tmp_path)
    now = datetime(2026, 9, 28, 9, 46)
    assert S.stale_data(pd.Timestamp("2026-09-25"), now.date()) is None      # Fri -> Mon ok
    assert "weekdays old" in S.stale_data(pd.Timestamp("2026-09-18"), now.date())
    assert S.broken_scores(pd.Series([1.0, 1.0, 1.0])) == "all predictions are identical"
    assert S.broken_scores(pd.Series([1.0, None, None, 2.0])).endswith("missing")
    assert S.broken_scores(pd.Series([0.1, 0.5, 0.3])) is None
    assert "live prices for only 10" in S.intraday_kill(c, now, pd.Timestamp("2026-09-25"),
                                                        "intraday", 100_000, {f"S{i}": 1.0 for i in range(10)}, 250)
    for day, pnl in (("2026-09-28", -2000), ("2026-09-29", -1500)):
        c.execute("INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
                  "costs, status, reason, pnl) VALUES ('intraday', 'X', 'long', 1, ?, 100, 0, "
                  "'closed', 'x', ?)", (f"{day} 09:45", pnl))
    assert "this week's loss" in S.intraday_week_loss(c, "intraday", date(2026, 9, 30), 100_000)
    assert S.intraday_week_loss(c, "intraday", date(2026, 10, 5), 100_000) is None  # new week
    for d, eq in (("2026-01-01", 100_000), ("2026-06-01", 140_000), ("2026-09-01", 100_000)):
        c.execute("INSERT INTO paper_equity VALUES ('longterm', ?, 0, 0, ?)", (d, eq))
    assert "below its peak" in S.longterm_drawdown(c)               # 140k -> 100k = -29%


def test_decay_needs_clearly_worse_than_random(tmp_path):
    c = conn_(tmp_path)
    days = pd.bdate_range("2026-08-01", periods=25)
    for i, d in enumerate(days):
        for k in range(10):
            c.execute("INSERT INTO predictions (horizon, date, symbol, direction, confidence, rank, "
                      "entry_price, correct, base_rate) VALUES ('intraday', ?, ?, 'up', 0.5, ?, 100, ?, 0.5)",
                      (f"{d:%Y-%m-%d}", f"S{k}", k + 1, int(k < 4)))        # 40% right vs 50%
    assert "40% right vs 50%" in S.decay(c, "intraday")
