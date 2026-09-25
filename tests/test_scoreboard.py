from datetime import datetime

from stockpredictor import db
from stockpredictor.models import longterm as M
from stockpredictor.paper import scoreboard as SB


def test_intraday_today_and_lines(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    day = "2026-09-25"
    rows = [("A", "up", 100), ("B", "up", 100), ("C", "down", 100), ("D", "down", 100)]
    c.executemany("INSERT INTO predictions (horizon, date, symbol, direction, entry_price) "
                  "VALUES ('intraday', ?, ?, ?, ?)", [(day, *r) for r in rows])
    c.executemany("INSERT INTO intraday_open VALUES (?, ?, ?)",
                  [(day, s, 100.0) for s in "ABCDEF"])
    prices = {"A": 101, "B": 99, "C": 98, "D": 97, "E": 101, "F": 102}
    it = SB.intraday_today(c, day, prices)
    assert (it["buy_right"], it["sell_right"]) == (1, 2)
    assert it["right"] == 0.75 and it["random_up"] == 0.5
    text = "\n".join(SB.lines({**SB.compute(c, datetime(2026, 9, 25, 11, 0)),
                               "intraday_today": it}))
    assert "buys 1/2 up, sells 2/2 down -> 75% right (random picks: 50%)" in text
    assert "Long-term judged: first results one week" in text


def test_by_direction_splits_up_and_down(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    rows = [("A", "up", 1), ("B", "up", 0), ("C", "down", 1)]
    c.executemany("INSERT INTO predictions (date, horizon, symbol, direction, confidence, "
                  "entry_price, correct, base_rate) VALUES ('2026-09-01', 'intraday', ?, ?, 0.6, "
                  "100, ?, 0.5)", rows)
    c.commit()
    out = SB.by_direction(c, "intraday", datetime(2026, 9, 25, 11, 0), prices={})
    up, down = out["up"][1], out["down"][1]
    assert (up["Accuracy"], up["Right"]) == (0.5, "1/2")
    assert (down["Accuracy"], down["Right"]) == (1.0, "1/1")


def test_longterm_today_uses_yesterdays_close(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    c.executemany("INSERT INTO predictions (date, horizon, symbol, direction, confidence, "
                  "entry_price, horizon_days) VALUES ('2026-09-24', 'longterm', ?, ?, 0.6, 100, ?)",
                  [("A", "up", M.HORIZON), ("B", "up", M.HORIZON), ("C", "down", M.HORIZON)])
    c.commit()
    prices, prev = {"A": 102.0, "B": 99.0, "C": 98.0, "D": 101.0}, {s: 100.0 for s in "ABCD"}
    out = SB.by_direction(c, "longterm", datetime(2026, 9, 25, 11, 0), prices, prev)
    assert out["up"][0]["Right"] == "1/2" and out["up"][0]["Random"] == 0.5
    assert out["down"][0]["Right"] == "1/1"


def test_close_model_accuracy_is_separate(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    c.executemany("INSERT INTO predictions (date, horizon, symbol, direction, confidence, "
                  "entry_price, correct, base_rate) VALUES ('2026-09-24', ?, 'A', 'up', 0.6, 100, ?, 0.5)",
                  [("intraday", 1), ("intraday_close", 0)])
    c.commit()
    now = datetime(2026, 9, 25, 11, 0)
    assert SB.by_direction(c, "intraday", now, prices={})["up"][1]["Right"] == "1/1"
    close = SB.by_direction(c, "intraday_close", now, prices={})["up"]
    assert close[1]["Right"] == "0/1" and "close" in close[0]["What"]
