from datetime import datetime

from stockpredictor import db
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
