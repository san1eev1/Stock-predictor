from datetime import datetime

import pandas as pd

from stockpredictor import db
from stockpredictor.live import background as BG
from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T


def test_replay_rounds_learn_from_previous_round(tmp_path, monkeypatch):
    targets = tuple(MI.Target(t.horizon, t.exit_col, tmp_path / t.horizon, t.label)
                    for t in MI.TARGETS)
    monkeypatch.setattr(MI, "TARGETS", targets)
    days = pd.bdate_range("2026-01-01", periods=80)
    feats = pd.DataFrame({"symbol": "A", "date": days, "target": 0.5})
    monkeypatch.setattr(T, "intraday_feats", lambda ctx, d, target: feats)
    seen, pnl = [], iter([100, 150, 180, -50, -60, -70])

    def fake_round(f, target, rules, capital, last_days=T.REPLAY_DAYS):
        seen.append("fb_weight" in f)                       # rounds 2 and 3 get feedback
        judged = pd.DataFrame({"symbol": ["A"], "date": [days[-1]], "correct": [0]})
        return ({"period": "p", "days": 60, "avg_day_pnl": next(pnl), "win_days": 0.5,
                 "accuracy": 0.52, "random": 0.5, "ic": 0.01}, judged)

    monkeypatch.setattr(T, "replay_round", fake_round)
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    out = T.replay_training("ctx", tmp_path, conn)
    assert seen == [False, True, True] * 2
    assert out[targets[0].horizon]["adopted"] and not out[targets[1].horizon]["adopted"]
    assert T.replay_file(targets[0]).exists() and not T.replay_file(targets[1]).exists()
    rows = conn.execute("SELECT horizon, round, adopted FROM replay_runs ORDER BY id").fetchall()
    assert [tuple(r) for r in rows][:3] == [(targets[0].horizon, 1, 0), (targets[0].horizon, 2, 0),
                                           (targets[0].horizon, 3, 1)]
    # the kept replay feedback is used by the live model's training
    fb = T.intraday_feedback(None, feats, targets[0].horizon)
    assert fb["fb_weight"].notna().sum() == 1


def test_replay_runs_once_a_day_after_the_close(tmp_path):
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    bg = BG.BackgroundTrainer(tmp_path / "t.db", tmp_path)
    assert not bg._replay_due(conn, datetime(2026, 9, 25, 14, 0))    # market still open
    assert bg._replay_due(conn, datetime(2026, 9, 25, 15, 45))
    conn.execute("INSERT INTO app_settings (key, value) VALUES ('replay_day', '2026-09-25')")
    assert not bg._replay_due(conn, datetime(2026, 9, 25, 18, 0))    # already done today
    assert bg._replay_due(conn, datetime(2026, 9, 26, 10, 0))        # Saturday
