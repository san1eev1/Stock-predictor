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


def test_tuning_round_is_checked_by_paper_trading(tmp_path, monkeypatch):
    target = MI.Target("intraday", "px_1230", tmp_path / "m", "until 12:30")
    days = pd.bdate_range("2026-01-01", periods=MI.MIN_TRAIN_DAYS + 20)
    monkeypatch.setattr(T, "intraday_feats", lambda ctx, d, t: pd.DataFrame({"date": days}))
    monkeypatch.setattr(T, "peer_walk_forward", lambda *a: None)
    monkeypatch.setattr(T, "candidates", lambda cur, n, seed, grid: [{"a": 1}, {"a": 2}])
    monkeypatch.setattr(T, "evaluate_intraday", lambda f, p, *a, **k: {"ic": 0.01 * p["a"]})
    saved = []
    monkeypatch.setattr(T, "_save_params", lambda d, p: saved.append(p))
    monkeypatch.setattr(MI.IntradayModel, "train", lambda *a, **k: type("M", (), {"save": lambda s, d: None})())
    paper = {1: 200, 2: 100}          # the new settings (a=2) paper-trade worse

    def check(f, t, params, rules=None):
        return {"period": "p", "days": 120, "avg_day_pnl": paper[params["a"]], "win_days": 0.5,
                "accuracy": 0.55, "random": 0.5, "ic": 0.01}

    monkeypatch.setattr(T, "paper_check", check)
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    r = T.tune_intraday("ctx", tmp_path, conn, log_all=False, target=target, check_days=10**6)
    assert not r["adopted"] and not saved and r["paper"]["avg_day_pnl"] == 200
    paper[2] = 300                    # now they paper-trade better: adopted
    r = T.tune_intraday("ctx", tmp_path, conn, log_all=False, target=target, check_days=10**6)
    assert r["adopted"] and saved == [{"a": 2}]
    rows = conn.execute("SELECT avg_day_pnl, adopted FROM tune_checks ORDER BY id").fetchall()
    assert [tuple(x) for x in rows] == [(200, 0), (300, 1)]


def test_rules_tuned_by_profit(tmp_path, monkeypatch):
    from stockpredictor.backtest import intraday as B

    target = MI.Target("intraday", "px_1230", tmp_path / "m", "until 12:30")
    days = pd.bdate_range("2026-01-01", periods=120)
    scores = pd.DataFrame({"symbol": "A", "date": days, "score": 1.0})
    monkeypatch.setattr(MI, "walk_forward", lambda f, params, last_days: scores)
    from stockpredictor.data import intraday as I
    feats = pd.DataFrame(columns=["symbol", "date", "c30", "px_1230", *I.LEVEL_COLS])

    def fake_sim(sc, summ, r, capital, exit_col, exit_minutes):
        # 2 buys, no sells, skipping weak days earns most; everything else loses costs
        pnl = 100.0 if (r.n_long, r.n_short, r.skip_quantile) == (2, 0, 0.5) else -300.0
        return {"days": pd.DataFrame({"date": days, "pnl": pnl})}

    monkeypatch.setattr(B, "simulate", fake_sim)
    rep = T.tune_rules(feats, target, B.IntradayRules(), max_n=5)
    assert rep["adopted"] and rep["rules"]["n_long"] == 2 and rep["rules"]["n_short"] == 0
    assert rep["rules"]["skip_quantile"] == 0.5 and rep["day_profit_recent"] == 100
    # the learned rules apply, capped by the Settings maximum
    r = T.rules_for(target, B.IntradayRules(n_long=1, n_short=5))
    assert (r.n_long, r.n_short, r.skip_quantile) == (1, 0, 0.5)
