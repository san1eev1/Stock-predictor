import threading

from stockpredictor.live import background as BG


def test_background_trainer_tunes_and_learns_live(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(BG.BackgroundTrainer, "_ctx", lambda self: "ctx")
    monkeypatch.setattr(BG.T, "tune_intraday", lambda ctx, d, conn, **k:
                        calls.append(("tune", k["target"].horizon)) or
                        {"ic": 0.02, "previous_ic": 0.01, "adopted": True})
    monkeypatch.setattr(BG.T, "retrain_intraday", lambda ctx, d, conn, force, target:
                        calls.append(("live", target.horizon)) or None)
    monkeypatch.setattr(BG.config, "CLOUD_TRAINING", True)
    bg = BG.BackgroundTrainer(tmp_path / "t.db", tmp_path, pause_min=0.001)
    from stockpredictor import db
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    bg._tune(conn)
    bg._tune(conn)                                            # alternates 12:30 / close models
    assert calls == [("tune", "intraday"), ("tune", "intraday_close")]
    assert bg.model_changed.is_set()
    bg._learn_live(conn, None)                                # no Angel keys: just retrain both
    assert calls[-2:] == [("live", "intraday"), ("live", "intraday_close")]
    assert isinstance(BG.MI.MODEL_LOCK, type(threading.RLock()))
