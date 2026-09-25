import threading

from stockpredictor.live import background as BG


def test_background_trainer_tunes_and_learns_live(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(BG.BackgroundTrainer, "_ctx", lambda self: "ctx")
    monkeypatch.setattr(BG.T, "tune_intraday", lambda ctx, d, conn, **k:
                        calls.append(("tune", k["log_all"])) or
                        {"ic": 0.02, "previous_ic": 0.01, "adopted": True})
    monkeypatch.setattr(BG.T, "retrain_intraday", lambda ctx, d, conn, force:
                        calls.append(("live", force)) or None)
    monkeypatch.setattr(BG.config, "CLOUD_TRAINING", True)
    bg = BG.BackgroundTrainer(tmp_path / "t.db", tmp_path, pause_min=0.001)
    from stockpredictor import db
    db.init_db(tmp_path / "t.db")
    bg._tune(db.connect(tmp_path / "t.db"))
    assert calls == [("tune", False)] and bg.model_changed.is_set()
    bg._learn_live(db.connect(tmp_path / "t.db"), None)       # no Angel keys: just retrain
    assert calls[-1] == ("live", True)
    assert isinstance(BG.MI.MODEL_LOCK, type(threading.RLock()))
