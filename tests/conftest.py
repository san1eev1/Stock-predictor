import pytest

from stockpredictor import config


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Tests never read the app's own local files (e.g. prices the Mac fetched itself)."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")


@pytest.fixture(autouse=True)
def mac_training(monkeypatch):
    """The training paths are tested with Mac training on (the app default is off)."""
    monkeypatch.setattr(config, "MAC_TRAINING", True)


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    """Tests never read the real trained models / learned trading rules."""
    from stockpredictor.models import intraday as MI

    root = tmp_path / "models"
    targets = tuple(MI.Target(t.horizon, t.exit_col, root / t.model_dir.name, t.label)
                    for t in MI.TARGETS)
    monkeypatch.setattr(MI, "TRADE", targets[0])
    monkeypatch.setattr(MI, "CLOSE", targets[1])
    monkeypatch.setattr(MI, "TARGETS", targets)
    monkeypatch.setattr(MI, "MODEL_DIR", targets[0].model_dir)
    monkeypatch.setattr(MI, "ROOT", root)
    monkeypatch.setattr(MI, "LOCAL_MODELS_DIR", tmp_path / "mac-models")
