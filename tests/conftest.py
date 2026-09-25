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
