import pytest

from stockpredictor import config


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Tests never read the app's own local files (e.g. prices the Mac fetched itself)."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
