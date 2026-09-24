from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "stockpredictor" / "app" / "main.py"


def test_app_starts_without_errors(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Long-term picks"
