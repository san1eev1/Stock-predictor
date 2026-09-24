from stockpredictor.config import load_settings


def test_load_settings_from_env_file(tmp_path, monkeypatch):
    for key in ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PIN", "ANGEL_TOTP_SECRET",
                "PAPER_CAPITAL_INTRADAY", "DB_PATH"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("ANGEL_API_KEY=abc\nPAPER_CAPITAL_INTRADAY=50000\n")
    s = load_settings(env)
    assert s.angel.api_key == "abc"
    assert not s.angel.is_complete
    assert s.paper_capital_intraday == 50000
    assert s.paper_capital_longterm == 100000
