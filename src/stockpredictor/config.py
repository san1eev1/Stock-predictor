"""Settings loaded from the local .env file (never committed)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class AngelCredentials:
    api_key: str
    client_code: str
    pin: str
    totp_secret: str

    @property
    def is_complete(self) -> bool:
        return all([self.api_key, self.client_code, self.pin, self.totp_secret])


@dataclass(frozen=True)
class Settings:
    db_path: Path
    angel: AngelCredentials
    paper_capital_intraday: float
    paper_capital_longterm: float


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env")

    db_path = Path(os.getenv("DB_PATH", DATA_DIR / "stockpredictor.db"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    return Settings(
        db_path=db_path,
        angel=AngelCredentials(
            api_key=os.getenv("ANGEL_API_KEY", ""),
            client_code=os.getenv("ANGEL_CLIENT_CODE", ""),
            pin=os.getenv("ANGEL_PIN", ""),
            totp_secret=os.getenv("ANGEL_TOTP_SECRET", ""),
        ),
        paper_capital_intraday=float(os.getenv("PAPER_CAPITAL_INTRADAY", "100000")),
        paper_capital_longterm=float(os.getenv("PAPER_CAPITAL_LONGTERM", "100000")),
    )
