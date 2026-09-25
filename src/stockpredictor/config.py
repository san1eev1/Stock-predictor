"""Settings loaded from the local .env file (never committed)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
load_dotenv(PROJECT_ROOT / ".env")

# Long-term and news models train on GitHub Actions on all history (2005 onwards) and are
# published to the `models` git branch; the Mac downloads them into trained-models/ and keeps
# only recent prices. CLOUD_TRAINING=0 in .env trains everything on the Mac instead.
CLOUD_TRAINING = os.getenv("CLOUD_TRAINING", "1") != "0"
# The Mac itself trains nothing (keeps it cool): live prices, picks and paper trading only;
# models come from GitHub. MAC_TRAINING=1 brings back background tuning on the Mac.
MAC_TRAINING = os.getenv("MAC_TRAINING", "0" if CLOUD_TRAINING else "1") != "0"
# All learning runs on GitHub (all history since 2005, twice every evening). MAC_DAILY_TRAINING=1
# adds one retrain a day on the Mac at 16:00, in its own short-lived process.
MAC_DAILY_TRAINING = os.getenv("MAC_DAILY_TRAINING", "0") != "0"
# The live monitor and dashboard keep only this many recent years in memory (predictions
# need recent data; training loads everything in its own short-lived process). 0 = all.
LIVE_CONTEXT_YEARS = 0 if MAC_TRAINING else int(os.getenv("LIVE_CONTEXT_YEARS", "3"))
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
SHARED_MODELS_DIR = PROJECT_ROOT / ("trained-models" if CLOUD_TRAINING else "models")
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"            # intraday model (uses Angel One data)
# Years of daily prices kept on the Mac (features need ~1 year of warm-up). 0 = all.
LOCAL_HISTORY_YEARS = 0 if (IN_GITHUB_ACTIONS or not CLOUD_TRAINING) \
    else int(os.getenv("LOCAL_HISTORY_YEARS", "4"))


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
