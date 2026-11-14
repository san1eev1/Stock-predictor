"""Market profiles. The app runs one market per process: `MARKET=us` (env) selects the US
(S&P 500) profile; the default is India (NSE). Market-specific constants elsewhere read
`config.MARKET`, so the long-term pipeline, paper book and news overlay are shared."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time


@dataclass(frozen=True)
class Market:
    name: str                       # "in" / "us"
    label: str                      # "India (NSE)"
    ticker_suffix: str              # Yahoo suffix: ".NS" / ""
    index: str                      # our name of the market index
    vix: str                        # volatility index (crash guard)
    indices: dict = field(default_factory=dict)   # our name -> Yahoo ticker
    global_cues: dict = field(default_factory=dict)
    tz: str = "Asia/Kolkata"
    close: time = time(15, 30)
    circuit_bands: tuple = (0.05, 0.10, 0.20)
    fractional: bool = False        # fractional shares allowed (US brokers)
    currency: str = "₹"
    data_branch: str = "market-data"
    store_dir: str = "market-data"
    model_suffix: str = ""          # models/<name><suffix>
    state_branch: str = "paper-state"
    db_name: str = "stockpredictor.db"
    rss_locale: str = "hl=en-IN&gl=IN&ceid=IN:en"
    query_word: str = "share"       # '"Name" share' / '"Name" stock'
    drop_foreign: bool = True       # drop headlines tagged NYSE:/NASDAQ: (foreign for India)


IN = Market(
    name="in", label="India (NSE)", ticker_suffix=".NS", index="NIFTY50", vix="INDIAVIX",
    indices={"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "INDIAVIX": "^INDIAVIX",
             "NIFTYIT": "^CNXIT", "NIFTYPHARMA": "^CNXPHARMA",
             "SP500": "^GSPC", "NASDAQ": "^IXIC", "USVIX": "^VIX", "NIKKEI": "^N225",
             "HANGSENG": "^HSI", "USDINR": "INR=X", "CRUDE": "CL=F"},
)

US = Market(
    name="us", label="US (S&P 500)", ticker_suffix="", index="SP500", vix="USVIX",
    indices={"SP500": "^GSPC", "NASDAQ": "^IXIC", "USVIX": "^VIX", "DOW": "^DJI",
             "CRUDE": "CL=F", "DXY": "DX-Y.NYB"},
    tz="America/New_York", close=time(16, 0), circuit_bands=(), fractional=True,
    currency="$", data_branch="market-data-us", store_dir="market-data-us",
    model_suffix="_us", state_branch="paper-state-us", db_name="stockpredictor_us.db",
    rss_locale="hl=en-US&gl=US&ceid=US:en", query_word="stock", drop_foreign=False,
)

MARKETS = {"in": IN, "us": US}


def get(name: str | None) -> Market:
    return MARKETS.get((name or "in").lower(), IN)
