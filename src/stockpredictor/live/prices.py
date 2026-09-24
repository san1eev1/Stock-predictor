"""Live prices: Angel One real-time when configured, Yahoo Finance as fallback."""

from __future__ import annotations

import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from stockpredictor.config import Settings

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN, MARKET_CLOSE = time(9, 15), time(15, 30)


def now_ist() -> datetime:
    return datetime.now(IST)


def in_market_hours(ts: datetime | None = None) -> bool:
    ts = ts or now_ist()
    return ts.weekday() < 5 and MARKET_OPEN <= ts.time() <= MARKET_CLOSE


def yahoo_prices(symbols: list[str]) -> tuple[dict[str, float], datetime | None]:
    """Latest 1-minute prices (may lag a few minutes) and the newest bar time."""
    import yfinance as yf

    tickers = [f"{s}.NS" for s in symbols]
    df = yf.download(tickers, period="1d", interval="1m", progress=False,
                     group_by="ticker", auto_adjust=False, threads=True)
    out, newest = {}, None
    for s, t in zip(symbols, tickers):
        try:
            closes = (df[t] if len(tickers) > 1 else df)["Close"].dropna()
        except KeyError:
            continue
        if not closes.empty:
            out[s] = float(closes.iloc[-1])
            ts = closes.index[-1].to_pydatetime()
            newest = ts if newest is None or ts > newest else newest
    return out, newest


class LivePrices:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._angel = None
        self._tokens: dict[str, str] | None = None
        self.source = "yahoo"
        self.last_error: str | None = None

    def _angel_client(self):
        if self._angel is None and self.settings.angel.is_complete:
            from stockpredictor.data.angelone import AngelDataClient, fetch_nse_equity_tokens

            self._angel = AngelDataClient(self.settings.angel)
            self._angel.login()
            self._tokens = fetch_nse_equity_tokens()
        return self._angel

    def get(self, symbols: list[str]) -> dict[str, float]:
        symbols = sorted(set(symbols))
        if not symbols:
            return {}
        try:
            client = self._angel_client()
            if client is not None:
                prices = client.ltp_many({s: self._tokens[s] for s in symbols if s in self._tokens})
                self.source, self.last_error = "angelone", None
                return prices
        except Exception as exc:  # fall back to Yahoo, retry Angel next time
            self.last_error = f"Angel One: {exc}"
            log.warning("Angel One live prices failed, using Yahoo: %s", exc)
            self._angel = None
        prices, _ = yahoo_prices(symbols)
        self.source = "yahoo"
        return prices

    def market_is_live(self) -> bool:
        """Market hours AND Nifty traded today (so NSE holidays are skipped)."""
        if not in_market_hours():
            return False
        try:
            _, newest = yahoo_prices(["RELIANCE"])
            return newest is not None and newest.astimezone(IST).date() == now_ist().date()
        except Exception:
            return True  # can't tell; assume open during market hours
