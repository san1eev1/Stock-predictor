"""Angel One SmartAPI client — MARKET DATA ONLY.

This module deliberately exposes no order placement, modification or
cancellation. It only logs in and reads prices / candles.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime

import pyotp
import requests

from stockpredictor.config import AngelCredentials

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)

INTERVALS = {
    "ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "TEN_MINUTE",
    "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY",
}

# SmartAPI historical data allows a limited number of days per request.
MAX_DAYS_PER_REQUEST = {
    "ONE_MINUTE": 30, "THREE_MINUTE": 60, "FIVE_MINUTE": 100, "TEN_MINUTE": 100,
    "FIFTEEN_MINUTE": 200, "THIRTY_MINUTE": 200, "ONE_HOUR": 400, "ONE_DAY": 2000,
}


def _quiet_smartapi() -> None:
    """SmartAPI logs every failed request with its headers (API key, session token) to the
    terminal and to logs/<date>/app.log. Turn that off; our own code reports errors."""
    import logging

    import logzero

    logzero.logfile(None)
    logzero.loglevel(logging.CRITICAL)


class AngelOneError(RuntimeError):
    pass


def fetch_nse_equity_tokens() -> dict[str, str]:
    """Map NSE cash symbols (e.g. RELIANCE) to Angel One instrument tokens."""
    resp = requests.get(SCRIP_MASTER_URL, timeout=60)
    resp.raise_for_status()
    tokens = {}
    for item in resp.json():
        if item.get("exch_seg") == "NSE" and item.get("symbol", "").endswith("-EQ"):
            tokens[item["symbol"][:-3]] = item["token"]
    return tokens


def save_tokens(conn: sqlite3.Connection, tokens: dict[str, str]) -> int:
    cur = conn.executemany(
        "UPDATE stocks SET angel_token = ? WHERE symbol = ?",
        [(tok, sym) for sym, tok in tokens.items()],
    )
    conn.commit()
    return cur.rowcount


class AngelDataClient:
    def __init__(self, creds: AngelCredentials):
        if not creds.is_complete:
            raise AngelOneError(
                "Angel One credentials missing — fill ANGEL_* values in your .env file.")
        self._creds = creds
        self._api = None

    def login(self) -> None:
        from SmartApi import SmartConnect  # imported lazily: optional until keys exist

        api = SmartConnect(api_key=self._creds.api_key)
        _quiet_smartapi()
        totp = pyotp.TOTP(self._creds.totp_secret).now()
        resp = api.generateSession(self._creds.client_code, self._creds.pin, totp)
        if not resp or not resp.get("status"):
            raise AngelOneError(f"Login failed: {resp and resp.get('message')}")
        self._api = api

    def _require_login(self):
        if self._api is None:
            self.login()
        return self._api

    def ltp(self, symbol: str, token: str) -> float:
        resp = self._require_login().ltpData("NSE", f"{symbol}-EQ", token)
        if not resp or not resp.get("status"):
            raise AngelOneError(f"LTP failed for {symbol}: {resp and resp.get('message')}")
        return float(resp["data"]["ltp"])

    def ltp_many(self, tokens: dict[str, str]) -> dict[str, float]:
        """Live prices for many stocks: {symbol: token} -> {symbol: ltp} (50 per request)."""
        by_token = {t: s for s, t in tokens.items()}
        out = {}
        items = list(by_token)
        for i in range(0, len(items), 50):
            resp = self._require_login().getMarketData("LTP", {"NSE": items[i:i + 50]})
            if not resp or not resp.get("status"):
                raise AngelOneError(f"Market data failed: {resp and resp.get('message')}")
            for row in resp["data"].get("fetched", []):
                sym = by_token.get(str(row.get("symbolToken")))
                if sym and row.get("ltp"):
                    out[sym] = float(row["ltp"])
            time.sleep(0.4)
        return out

    def candles(self, token: str, interval: str,
                start: datetime, end: datetime) -> list[tuple]:
        """Return [(ts, open, high, low, close, volume), ...] for one request window."""
        if interval not in INTERVALS:
            raise ValueError(f"Unknown interval {interval}")
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval,
            "fromdate": start.strftime("%Y-%m-%d %H:%M"),
            "todate": end.strftime("%Y-%m-%d %H:%M"),
        }
        # Angel One throttles bursts (answers with an unparsable page): back off and retry.
        for wait in (0, 2, 6, 15):
            time.sleep(wait)
            try:
                resp = self._require_login().getCandleData(params)
            except Exception as exc:
                resp, err = None, exc
            else:
                err = None
            time.sleep(0.5)  # stay under SmartAPI rate limits
            if resp and resp.get("status"):
                return [tuple(row) for row in (resp.get("data") or [])]
        raise AngelOneError(f"Candle fetch failed: {err or (resp and resp.get('message'))}")
