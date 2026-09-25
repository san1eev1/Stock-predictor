"""Today's first 30 minutes (9:15-9:45) for many stocks: Angel One or Yahoo."""

from __future__ import annotations

import logging
from datetime import date, datetime

from stockpredictor.data import intraday as I

log = logging.getLogger(__name__)


def first30(prices_client, symbols: list[str], today: date) -> dict[str, dict]:
    """{symbol: {open, h30, l30, c30, v30, vwap30}} using the LivePrices source."""
    out: dict[str, dict] = {}
    try:
        client = prices_client._angel_client()
    except Exception as exc:
        log.warning("Angel One unavailable (%s); using Yahoo", exc)
        client = None
    if client is not None:
        start = datetime(today.year, today.month, today.day, 9, 15)
        end = datetime(today.year, today.month, today.day, 9, 45)
        for s in symbols:
            token = prices_client._tokens.get(s)
            if not token:
                continue
            try:
                # 1-minute bars: the same opening summary plus the fine 1/3-minute features
                bars = I.angel_bars(client, token, today, today, interval="ONE_MINUTE")
                bars = bars[(bars["ts"] >= start) & (bars["ts"] < end)]
                summ = I.summarize_first30(bars)
                if summ:
                    from stockpredictor.data.fine import summarize_fine

                    out[s] = {**summ, **(summarize_fine(bars) or {})}
            except Exception as exc:
                log.warning("first30 %s: %s", s, exc)
    missing = [s for s in symbols if s not in out]
    if missing:
        for s, bars in I.yahoo_bars(missing, period="1d").items():
            bars = bars[bars["ts"].dt.date == today]
            summ = I.summarize_first30(bars)
            if summ:
                out[s] = summ
    return out
