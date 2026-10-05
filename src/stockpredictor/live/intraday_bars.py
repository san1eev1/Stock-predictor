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
        import time as _time

        # Two passes: stocks Angel One refuses (rate limit) get a second try after a pause;
        # whatever is still missing comes from Yahoo (without the 1-minute fine features).
        todo = list(symbols)
        for pass_ in (1, 2):
            if pass_ == 2:
                todo = [s for s in symbols if s not in out]
                if not todo:
                    break
                _time.sleep(10)
            _first30_pass(client, prices_client, todo, today, start, end, out)
    missing = [s for s in symbols if s not in out]
    if missing:
        for s, bars in I.yahoo_bars(missing, period="1d").items():
            bars = bars[bars["ts"].dt.date == today]
            summ = I.summarize_first30(bars)
            if summ:
                out[s] = summ
    return out


def _first30_pass(client, prices_client, symbols, today, start, end, out) -> None:
    """One pass over `symbols` (Angel One 1-minute bars); stops after 5 refusals in a row."""
    refused = 0
    for s in symbols:
        if refused >= 5:              # Angel One is throttling us: Yahoo for the rest
            log.warning("first30: Angel One refusing requests; Yahoo for the remaining stocks")
            break
        token = prices_client._tokens.get(s)
        if not token:
            continue
        try:
            # 1-minute bars: the same opening summary plus the fine 1/3-minute features
            # live picks can't wait: one quick retry, then Yahoo fills the gaps
            bars = I.angel_bars(client, token, today, today, interval="ONE_MINUTE",
                                waits=(0, 1.5))
            bars = bars[(bars["ts"] >= start) & (bars["ts"] < end)]
            summ = I.summarize_first30(bars)
            if summ:
                from stockpredictor.data.fine import summarize_fine

                out[s] = {**summ, **(summarize_fine(bars) or {})}
            refused = 0
        except Exception as exc:
            refused += 1
            log.warning("first30 %s: %s", s, exc)
