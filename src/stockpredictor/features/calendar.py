"""Market calendar features: F&O expiry days and ex-dividend days.

Expiry days move stocks differently (index rebalancing of positions, pinning), and an
ex-dividend day opens lower by the dividend - not a real signal. NSE's F&O expiry was the
last Thursday of the month (weekly: every Thursday) until August 2025 and is the last
Tuesday (weekly: Tuesday) from September 2025. A holiday moves it to the trading day before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TUESDAY_FROM = pd.Timestamp("2025-09-01")


def _expiry_weekday(day: pd.Timestamp) -> int:
    return 1 if day >= TUESDAY_FROM else 3            # Tuesday / Thursday


def _on_or_before(day: pd.Timestamp, trading: pd.DatetimeIndex) -> pd.Timestamp:
    if day > trading[-1]:                 # future: holidays not known yet
        return day
    i = trading.searchsorted(day, side="right") - 1
    return trading[i] if i >= 0 else day


def expiry_days(trading_days) -> tuple[set, set]:
    """(monthly, weekly) expiry dates among `trading_days`."""
    trading = pd.DatetimeIndex(sorted(set(pd.to_datetime(trading_days))))
    if trading.empty:
        return set(), set()
    monthly, weekly = set(), set()
    for month in pd.period_range(trading[0], trading[-1] + pd.Timedelta(days=40), freq="M"):
        last = month.end_time.normalize()
        wd = _expiry_weekday(last)
        day = last - pd.Timedelta(days=(last.weekday() - wd) % 7)
        monthly.add(_on_or_before(day, trading))
    for week_start in pd.date_range(trading[0] - pd.Timedelta(days=7), trading[-1], freq="W-MON"):
        day = week_start + pd.Timedelta(days=_expiry_weekday(week_start))
        if trading[0] <= day <= trading[-1] + pd.Timedelta(days=7):
            weekly.add(_on_or_before(day, trading))
    return monthly, weekly


def add_calendar(s: pd.DataFrame, trading_days, actions: pd.DataFrame | None = None,
                 price_col: str = "prev_close") -> pd.DataFrame:
    """Adds: day_of_week, is_expiry (monthly), is_weekly_expiry, days_to_expiry (business
    days to the next monthly expiry), exdiv_today and div_yield_today."""
    s = s.copy()
    days = sorted(set(pd.to_datetime(trading_days)) | set(s["date"]))
    monthly, weekly = expiry_days(days)
    d = s["date"]
    s["day_of_week"] = d.dt.weekday
    s["is_expiry"] = d.isin(monthly).astype(float)
    s["is_weekly_expiry"] = d.isin(weekly).astype(float)
    ms = np.array(sorted(monthly), dtype="datetime64[D]")
    dd = d.to_numpy().astype("datetime64[D]")
    nxt = ms[np.minimum(np.searchsorted(ms, dd), len(ms) - 1)] if len(ms) else dd
    s["days_to_expiry"] = np.where(nxt >= dd, np.busday_count(dd, nxt), np.nan)
    s["exdiv_today"], s["div_yield_today"] = 0.0, 0.0
    if actions is not None and not actions.empty and "kind" in actions:
        div = actions[actions["kind"] == "dividend"][["symbol", "date", "value"]].copy()
        div["date"] = pd.to_datetime(div["date"])
        div = div.groupby(["symbol", "date"], as_index=False)["value"].sum()
        m = s[["symbol", "date"]].merge(div, on=["symbol", "date"], how="left")["value"].to_numpy()
        has = ~np.isnan(m)
        s["exdiv_today"] = has.astype(float)
        if price_col in s:
            s["div_yield_today"] = np.where(has, m / s[price_col].to_numpy(float), 0.0)
    return s
