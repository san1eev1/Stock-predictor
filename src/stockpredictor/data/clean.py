"""Price-data cleaning: fix the outliers that are data problems, keep real market moves.

1. Bad rows: a one-day jump that fully reverts the next day, usually with zero volume
   and no trading range (e.g. many stocks on 2005-07-28). The row is dropped.
2. Unadjusted corporate events: splits, bonuses and demergers the data source did not
   adjust for. They show up as a big overnight gap (|gap| >= 25%) on an otherwise calm day
   that does not revert. Real crashes look different: exchange price bands limit the
   opening gap (YES Bank 2020 opened -10%, then swung 170% intraday). The history before
   the event is scaled by the gap, as a proper adjustment would, so returns, momentum and
   labels stay continuous. Prices from the event onwards are unchanged.
3. Empty rows: zero volume and no range (stale copies of the previous day). Dropped.
"""

from __future__ import annotations

import pandas as pd

SPIKE = 0.25          # one-day move that counts as a spike if it reverts
REVERT_TOL = 0.10     # ...within this of the price before the spike
BREAK_GAP = 0.25      # opening gap that no normal trading day produces
CALM_RANGE = 0.20     # day's high-low range below this share of the price
PRICE_COLS = ["open", "high", "low", "close", "adj_close"]


def clean_daily(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(cleaned copy, report) for a symbol/date/OHLC(V) frame."""
    df = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    cols = [c for c in PRICE_COLS if c in df]
    df[cols] = df[cols].astype(float)
    if "volume" in df:
        df["volume"] = df["volume"].astype(float)
    report = {"empty_rows": 0, "spikes": [], "breaks": []}

    if "volume" in df:
        empty = (df["volume"] == 0) & (df["high"] == df["low"])
        report["empty_rows"] = int(empty.sum())
        df = df[~empty].reset_index(drop=True)

    # 1. spikes that revert the next day
    prev = df.groupby("symbol")["close"].shift()
    nxt = df.groupby("symbol")["close"].shift(-1)
    r = df["close"] / prev - 1
    spike = (r.abs() > SPIKE) & ((nxt / prev - 1).abs() < REVERT_TOL)
    report["spikes"] = [(s, f"{d:%Y-%m-%d}") for s, d in df.loc[spike, ["symbol", "date"]].values]
    df = df[~spike].reset_index(drop=True)

    # 2. unadjusted corporate events: back-adjust the history before them
    prev = df.groupby("symbol")["close"].shift()
    gap = df["open"] / prev - 1
    rng = (df["high"] - df["low"]) / df["close"]
    brk = (gap.abs() >= BREAK_GAP) & (rng < CALM_RANGE)
    for i in df.index[brk]:
        sym, day, factor = df.at[i, "symbol"], df.at[i, "date"], df.at[i, "open"] / prev[i]
        before = (df["symbol"] == sym) & (df["date"] < day)
        df.loc[before, cols] = df.loc[before, cols] * factor
        if "volume" in df:
            df.loc[before, "volume"] = df.loc[before, "volume"] / factor
        report["breaks"].append((sym, f"{day:%Y-%m-%d}", round(float(factor), 4)))
    return df, report
