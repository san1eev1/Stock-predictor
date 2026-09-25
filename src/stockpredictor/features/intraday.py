"""Intraday features at 9:45 for each stock-day, plus the 9:45 -> 12:30 target.

Everything uses only information available at 9:45: the first 30 minutes of
today and daily data up to yesterday's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.data import intraday as I

# Daily (previous-day) context taken from the long-term feature table.
DAILY_CONTEXT = ["ret_5", "ret_21", "ret_63", "rsi_14", "dist_ma50", "dist_ma200",
                 "dist_52w_high", "vol_21", "bb_pctb", "vol_ratio_20_120", "beta_252",
                 "mkt_ret_63", "mkt_vol_21", "vix", "vix_pct_252", "wk_streak"]
RANKED = ["gap", "r30", "rel_r30", "vwap_dev", "vol30_adv", "pos30", "ret_5"]
PRICE_COLS = ["open", "h30", "l30", "c30", "vwap30", "high_after", "low_after", "px_1515", "close",
              I.EXIT_COL]
EXCLUDE = {"symbol", "date", "source", "target", "target_ret", "prev_date", "prev_close", "adv20",
           "fb_weight",
           *PRICE_COLS,
           "v30", *I.LEVEL_COLS}


def adjust_splits(summ: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Angel One prices are raw; scale them to match split-adjusted daily prices."""
    splits = actions[actions["kind"] == "split"]
    if splits.empty or "source" not in summ:
        return summ
    summ = summ.copy()
    raw = summ["source"] == "angelone"
    for sym, g in splits.groupby("symbol"):
        rows = raw & (summ["symbol"] == sym)
        for d, v in zip(pd.to_datetime(g["date"]), g["value"]):
            before = rows & (summ["date"] < d)
            summ.loc[before, PRICE_COLS] = summ.loc[before, PRICE_COLS] / v
            summ.loc[before, "v30"] = summ.loc[before, "v30"] * v
    return summ


def _daily_context(daily: pd.DataFrame, lt_feats: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, date): yesterday-close context available before today's open."""
    d = daily.sort_values(["symbol", "date"]).copy()
    g = d.groupby("symbol")
    tr = pd.concat([d["high"] - d["low"], (d["high"] - g["close"].shift()).abs(),
                    (d["low"] - g["close"].shift()).abs()], axis=1).max(axis=1)
    d["atr_pct"] = (tr / d["close"]).groupby(d["symbol"]).transform(
        lambda x: x.rolling(14, min_periods=10).mean())
    d["ret_1"] = g["close"].pct_change()
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["d_body"] = (d["close"] - d["open"]) / rng
    d["d_upper"] = (d["high"] - d[["open", "close"]].max(axis=1)) / rng
    d["d_lower"] = (d[["open", "close"]].min(axis=1) - d["low"]) / rng
    d["adv20"] = g["volume"].transform(lambda x: x.replace(0, np.nan).rolling(20, min_periods=10).mean())
    ctx = d[["symbol", "date", "close", "atr_pct", "ret_1", "d_body", "d_upper", "d_lower", "adv20"]]
    ctx = ctx.rename(columns={"close": "prev_close"})
    cols = [c for c in DAILY_CONTEXT if c in lt_feats]
    return ctx.merge(lt_feats[["symbol", "date", *cols]], on=["symbol", "date"], how="left")


def build(summ: pd.DataFrame, daily: pd.DataFrame, lt_feats: pd.DataFrame,
          actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for every summary row that has a previous trading day in `daily`."""
    if summ.empty:
        return summ
    s = summ.copy()
    s["date"] = pd.to_datetime(s["date"])
    if actions is not None:
        s = adjust_splits(s, actions)
    ctx = _daily_context(daily, lt_feats).rename(columns={"date": "prev_date"})
    s = s.sort_values("date")
    ctx = ctx.sort_values("prev_date")
    # Previous trading day's context: last daily row strictly before today.
    s = pd.merge_asof(s, ctx, left_on="date", right_on="prev_date", by="symbol",
                      allow_exact_matches=False)
    s = s.dropna(subset=["prev_close"])

    s["gap"] = s["open"] / s["prev_close"] - 1
    s["r30"] = s["c30"] / s["open"] - 1
    s["r_prev"] = s["c30"] / s["prev_close"] - 1
    s["range30_atr"] = (s["h30"] - s["l30"]) / s["prev_close"] / s["atr_pct"]
    s["gap_atr"] = s["gap"] / s["atr_pct"]
    s["r30_atr"] = s["r30"] / s["atr_pct"]
    s["pos30"] = ((s["c30"] - s["l30"]) / (s["h30"] - s["l30"]).replace(0, np.nan))
    s["vwap_dev"] = s["c30"] / s["vwap30"] - 1
    s["vol30_adv"] = s["v30"] / s["adv20"]
    s = s.sort_values(["symbol", "date"])
    s["vol30_rel"] = s["v30"] / s.groupby("symbol")["v30"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean())

    by_day = s.groupby("date")
    s["mkt_r30"] = by_day["r30"].transform("median")
    s["mkt_gap"] = by_day["gap"].transform("median")
    s["breadth30"] = by_day["r30"].transform(lambda x: (x > 0).mean())
    s["rel_r30"] = s["r30"] - s["mkt_r30"]
    s["rel_gap"] = s["gap"] - s["mkt_gap"]
    for col in RANKED:
        s[f"{col}_rank"] = s.groupby("date")[col].rank(pct=True)

    s["target_ret"] = s[I.EXIT_COL] / s["c30"] - 1          # trades are squared off at 12:30
    s["target"] = s.groupby("date")["target_ret"].rank(pct=True)
    return s.sort_values(["date", "symbol"]).reset_index(drop=True)


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in EXCLUDE]
