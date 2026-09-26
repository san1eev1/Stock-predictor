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
                 "mkt_ret_63", "mkt_vol_21", "vix", "vix_pct_252", "wk_streak",
                 "deliv_pct_20", "deliv_pct_rel", "results_reaction",
                 # overnight cues: the previous day's row holds the last close before 9:15
                 "g_sp500_ret1_asof", "g_nasdaq_ret1_asof", "g_usvix_ret1_asof",
                 "g_nikkei_ret1_asof", "g_hangseng_ret1_asof", "g_usdinr_ret5_asof",
                 "g_crude_ret5_asof"]
RANKED = ["gap", "r30", "rel_r30", "vwap_dev", "vol30_adv", "pos30", "ret_5"]
PRICE_COLS = ["open", "h30", "l30", "c30", "vwap30", "high_after", "low_after", "px_1515", "close",
              I.EXIT_COL, "iep"]
EXCLUDE = {"symbol", "date", "source", "target", "target_ret", "prev_date", "prev_close", "adv20",
           "target_trade", "trade_long_ret", "trade_short_ret", "industry",
           "po_qty", "buy_qty", "sell_qty",
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


BASIS_TOL = 0.02      # intraday vs official close differ a little; beyond this: other basis


def align_to_daily(summ: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Put intraday prices on the daily history's price basis. The daily history is adjusted
    for splits, bonuses and demergers (data/clean.py); some intraday rows are not, which made
    gap and 'vs yesterday' features wrong for those stock-days. The factor comes from the
    PREVIOUS day (today's close is not known at 9:45: no look-ahead), which is the same
    factor except on the event day itself. Ratios within a day are unchanged."""
    d = daily[["symbol", "date", "close"]].rename(columns={"close": "_dclose"})
    s = summ.merge(d, on=["symbol", "date"], how="left").sort_values(["symbol", "date"])
    k = s["_dclose"] / s["close"]
    k = k.where(k.isna() | ((k - 1).abs() > BASIS_TOL), 1.0)
    k = k.groupby(s["symbol"]).shift(1)
    k = k.groupby(s["symbol"]).ffill().fillna(1.0)
    cols = [c for c in PRICE_COLS if c in s]
    s[cols] = s[cols].mul(k, axis=0)
    if "v30" in s:
        s["v30"] = s["v30"] / k
    return s.drop(columns="_dclose").reset_index(drop=True)


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


TRADE_LABEL_RULES = (1.0, 2.0)      # stop-loss %, target % behind the trade-outcome label


ODD_COLS = ["mkt_r30", "mkt_gap", "breadth30", "disp30", "vix"]
ODD_WINDOW = 40          # trading days of "normal" to compare today with


def day_oddness(s: pd.DataFrame) -> pd.DataFrame:
    """'Is today unusual?': per date, the largest |z-score| of the market's 9:45 mood (average
    move and gap, breadth, dispersion, VIX) against the previous ODD_WINDOW days (only days
    before: no look-ahead). Above ~3 the model has rarely seen a morning like it."""
    cols = [c for c in ODD_COLS if c in s]
    day = s.groupby("date")[cols].first().sort_index()
    ref = day.shift(1).rolling(ODD_WINDOW, min_periods=20)
    z = ((day - ref.mean()) / ref.std().replace(0, np.nan)).abs()
    return z.max(axis=1).rename("day_oddness").reset_index()


def build(summ: pd.DataFrame, daily: pd.DataFrame, lt_feats: pd.DataFrame,
          actions: pd.DataFrame | None = None, exit_col: str = I.EXIT_COL,
          sectors: dict | None = None, earnings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for every summary row that has a previous trading day in `daily`.
    The target is the 9:45 -> `exit_col` move (12:30 exit by default, or "close")."""
    if summ.empty:
        return summ
    s = summ.copy()
    s["date"] = pd.to_datetime(s["date"])
    if actions is not None:
        s = adjust_splits(s, actions)
    s = align_to_daily(s, daily)
    ctx = _daily_context(daily, lt_feats).rename(columns={"date": "prev_date"})
    s = s.sort_values("date")
    ctx = ctx.sort_values("prev_date")
    # Previous trading day's context: last daily row strictly before today.
    s = pd.merge_asof(s, ctx, left_on="date", right_on="prev_date", by="symbol",
                      allow_exact_matches=False)
    s = s.dropna(subset=["prev_close"])

    from stockpredictor.data.preopen import add_features as preopen_features

    s = preopen_features(s)                   # pre-open auction (empty until collected)
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
    s["disp30"] = by_day["r30"].transform("std")          # how spread out the moves are
    s = s.merge(day_oddness(s), on="date", how="left")
    for col in RANKED:
        s[f"{col}_rank"] = s.groupby("date")[col].rank(pct=True)

    if sectors:                           # the stock vs its own sector in the first 30 min
        s["industry"] = s["symbol"].map(sectors).fillna("other")
        by_sec = s.groupby(["date", "industry"])
        s["sec_r30"] = by_sec["r30"].transform("median")
        s["sec_gap"] = by_sec["gap"].transform("median")
        s["sec_breadth30"] = by_sec["r30"].transform(lambda x: (x > 0).mean())
        s["rel_sec_r30"] = s["r30"] - s["sec_r30"]
        s["rel_sec_gap"] = s["gap"] - s["sec_gap"]
        s["sec_vs_mkt_r30"] = s["sec_r30"] - s["mkt_r30"]
    from stockpredictor.features.calendar import add_calendar

    s = add_calendar(s, daily["date"].unique(), actions)
    if earnings is not None and not earnings.empty:        # results-day moves
        from stockpredictor.data.earnings import add_features as earnings_features

        s = earnings_features(s, earnings)

    s["target_ret"] = s[exit_col] / s["c30"] - 1            # 12:30 square-off, or the close
    s["target"] = s.groupby("date")["target_ret"].rank(pct=True)
    # Trade-outcome label: what a buy and a sell at 9:45 actually return with a stop-loss
    # and target (whichever is hit first, else the square-off). Used when params.label ==
    # "trade": the model then learns what the trades earn, not only the raw move.
    from stockpredictor.backtest.intraday import trade_exits

    minutes = I.EXIT_MINUTES if exit_col == I.EXIT_COL else I.WINDOW_MINUTES
    has = s[exit_col].notna()
    sl, tp = TRADE_LABEL_RULES
    s["trade_long_ret"] = np.nan
    s["trade_short_ret"] = np.nan
    if has.any():
        h = s[has]
        s.loc[has, "trade_long_ret"] = trade_exits(h, "long", sl, tp, exit_col, minutes) / h["c30"] - 1
        s.loc[has, "trade_short_ret"] = 1 - trade_exits(h, "short", sl, tp, exit_col, minutes) / h["c30"]
    s["target_trade"] = (s["trade_long_ret"] - s["trade_short_ret"]).groupby(s["date"]).rank(pct=True)
    return s.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_for(summ: pd.DataFrame, ctx, store_dir, exit_col: str = I.EXIT_COL,
              daily: pd.DataFrame | None = None,
              live_preopen: pd.DataFrame | None = None) -> pd.DataFrame:
    """`build` with everything from the market context and data store (corporate actions,
    sectors, results dates) - the one way training and live picks build features."""
    from stockpredictor import store
    from stockpredictor.data import earnings as ER
    from stockpredictor.data import fine, preopen

    summ = preopen.attach(fine.attach(summ), store_dir, live_preopen)
    return build(summ, ctx.daily if daily is None else daily, ctx.feats,
                 store.load_actions(store_dir), exit_col=exit_col,
                 sectors=dict(zip(ctx.universe["symbol"], ctx.universe["industry"])),
                 earnings=ER.load(store_dir))


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in EXCLUDE]
