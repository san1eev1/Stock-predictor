"""Long-term features: one row per stock per day, using only data up to that day.

Groups: momentum, trend, 52-week range, volatility/risk, technical indicators,
volume/liquidity, relative strength vs Nifty and sector, weekly candle
patterns, market regime, and cross-sectional ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY_DAYS = 200

MARKET_INDEX = "NIFTY50"

RANKED = ["ret_5", "ret_10", "ret_21", "ret_63", "ret_126", "mom_12_1", "dist_52w_high", "vol_63",
          "rs_nifty_63", "rs_sector_63", "vol_ratio_20_120", "rsi_14", "dist_ma20", "pos_5"]


# --- Indicator helpers (per single-stock series) ------------------------------

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _max_drawdown(close: pd.Series, n: int) -> pd.Series:
    peak = close.rolling(n, min_periods=n // 2).max()
    return (close / peak - 1).rolling(n, min_periods=n // 2).min()


def _stock_features(g: pd.DataFrame) -> pd.DataFrame:
    px = g["adj_close"].fillna(g["close"])
    ret = px.pct_change()
    out = pd.DataFrame(index=g.index)

    for n in (1, 5, 10, 21, 63, 126, 252):
        out[f"ret_{n}"] = px / px.shift(n) - 1
    out["mom_12_1"] = px.shift(21) / px.shift(252) - 1   # 12-month momentum skipping last month

    # Short-term (1-week horizon) signals: recent trend, stretch and range.
    out["dist_ma10"] = px / px.rolling(10).mean() - 1
    out["dist_ma20"] = px / px.rolling(20).mean() - 1
    out["vol_5"] = ret.rolling(5).std() * np.sqrt(252)
    out["range_5"] = ((g["high"] - g["low"]) / g["close"]).rolling(5).mean()
    hi5, lo5 = g["high"].rolling(5).max(), g["low"].rolling(5).min()
    out["pos_5"] = (g["close"] - lo5) / (hi5 - lo5).replace(0, np.nan)
    out["up_days_10"] = (ret > 0).rolling(10).mean()

    ma50, ma200 = px.rolling(50).mean(), px.rolling(200).mean()
    out["dist_ma50"] = px / ma50 - 1
    out["dist_ma200"] = px / ma200 - 1
    out["ma50_over_ma200"] = ma50 / ma200 - 1

    out["dist_52w_high"] = g["close"] / g["high"].rolling(252, min_periods=126).max() - 1
    out["dist_52w_low"] = g["close"] / g["low"].rolling(252, min_periods=126).min() - 1

    out["vol_21"] = ret.rolling(21).std() * np.sqrt(252)
    out["vol_63"] = ret.rolling(63).std() * np.sqrt(252)
    out["vol_126"] = ret.rolling(126).std() * np.sqrt(252)
    out["maxdd_126"] = _max_drawdown(px, 126)

    out["rsi_14"] = _rsi(px)
    ema12, ema26 = px.ewm(span=12, adjust=False).mean(), px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / px
    out["adx_14"] = _adx(g["high"], g["low"], g["close"])
    mid, sd = px.rolling(20).mean(), px.rolling(20).std()
    out["bb_pctb"] = (px - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)

    # Yahoo sometimes has placeholder days with volume 0: treat as missing.
    vol = g["volume"].astype(float).replace(0, np.nan)
    out["vol_ratio_20_120"] = (vol.rolling(20, min_periods=15).mean()
                               / vol.rolling(120, min_periods=60).mean())
    out["turnover_log"] = np.log((g["close"] * vol).rolling(20, min_periods=10).mean())
    out["up_volume_ratio_20"] = ((vol * (ret > 0)).rolling(20, min_periods=10).sum()
                                 / vol.rolling(20, min_periods=10).sum())
    out["history_days"] = np.arange(1, len(g) + 1)
    out["_ret1"] = ret
    return out


def _weekly_candles(g: pd.DataFrame) -> pd.DataFrame:
    """Week-to-date candle and previous completed week's candle (causal)."""
    week = g["date"].dt.to_period("W-FRI")
    grp = g.groupby(week)
    w = pd.DataFrame({
        "wk_open": grp["open"].transform("first"),
        "wk_high": grp["high"].cummax(),
        "wk_low": grp["low"].cummin(),
        "wk_close": g["close"],
    }, index=g.index)

    last = w.groupby(week).tail(1)
    prev = last.shift(1)
    prev.index = week.loc[last.index]
    p = prev.reindex(week).set_axis(g.index)

    rng = (w["wk_high"] - w["wk_low"]).replace(0, np.nan)
    body = w["wk_close"] - w["wk_open"]
    out = pd.DataFrame(index=g.index)
    out["wk_body"] = body / rng
    out["wk_upper_wick"] = (w["wk_high"] - w[["wk_open", "wk_close"]].max(axis=1)) / rng
    out["wk_lower_wick"] = (w[["wk_open", "wk_close"]].min(axis=1) - w["wk_low"]) / rng
    out["wk_doji"] = (body.abs() / rng < 0.1).astype(float)
    out["wk_hammer"] = ((out["wk_lower_wick"] > 0.6) & (out["wk_body"].abs() < 0.3)).astype(float)
    prev_body = p["wk_close"] - p["wk_open"]
    out["wk_bull_engulf"] = ((prev_body < 0) & (body > 0) & (w["wk_close"] > p["wk_open"])
                             & (w["wk_open"] < p["wk_close"])).astype(float)
    out["wk_bear_engulf"] = ((prev_body > 0) & (body < 0) & (w["wk_close"] < p["wk_open"])
                             & (w["wk_open"] > p["wk_close"])).astype(float)
    out["wk_inside"] = ((w["wk_high"] < p["wk_high"]) & (w["wk_low"] > p["wk_low"])).astype(float)

    # Consecutive completed up (+) or down (-) weeks, capped at 4.
    up = np.sign(last["wk_close"] - last["wk_open"])
    streak = up.groupby((up != up.shift()).cumsum()).cumcount().add(1).mul(up).clip(-4, 4)
    streak.index = week.loc[last.index]
    out["wk_streak"] = streak.shift(1).reindex(week).set_axis(g.index)
    return out


# --- Market context ------------------------------------------------------------

def _index_returns(indices: pd.DataFrame) -> pd.DataFrame:
    """Per index and date: 63 / 126-day return and daily return."""
    idx = indices.sort_values(["symbol", "date"]).copy()
    px = idx["adj_close"].fillna(idx["close"])
    grp = px.groupby(idx["symbol"])
    idx["ret_5"] = px / grp.shift(5) - 1
    idx["ret_63"] = px / grp.shift(63) - 1
    idx["ret_126"] = px / grp.shift(126) - 1
    idx["ret_1"] = grp.pct_change()
    return idx[["symbol", "date", "ret_1", "ret_5", "ret_63", "ret_126"]]


def market_regime(indices: pd.DataFrame) -> pd.DataFrame:
    nifty = indices[indices["symbol"] == MARKET_INDEX].set_index("date")["close"].sort_index()
    out = pd.DataFrame(index=nifty.index)
    out["mkt_above_ma200"] = (nifty > nifty.rolling(200).mean()).astype(float)
    out["mkt_ret_63"] = nifty / nifty.shift(63) - 1
    out["mkt_vol_21"] = nifty.pct_change().rolling(21).std() * np.sqrt(252)
    vix = indices[indices["symbol"] == "INDIAVIX"].set_index("date")["close"].sort_index()
    if not vix.empty:
        out["vix"] = vix.reindex(out.index).ffill()
        out["vix_pct_252"] = out["vix"].rolling(252, min_periods=60).rank(pct=True)
    return out.reset_index()


# --- Public API ------------------------------------------------------------------

def build_features(daily: pd.DataFrame, indices: pd.DataFrame,
                   universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return features for every (symbol, date) with enough history."""
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)

    parts = []
    for _, g in daily.groupby("symbol", sort=False):
        f = pd.concat([_stock_features(g), _weekly_candles(g)], axis=1)
        parts.append(pd.concat([g[["symbol", "date", "close"]], f], axis=1))
    feats = pd.concat(parts, ignore_index=True)

    # Relative strength and beta vs Nifty 50.
    idx = _index_returns(indices)
    nifty = idx[idx["symbol"] == MARKET_INDEX].drop(columns="symbol").add_prefix("nifty_")
    feats = feats.merge(nifty, left_on="date", right_on="nifty_date", how="left")
    feats["rs_nifty_63"] = feats["ret_63"] - feats["nifty_ret_63"]
    feats["rs_nifty_5"] = feats["ret_5"] - feats["nifty_ret_5"]
    feats["rs_nifty_126"] = feats["ret_126"] - feats["nifty_ret_126"]
    feats["beta_252"] = _rolling_beta(feats)

    # Relative strength vs sector peers: average 3-month return of the other
    # Nifty 100 stocks in the same NSE industry (Yahoo lacks most sector indices).
    industry = pd.Series("", index=feats.index)
    if universe is not None and "industry" in universe:
        industry = feats["symbol"].map(universe.set_index("symbol")["industry"]).fillna("")
    grp = feats["ret_63"].groupby([feats["date"], industry])
    peer_sum, peer_n = grp.transform("sum"), grp.transform("count")
    own = feats["ret_63"].notna()
    peers = peer_n - own.astype(int)
    feats["sector_ret_63"] = ((peer_sum - feats["ret_63"].fillna(0)) / peers).where(
        (peers > 0) & (industry != ""))
    feats["sector_ret_63"] = feats["sector_ret_63"].fillna(feats["nifty_ret_63"])
    feats["rs_sector_63"] = feats["ret_63"] - feats["sector_ret_63"]

    feats = feats.merge(market_regime(indices), on="date", how="left")

    for col in RANKED:
        feats[f"{col}_rank"] = feats.groupby("date")[col].rank(pct=True)

    feats = feats[feats["history_days"] >= MIN_HISTORY_DAYS]
    drop = [c for c in feats.columns if c.startswith("_") or c.startswith("nifty_")]
    return feats.drop(columns=drop).reset_index(drop=True)


def _rolling_beta(feats: pd.DataFrame, n: int = 252) -> pd.Series:
    def beta(g):
        cov = g["_ret1"].rolling(n, min_periods=126).cov(g["nifty_ret_1"])
        var = g["nifty_ret_1"].rolling(n, min_periods=126).var()
        return cov / var
    return feats.groupby("symbol", group_keys=False)[["_ret1", "nifty_ret_1"]].apply(beta)


def weekly_snapshots(feats: pd.DataFrame) -> pd.DataFrame:
    """Last trading day of each week per stock — used for training (less overlap)."""
    week = feats["date"].dt.to_period("W-FRI")
    return feats.loc[feats.groupby([feats["symbol"], week])["date"].idxmax()].reset_index(drop=True)


def latest(feats: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per stock — used for today's predictions."""
    return feats.loc[feats.groupby("symbol")["date"].idxmax()].reset_index(drop=True)


FEATURE_COLUMNS_EXCLUDE = {"symbol", "date", "close", "history_days"}


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in FEATURE_COLUMNS_EXCLUDE]
