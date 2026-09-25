"""Classic chart-reading methods turned into model features (one stock, causal).

The LightGBM model learns how much each signal is worth and how they combine, so the
signals are kept as plain numbers / 0-1 flags rather than hand-written buy/sell rules.

Groups:
  candles      daily candlestick patterns (1-3 day) and a net bullish-bearish score
  oscillators  stochastic, Williams %R, CCI, money flow index, rate of change
  trend        Supertrend, Ichimoku, Aroon, Donchian breakouts, Heikin-Ashi, Keltner squeeze
  volume       on-balance volume trend, Chaikin money flow
  statistics   trend slope and quality (R^2), autocorrelation, variance ratio,
               efficiency ratio, skew/kurtosis, z-score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BULLISH = ["cdl_hammer", "cdl_bull_engulf", "cdl_piercing", "cdl_morning_star",
           "cdl_three_white", "cdl_bull_harami", "cdl_bull_marubozu"]
BEARISH = ["cdl_shooting_star", "cdl_bear_engulf", "cdl_dark_cloud", "cdl_evening_star",
           "cdl_three_black", "cdl_bear_harami", "cdl_bear_marubozu"]


def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift()
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _aroon(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
    res = np.full(len(h), np.nan)
    if len(h) > n:
        win = np.lib.stride_tricks.sliding_window_view
        since_hi = n - np.argmax(win(h.to_numpy(float), n + 1), axis=1)
        since_lo = n - np.argmin(win(l.to_numpy(float), n + 1), axis=1)
        res[n:] = 100 * (since_lo - since_hi) / n
    return pd.Series(res, index=h.index)


def candles(g: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = g["open"], g["high"], g["low"], g["close"]
    rng = (h - l).replace(0, np.nan)
    body = c - o
    top, bot = pd.concat([o, c], axis=1).max(axis=1), pd.concat([o, c], axis=1).min(axis=1)
    upper, lower = (h - top) / rng, (bot - l) / rng
    rel = body / rng
    avg_body = body.abs().rolling(10, min_periods=5).mean()
    trend5 = c / c.shift(5) - 1               # context: patterns matter after a move
    down, up = trend5 < 0, trend5 > 0
    o1, c1, b1 = o.shift(1), c.shift(1), body.shift(1)
    o2, c2, b2 = o.shift(2), c.shift(2), body.shift(2)

    out = pd.DataFrame(index=g.index)
    out["cdl_body"] = rel
    out["cdl_upper_wick"] = upper
    out["cdl_lower_wick"] = lower
    out["cdl_doji"] = (rel.abs() < 0.1).astype(float)
    small = rel.abs() < 0.35
    out["cdl_hammer"] = (small & (lower > 0.55) & (upper < 0.15) & down).astype(float)
    out["cdl_shooting_star"] = (small & (upper > 0.55) & (lower < 0.15) & up).astype(float)
    out["cdl_bull_engulf"] = ((b1 < 0) & (body > 0) & (c >= o1) & (o <= c1)).astype(float)
    out["cdl_bear_engulf"] = ((b1 > 0) & (body < 0) & (c <= o1) & (o >= c1)).astype(float)
    out["cdl_piercing"] = ((b1 < 0) & (o < c1) & (c > (o1 + c1) / 2) & (c < o1)).astype(float)
    out["cdl_dark_cloud"] = ((b1 > 0) & (o > c1) & (c < (o1 + c1) / 2) & (c > o1)).astype(float)
    star = b1.abs() < 0.3 * avg_body
    out["cdl_morning_star"] = ((b2 < -avg_body) & star & (body > avg_body)
                               & (c > (o2 + c2) / 2)).astype(float)
    out["cdl_evening_star"] = ((b2 > avg_body) & star & (body < -avg_body)
                               & (c < (o2 + c2) / 2)).astype(float)
    out["cdl_three_white"] = ((body > 0) & (b1 > 0) & (b2 > 0) & (c > c1) & (c1 > c2)
                              & (upper < 0.3)).astype(float)
    out["cdl_three_black"] = ((body < 0) & (b1 < 0) & (b2 < 0) & (c < c1) & (c1 < c2)
                              & (lower < 0.3)).astype(float)
    inside = (top < pd.concat([o1, c1], axis=1).max(axis=1)) & \
             (bot > pd.concat([o1, c1], axis=1).min(axis=1))
    out["cdl_bull_harami"] = (inside & (b1 < -avg_body) & (body > 0)).astype(float)
    out["cdl_bear_harami"] = (inside & (b1 > avg_body) & (body < 0)).astype(float)
    out["cdl_bull_marubozu"] = ((rel > 0.9) & (body > avg_body)).astype(float)
    out["cdl_bear_marubozu"] = ((rel < -0.9) & (-body > avg_body)).astype(float)
    out["cdl_gap"] = (o / c1 - 1).clip(-0.2, 0.2)   # raw prices: splits look like huge gaps
    # Net pattern score over the last 5 days (recent patterns still matter a few days on).
    net = out[BULLISH].sum(axis=1) - out[BEARISH].sum(axis=1)
    out["cdl_score_5"] = net.rolling(5, min_periods=1).sum()
    return out


def oscillators(g: pd.DataFrame) -> pd.DataFrame:
    h, l, c = g["high"], g["low"], g["close"]
    vol = g["volume"].astype(float).replace(0, np.nan)
    out = pd.DataFrame(index=g.index)
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    out["stoch_k"] = k.rolling(3).mean()
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()
    out["williams_r"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    tp = (h + l + c) / 3
    mad = (tp - tp.rolling(20).mean()).abs().rolling(20).mean()
    out["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))
    flow = tp * vol
    pos = flow.where(tp > tp.shift(), 0.0).rolling(14, min_periods=10).sum()
    neg = flow.where(tp < tp.shift(), 0.0).rolling(14, min_periods=10).sum()
    out["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    out["roc_10"] = c / c.shift(10) - 1
    return out


def trend_systems(g: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = g["open"], g["high"], g["low"], g["close"]
    out = pd.DataFrame(index=g.index)
    tr = _true_range(h, l, c)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["atr_pct"] = atr / c

    # Supertrend (10, 3): +1 uptrend, -1 downtrend, and distance to the stop line.
    atr10 = tr.ewm(alpha=1 / 10, adjust=False, min_periods=10).mean().to_numpy()
    mid = ((h + l) / 2).to_numpy()
    cl = c.to_numpy()
    upper, lower = mid + 3 * atr10, mid - 3 * atr10
    fu, fl = upper.copy(), lower.copy()
    direction = np.full(len(cl), np.nan)
    line = np.full(len(cl), np.nan)
    d = 1.0
    for i in range(1, len(cl)):
        if np.isnan(atr10[i]):
            continue
        if not np.isnan(fu[i - 1]) and (upper[i] > fu[i - 1] and cl[i - 1] <= fu[i - 1]):
            fu[i] = fu[i - 1]
        if not np.isnan(fl[i - 1]) and (lower[i] < fl[i - 1] and cl[i - 1] >= fl[i - 1]):
            fl[i] = fl[i - 1]
        if cl[i] > fu[i - 1] if not np.isnan(fu[i - 1]) else False:
            d = 1.0
        elif cl[i] < fl[i - 1] if not np.isnan(fl[i - 1]) else False:
            d = -1.0
        direction[i] = d
        line[i] = fl[i] if d > 0 else fu[i]
    out["supertrend_dir"] = direction
    out["supertrend_dist"] = cl / line - 1

    # Ichimoku: price vs cloud, conversion vs base line.
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    out["ichi_above_cloud"] = (c / cloud_top - 1).where(c > cloud_top, 0.0) \
        + (c / cloud_bot - 1).where(c < cloud_bot, 0.0)
    out["ichi_tk"] = tenkan / kijun - 1

    # Aroon oscillator (25): how recently the high vs the low was made.
    out["aroon_osc"] = _aroon(h, l, 25)

    # Donchian breakouts: position in the 20/55-day channel (turtle-trader levels).
    for w in (20, 55):
        hi, lo = h.shift().rolling(w).max(), l.shift().rolling(w).min()
        out[f"donchian_pos_{w}"] = (c - lo) / (hi - lo).replace(0, np.nan)

    # Heikin-Ashi: smoothed candle colour streak (+ up days, - down days), capped at 10.
    ha_close = (o + h + l + c) / 4
    hc = ha_close.to_numpy(float)
    ho = hc.copy()
    ho[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(ho)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    sign = pd.Series(np.sign(hc - ho), index=g.index)
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount().add(1) * sign
    out["ha_streak"] = streak.clip(-10, 10)

    # Keltner squeeze: Bollinger width / Keltner width (< 1 = volatility coiled up).
    sd = c.rolling(20).std()
    out["squeeze"] = (4 * sd) / (4 * atr).replace(0, np.nan)
    return out


def volume_flow(g: pd.DataFrame) -> pd.DataFrame:
    h, l, c = g["high"], g["low"], g["close"]
    vol = g["volume"].astype(float).replace(0, np.nan)
    out = pd.DataFrame(index=g.index)
    obv = (np.sign(c.diff()).fillna(0) * vol.fillna(0)).cumsum()
    avg = vol.rolling(20, min_periods=10).mean()
    out["obv_slope_20"] = (obv - obv.shift(20)) / (20 * avg)
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    out["cmf_20"] = (mfm * vol).rolling(20, min_periods=10).sum() \
        / vol.rolling(20, min_periods=10).sum()
    return out


def statistics(px: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=px.index)
    logp = np.log(px)
    ret = logp.diff()
    # Linear trend over 63 days: annualised slope of log price and its R^2 (trend quality).
    n = 63
    t = pd.Series(np.arange(len(px), dtype=float), index=px.index)
    cov = logp.rolling(n).cov(t)
    var_t = t.rolling(n).var()
    out["trend_slope_63"] = cov / var_t * 252
    out["trend_r2_63"] = logp.rolling(n).corr(t) ** 2
    out["autocorr_63"] = ret.rolling(n).corr(ret.shift())
    # Variance ratio (Lo-MacKinlay): > 1 trending, < 1 mean-reverting.
    r5 = logp.diff(5)
    out["var_ratio_126"] = r5.rolling(126, min_periods=100).var() \
        / (5 * ret.rolling(126, min_periods=100).var())
    # Kaufman efficiency ratio: net move / total path (1 = straight line).
    out["efficiency_20"] = (logp - logp.shift(20)).abs() / ret.abs().rolling(20).sum()
    out["skew_63"] = ret.rolling(n).skew()
    out["kurt_63"] = ret.rolling(n).kurt()
    out["zscore_50"] = (px - px.rolling(50).mean()) / px.rolling(50).std()
    return out


def _group_columns() -> dict[str, list[str]]:
    g = pd.DataFrame({"open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3, "close": [1.0] * 3,
                      "adj_close": [1.0] * 3, "volume": [1.0] * 3})
    return {"candles": list(candles(g).columns), "oscillators": list(oscillators(g).columns),
            "trend": list(trend_systems(g).columns), "volume": list(volume_flow(g).columns),
            "statistics": list(statistics(g["close"]).columns)}


# Column names per group (plus the cross-sectional ranks longterm.py adds for some of them).
GROUPS = _group_columns()
for _grp, _col in (("oscillators", "stoch_k"), ("volume", "cmf_20"),
                   ("statistics", "trend_slope_63"), ("trend", "supertrend_dist")):
    GROUPS[_grp].append(f"{_col}_rank")
# Only trend systems improved the weekly model out-of-sample (2019-2026 walk-forward);
# candles / statistics lowered it. Weekly self-tuning can switch other groups on
# if they start to prove themselves.
DEFAULT_GROUPS = ["trend"]


def excluded_columns(groups: list[str] | None) -> set[str]:
    on = set(DEFAULT_GROUPS if groups is None else groups)
    return {c for g, cols in GROUPS.items() if g not in on for c in cols}


def build(g: pd.DataFrame) -> pd.DataFrame:
    """All technical features for one stock's daily rows (sorted by date)."""
    px = g["adj_close"].fillna(g["close"])
    return pd.concat([candles(g), oscillators(g), trend_systems(g), volume_flow(g),
                      statistics(px)], axis=1)
