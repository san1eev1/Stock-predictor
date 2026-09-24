"""Plain-English names for model features (used to explain picks in the app)."""

LABELS = {
    "ret_5": "1-week return", "ret_21": "1-month return", "ret_63": "3-month return",
    "ret_126": "6-month return", "ret_252": "1-year return",
    "mom_12_1": "12-month momentum", "dist_ma50": "distance from 50-day average",
    "dist_ma200": "distance from 200-day average", "ma50_over_ma200": "50 vs 200-day trend",
    "dist_52w_high": "distance from 52-week high", "dist_52w_low": "distance from 52-week low",
    "vol_21": "1-month volatility", "vol_63": "3-month volatility", "vol_126": "6-month volatility",
    "maxdd_126": "6-month max drawdown", "rsi_14": "RSI", "macd_hist": "MACD",
    "adx_14": "trend strength (ADX)", "bb_pctb": "Bollinger position",
    "vol_ratio_20_120": "recent volume vs normal", "turnover_log": "trading value (liquidity)",
    "up_volume_ratio_20": "buying volume share", "wk_body": "weekly candle body",
    "wk_upper_wick": "weekly upper wick", "wk_lower_wick": "weekly lower wick",
    "wk_doji": "weekly doji", "wk_hammer": "weekly hammer",
    "wk_bull_engulf": "weekly bullish engulfing", "wk_bear_engulf": "weekly bearish engulfing",
    "wk_inside": "weekly inside bar", "wk_streak": "weekly up/down streak",
    "rs_nifty_63": "3-month strength vs Nifty", "rs_nifty_126": "6-month strength vs Nifty",
    "beta_252": "beta", "sector_ret_63": "sector 3-month return",
    "rs_sector_63": "strength vs sector peers", "mkt_above_ma200": "Nifty above 200-day average",
    "mkt_ret_63": "Nifty 3-month return", "mkt_vol_21": "market volatility",
    "vix": "India VIX", "vix_pct_252": "VIX vs past year",
}


def label(feature: str) -> str:
    if feature.endswith("_rank"):
        return LABELS.get(feature[:-5], feature[:-5]) + " (rank)"
    return LABELS.get(feature, feature)


def reason_text(reasons: list[str]) -> str:
    """['+ ret_63', '- vol_63'] -> '↑ 3-month return · ↓ 3-month volatility'."""
    return " · ".join(("↑ " if r.startswith("+") else "↓ ") + label(r[2:]) for r in reasons)
