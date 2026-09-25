import numpy as np
import pandas as pd

from stockpredictor.features import technical as T


def ohlc(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["adj_close"] = df["close"]
    df["volume"] = 1000.0
    df["date"] = pd.bdate_range("2024-01-01", periods=len(df))
    return df


def random_walk(n=400, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    df = ohlc(np.c_[o, h, l, c])
    df["volume"] = rng.uniform(1e5, 2e5, n)
    return df


def test_no_lookahead():
    g = random_walk()
    full = T.build(g)
    cut = T.build(g.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], cut, check_exact=False, rtol=1e-6)


def test_candle_patterns():
    down = [[110 - 2 * i, 111 - 2 * i, 108 - 2 * i, 109 - 2 * i] for i in range(6)]
    engulf = ohlc(down + [[97, 104, 96.5, 103]])          # red day, then big green day
    assert T.candles(engulf)["cdl_bull_engulf"].iloc[-1] == 1
    hammer = ohlc(down + [[98, 98.3, 92, 98.2]])            # long lower wick after a fall
    assert T.candles(hammer)["cdl_hammer"].iloc[-1] == 1
    up = [[100 + i, 101.2 + i, 99.8 + i, 101 + i] for i in range(6)]
    assert T.candles(ohlc(up))["cdl_three_white"].iloc[-1] == 1


def test_trend_signals_follow_direction():
    rising = ohlc([[100 + i, 101.5 + i, 99.5 + i, 101 + i] for i in range(120)])
    f = T.build(rising).iloc[-1]
    assert f["supertrend_dir"] == 1 and f["aroon_osc"] > 50 and f["trend_r2_63"] > 0.95
    falling = ohlc([[300 - i, 300.5 - i, 298.5 - i, 299 - i] for i in range(120)])
    assert T.build(falling).iloc[-1]["supertrend_dir"] == -1
