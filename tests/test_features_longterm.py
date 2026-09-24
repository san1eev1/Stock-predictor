import numpy as np
import pandas as pd
import pytest

from stockpredictor.features import longterm


def synthetic(n_days=600, symbols=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    def candles(sym, drift, vol):
        close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n_days)))
        open_ = close * (1 + rng.normal(0, vol / 3, n_days))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, vol / 2, n_days)))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, vol / 2, n_days)))
        return pd.DataFrame({"symbol": sym, "date": dates, "open": open_, "high": high,
                             "low": low, "close": close, "adj_close": close,
                             "volume": rng.integers(1e5, 1e6, n_days), "source": "test"})

    daily = pd.concat([candles(s, 0.0005 * i, 0.02) for i, s in enumerate(symbols)],
                      ignore_index=True)
    indices = pd.concat([candles("NIFTY50", 0.0003, 0.01), candles("NIFTYIT", 0.0004, 0.015),
                         candles("INDIAVIX", 0, 0.05)], ignore_index=True)
    universe = pd.DataFrame({"symbol": list(symbols),
                             "industry": ["Banks", "Banks", "Unknown"]})
    return daily, indices, universe


@pytest.fixture(scope="module")
def data():
    daily, indices, universe = synthetic()
    return daily, indices, universe, longterm.build_features(daily, indices, universe)


def test_shape_and_min_history(data):
    daily, _, _, feats = data
    assert feats["symbol"].nunique() == 3
    assert feats.groupby("symbol").size().eq(600 - longterm.MIN_HISTORY_DAYS + 1).all()
    assert not feats.duplicated(["symbol", "date"]).any()


def test_expected_columns_present(data):
    cols = set(longterm.feature_columns(data[3]))
    assert data[3]["up_volume_ratio_20"].notna().mean() > 0.95
    for c in ["ret_63", "mom_12_1", "dist_ma200", "dist_52w_high", "vol_63", "maxdd_126",
              "rsi_14", "macd_hist", "adx_14", "bb_pctb", "vol_ratio_20_120", "beta_252",
              "rs_nifty_63", "rs_sector_63", "wk_body", "wk_bull_engulf", "wk_streak",
              "mkt_above_ma200", "vix_pct_252", "ret_63_rank"]:
        assert c in cols, c


def test_values_sane(data):
    feats = data[3]
    assert feats["rsi_14"].between(0, 100).all()
    assert feats["ret_63_rank"].between(0, 1).all()
    assert (feats["maxdd_126"].dropna() <= 0).all()
    last = feats.dropna(subset=["ret_21"]).iloc[-1]
    daily = data[0].set_index(["symbol", "date"])["close"]
    sym = daily.loc[last["symbol"]]
    i = sym.index.get_loc(last["date"])
    assert last["ret_21"] == pytest.approx(sym.iloc[i] / sym.iloc[i - 21] - 1)


def test_sector_strength_uses_peers(data):
    feats = data[3].dropna(subset=["ret_63"])
    wide = feats.pivot(index="date", columns="symbol", values="ret_63")
    aaa = feats[feats["symbol"] == "AAA"].set_index("date")
    # AAA's only peer in "Banks" is BBB, so its sector return is BBB's return.
    assert np.allclose(aaa["sector_ret_63"], wide.loc[aaa.index, "BBB"])
    # CCC has no peers -> falls back to Nifty.
    ccc = feats[feats["symbol"] == "CCC"].dropna(subset=["rs_nifty_63"])
    assert np.allclose(ccc["rs_sector_63"], ccc["rs_nifty_63"])


@pytest.mark.parametrize("cut", ["2021-03-17", "2021-06-04", "2022-02-02"])
def test_no_lookahead(data, cut):
    """Features at a date must be identical whether or not later data exists."""
    daily, indices, universe, full = data
    cut = pd.Timestamp(cut)
    trunc = longterm.build_features(daily[daily["date"] <= cut],
                                    indices[indices["date"] <= cut], universe)
    a = full[full["date"] == cut].set_index("symbol").sort_index()
    b = trunc[trunc["date"] == cut].set_index("symbol").sort_index()
    assert len(a) == 3
    pd.testing.assert_frame_equal(a[b.columns], b, check_exact=False, rtol=1e-9)


def test_snapshots(data):
    feats = data[3]
    weekly = longterm.weekly_snapshots(feats)
    assert (weekly["date"].dt.dayofweek == 4).mean() > 0.95   # Fridays (bdate range)
    assert set(longterm.latest(feats)["date"]) == {feats["date"].max()}
