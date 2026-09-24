import numpy as np
import pandas as pd
import pytest

from stockpredictor.data import intraday as I
from stockpredictor.features import intraday as FI
from stockpredictor.features import longterm as F
from test_features_longterm import synthetic


def fake_summaries(daily, n_days=60, seed=1):
    """Intraday summaries consistent with synthetic daily candles."""
    rng = np.random.default_rng(seed)
    rows = []
    recent = daily.sort_values(["symbol", "date"]).groupby("symbol").tail(n_days)
    for (sym, d), r in recent.set_index(["symbol", "date"]).iterrows():
        c30 = r["open"] * (1 + rng.normal(0, 0.005))
        row = {"symbol": sym, "date": d.strftime("%Y-%m-%d"), "open": r["open"],
               "h30": max(r["open"], c30) * 1.002, "l30": min(r["open"], c30) * 0.998,
               "c30": c30, "v30": 1000.0, "vwap30": (r["open"] + c30) / 2,
               "high_after": r["high"], "low_after": r["low"], "px_1515": r["close"],
               "close": r["close"], "source": "yahoo"}
        for lv in I.LEVELS:
            row[I.level_col("u", lv)] = 60 if r["high"] >= c30 * (1 + lv / 100) else np.nan
            row[I.level_col("d", lv)] = 90 if r["low"] <= c30 * (1 - lv / 100) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    daily, indices, universe = synthetic(n_days=400, symbols=[f"S{i}" for i in range(12)])
    lt = F.build_features(daily, indices, universe)
    summ = fake_summaries(daily)
    return daily, lt, summ, FI.build(summ, daily, lt)


def test_uses_previous_day_context(data):
    daily, lt, summ, f = data
    row = f.iloc[-1]
    d = daily[daily["symbol"] == row["symbol"]].set_index("date")
    prev = d.index[d.index < row["date"]].max()
    assert row["prev_date"] == prev
    assert row["prev_close"] == pytest.approx(d.loc[prev, "close"])
    assert row["gap"] == pytest.approx(row["open"] / d.loc[prev, "close"] - 1)


def test_target_and_ranks(data):
    f = data[3]
    assert f["target"].between(0, 1).all()
    assert f["r30_rank"].between(0, 1).all()
    cols = FI.feature_columns(f)
    assert "target_ret" not in cols and "px_1515" not in cols and "u100" not in cols
    assert {"gap", "rel_r30", "vwap_dev", "breadth30", "rsi_14", "atr_pct"} <= set(cols)


def test_no_future_leak_in_features(data):
    """Changing today's afternoon prices must not change any feature."""
    daily, lt, summ, f = data
    s2 = summ.copy()
    last = s2["date"] == s2["date"].max()
    s2.loc[last, ["px_1515", "close", "high_after", "low_after"]] *= 1.5
    f2 = FI.build(s2, daily, lt)
    cols = FI.feature_columns(f)
    pd.testing.assert_frame_equal(f[cols], f2[cols])


def test_split_adjustment_only_for_angel():
    summ = pd.DataFrame([{"symbol": "A", "date": pd.Timestamp("2024-01-02"), "source": s,
                          **{c: 200.0 for c in FI.PRICE_COLS}, "v30": 10.0}
                         for s in ("angelone", "yahoo")])
    actions = pd.DataFrame([{"symbol": "A", "date": "2024-06-01", "kind": "split", "value": 2.0}])
    out = FI.adjust_splits(summ, actions)
    assert out["c30"].tolist() == [100.0, 200.0] and out["v30"].tolist() == [20.0, 10.0]
