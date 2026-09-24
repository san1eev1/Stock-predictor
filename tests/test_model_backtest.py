import numpy as np
import pandas as pd
import pytest

from stockpredictor.backtest import portfolio as P
from stockpredictor.costs import DeliveryCosts
from stockpredictor.features import longterm as F
from stockpredictor.models import longterm as M
from test_features_longterm import synthetic


def test_delivery_costs():
    c = DeliveryCosts()
    buy = c.cost("buy", 10_000)
    sell = c.cost("sell", 10_000)
    # STT 10 + brokerage 10 + slippage 5 + stamp 1.5 + GST 1.86 + exchange 0.3
    assert buy == pytest.approx(28.66, abs=0.01)
    assert sell > buy  # DP charge on sell
    assert c.cost("buy", 0) == 0


def test_decide_rules():
    rules = P.Rules(n_hold=2, exit_rank=3, stop_loss=0.1)
    scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1})
    hold = {"A": P.Position(1, 100, pd.Timestamp("2024-01-01")),
            "D": P.Position(1, 100, pd.Timestamp("2024-01-01"))}
    prices = {s: 100.0 for s in scores.index}
    sells, buys = P.decide(hold, scores, prices, rules, rebalance=True)
    assert sells == [("D", "rank dropped")] and buys == ["B"]
    # Not a rebalance day: only risk exits, no buys.
    prices["A"] = 89.0
    sells, buys = P.decide(hold, scores, prices, rules, rebalance=False)
    assert sells == [("A", "stop-loss")] and buys == []
    # Negative news blocks buys; only severe news sells a holding.
    sells, buys = P.decide({}, scores, prices, rules, rebalance=True, negative_news={"A"})
    assert sells == [] and buys == ["B"]
    prices["A"] = 100.0
    held = {"A": P.Position(1, 100, pd.Timestamp("2024-01-01"))}
    assert P.decide(held, scores, prices, rules, False, negative_news={"A"})[0] == []
    assert P.decide(held, scores, prices, rules, False, severe_news={"A"})[0] == [("A", "negative news")]


def test_simulate_uses_next_day_and_charges_costs():
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.DataFrame({"A": [100, 110, 120, 130, 140, 150],
                          "B": [100, 100, 100, 100, 100, 100]}, index=dates, dtype=float)
    scores = pd.DataFrame([(s, d, 1.0 if s == "A" else 0.0) for d in dates for s in "AB"],
                          columns=["symbol", "date", "score"])
    res = P.simulate(scores, close, P.Rules(n_hold=1, exit_rank=1), capital=10_000)
    first = res.trades.iloc[0]
    assert first["date"] == dates[1] and first["symbol"] == "A" and first["price"] == 110
    assert res.total_costs > 0
    assert res.equity.iloc[-1] > 10_000


@pytest.fixture(scope="module")
def labeled():
    daily, indices, universe = synthetic(n_days=900, symbols=[f"S{i}" for i in range(12)])
    feats = F.build_features(daily, indices, None)
    return daily, indices, feats, M.add_labels(F.weekly_snapshots(feats), daily, indices)


def test_labels_are_forward_excess(labeled):
    daily, indices, _, lab = labeled
    row = lab.dropna(subset=["fwd_ret"]).iloc[0]
    px = daily[daily["symbol"] == row["symbol"]].set_index("date")["close"]
    i = px.index.get_loc(row["date"])
    assert row["fwd_ret"] == pytest.approx(px.iloc[i + M.HORIZON] / px.iloc[i] - 1)
    assert lab["target"].dropna().between(0, 1).all()
    assert lab[lab["date"] == lab["date"].max()]["target"].isna().all()   # future unknown


def test_train_save_load_explain(labeled, tmp_path):
    _, _, feats, lab = labeled
    model = M.LongTermModel.train(lab)
    latest = F.latest(feats)
    s1 = model.score(latest)
    model.save(tmp_path)
    loaded = M.LongTermModel.load(tmp_path)
    assert np.allclose(s1, loaded.score(latest))
    reasons = loaded.explain(latest.head(2))
    assert len(reasons) == 2 and all(r[0][0] in "+-" for r in reasons)


def test_walk_forward_trains_only_on_past(labeled, monkeypatch):
    _, _, feats, lab = labeled
    seen = []
    real_fit = M._fit
    monkeypatch.setattr(M, "_fit", lambda train, cols: (seen.append(train["date"].max()),
                                                        real_fit(train, cols))[1])
    monkeypatch.setattr(M, "walk_forward", M.walk_forward)
    lab = pd.concat([lab] * 20)  # enough rows for the minimum training size
    out = M.walk_forward(lab, feats, start_year=2022)
    for year, last_train in zip(sorted(out["date"].dt.year.unique()), seen):
        assert last_train < pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=M.EMBARGO_DAYS)
