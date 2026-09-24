import numpy as np
import pandas as pd

from stockpredictor.models import engine


def data(n_days=120, n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=n_days), n)
    x = rng.normal(size=(len(dates), 3))
    y = 0.5 + 0.1 * x[:, 0] + rng.normal(0, 0.1, len(dates))
    return pd.DataFrame({"date": dates, "a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "target": y})


def test_early_stopping_picks_rounds_and_ensemble_averages(tmp_path):
    df = data()
    params = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 7,
              "num_rounds": 400, "early_stopping": True, "n_seeds": 3, "verbose": -1}
    m = engine.fit(df, ["a", "b", "c"], params)
    assert len(m.boosters) == 3
    assert engine.PATIENCE <= m.rounds < 400          # stopped well before the cap
    pred = m.predict(df[["a", "b", "c"]].astype(np.float32))
    assert np.corrcoef(pred, df["a"])[0, 1] > 0.9
    m.save(tmp_path)
    m2 = engine.Ensemble.load(tmp_path)
    assert len(m2.boosters) == 3
    assert np.allclose(pred, m2.predict(df[["a", "b", "c"]].astype(np.float32)))
    assert m2.feature_importance().argmax() == 0


def test_no_early_stopping_uses_fixed_rounds():
    df = data(n_days=30)
    m = engine.fit(df, ["a", "b", "c"], {"objective": "regression", "num_rounds": 25,
                                         "early_stopping": False, "n_seeds": 1, "verbose": -1})
    assert m.rounds == 25 and m.boosters[0].num_trees() == 25
