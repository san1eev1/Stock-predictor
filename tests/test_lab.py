import sqlite3
from datetime import datetime

import pandas as pd
import pytest

from stockpredictor import db
from stockpredictor.data import news as N
from stockpredictor.paper import intraday as PI
from stockpredictor.paper import lab as LAB


@pytest.fixture
def conn(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def _feats():
    return pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        "gap": [0.03, -0.025, 0.001, 0.0, 0.002, -0.001],
        "gap_atr": [1.8, -1.5, 0.1, 0.0, 0.1, -0.1],
        "r_prev": [0.02, -0.03, 0.01, -0.01, 0.004, 0.0],
        "r30": [-0.004, 0.005, 0.01, -0.012, 0.004, 0.001],
        "vol30_adv": [0.1, 0.2, 0.6, 0.5, 0.05, 0.01],
        "ret_5": [0.08, -0.07, 0.01, -0.02, 0.03, 0.0],
        "c30": [100.0, 200.0, 50.0, 80.0, 30.0, 10.0],
        "h30": [101.0, 202.0, 50.5, 81.0, 30.2, 10.1],
        "l30": [99.0, 198.0, 49.5, 79.0, 29.8, 9.9],
        "prev_close": [97.0, 205.0, 49.0, 81.0, 29.9, 10.0]})


def test_candidates_follow_each_rule():
    f = _feats()
    ns = pd.DataFrame({"symbol": ["CCC", "DDD"], "n": [2, 1], "tone": [0.8, -0.7],
                       "sector_tone": [None, None], "score": [0.8, -0.7], "headline": ["x", "y"]})
    news = LAB.candidates("lab_news", f, ns, set())
    assert dict(zip(news["symbol"], news["side"])) == {"CCC": "long", "DDD": "short"}
    fade = LAB.candidates("lab_fade", f, ns, {"CCC", "DDD"})
    assert dict(zip(fade["symbol"], fade["side"])) == {"AAA": "short", "BBB": "long"}
    assert fade.set_index("symbol").loc["AAA", "target"] == 97.0       # back to the close
    fade_news = LAB.candidates("lab_fade", f, ns, {"AAA"})               # AAA had news
    assert "AAA" not in set(fade_news["symbol"])
    inplay = LAB.candidates("lab_inplay", f, ns, set())
    top = inplay.set_index("symbol")
    assert top.loc["CCC", "side"] == "long" and top.loc["CCC", "stop"] == 49.5
    assert top.loc["DDD", "side"] == "short" and top.loc["DDD", "stop"] == 81.0
    rev = LAB.candidates("lab_reversal", f, ns, set())
    assert rev[rev["side"] == "long"].iloc[0]["symbol"] == "BBB"
    assert rev[rev["side"] == "short"].iloc[0]["symbol"] == "AAA"


def test_open_book_trades_and_square_off(conn):
    f = _feats()
    picks = LAB.candidates("lab_inplay", f, pd.DataFrame(columns=["symbol"]), set())
    prices = dict(zip(f["symbol"], f["c30"]))
    opened = LAB.open_book(conn, "lab_inplay", picks, f, prices, pd.Timestamp("2026-11-27"),
                           10_000)
    assert opened and len(PI.open_trades(conn, "lab_inplay")) == len(opened)
    n_pred = conn.execute("SELECT COUNT(*) FROM predictions WHERE horizon='lab_inplay'").fetchone()[0]
    assert n_pred == len(picks)
    PI.square_off(conn, {s: p * 1.01 for s, p in prices.items()}, "2026-11-27 15:15", "lab_inplay")
    assert not PI.open_trades(conn, "lab_inplay")
    h = LAB.history(conn)
    assert set(h["book"]) == {"lab_inplay"}


def test_news_scores_company_and_sector():
    uni = pd.DataFrame({"symbol": ["AAA", "BBB"], "industry": ["Metals & Mining", "Power"]})
    t = pd.Timestamp("2026-11-27 03:00", tz="UTC")
    news = pd.DataFrame({
        "symbol": ["AAA", "@steel", "@steel", "BBB"],
        "published": [t, t, t, t - pd.Timedelta(days=3)],
        "title": ["AAA wins Rs 500 crore order", "Steel import duty raised", "Steel prices firm",
                  "BBB old news"],
        "sentiment": [0.9, 0.6, 0.4, -0.9], "relevance": [1.2, 1.0, 1.0, 1.0], "source": "x"})
    ns = LAB.news_scores(news, uni, t - pd.Timedelta(hours=18), t + pd.Timedelta(hours=1))
    a = ns.set_index("symbol").loc["AAA"]
    assert a["n"] == 1 and a["tone"] == pytest.approx(0.9)
    assert a["sector_tone"] == pytest.approx(0.5)
    assert a["score"] == pytest.approx(0.9 + LAB.SECTOR_WEIGHT * 0.5)
    assert "BBB" not in set(ns["symbol"])                 # outside the window


def test_noise_filter_drops_commentary_keeps_events():
    assert N.is_noise("Biocon Ltd. Downgraded to Strong Sell Amid Weak Financials")
    assert N.is_noise("5 PSU Bank Stocks Trading Below Their Historical Valuations")
    assert N.is_noise("Tech Mahindra Ltd Sees Sharp Open Interest Surge")
    assert N.is_noise("Dr. Reddy's Laboratories Share Price News and Valuation")
    assert N.is_noise("Anything at all", source="Univest")
    assert not N.is_noise("Wipro secures $500 million contract from energy major BP")
    assert not N.is_noise("Lupin gets USFDA approval for single-dose dalbavancin injection")
    assert not N.is_noise("Dalmia Bharat approves ₹700 crore share buyback via open market route")


def test_pick_quality_trades_3_to_5_and_skips_news_against_and_sideways():
    n = 40
    day = pd.DataFrame({"symbol": [f"S{i}" for i in range(n)],
                        "range30_atr": [0.1 if i in (1, 2) else 1.0 for i in range(n)],
                        "sec_r30": [0.01] * n, "mkt_r30": [0.002] * n})
    longs = day.head(25)
    out, k, notes = PI.pick_quality(longs, day, "long", {"S0": -0.8, "S3": 0.6})
    assert 3 <= k <= 5
    traded = list(out["symbol"].head(k))
    assert "S0" not in traded and "S1" not in traded and "S2" not in traded
    assert traded[0] == "S3"                                  # news agrees: most points
    assert notes[list(out["symbol"]).index("S0")] == "news against"
    flat = day.assign(sec_r30=-0.01, mkt_r30=-0.01)
    _, k2, _ = PI.pick_quality(flat.head(25), flat, "long", {})
    assert k2 == 5 or k2 == 3                                 # always within 3-5
