import json
import subprocess
from datetime import date

import pandas as pd

from stockpredictor import db, store
from stockpredictor.data import news as N
from stockpredictor.models import trainer as T
from stockpredictor.nlp import relevance as R


def test_mac_keeps_only_recent_price_files():
    p = store._sparse_patterns(4, date(2026, 9, 25))
    assert "/daily/2023.csv" in p and "/daily/2026.csv" in p and "/daily/2022.csv" not in p
    assert "/indices/2023.csv" in p and "/news/" in p and "/universe.csv" in p


def test_feedback_branch_roundtrip(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    judged = pd.DataFrame({"symbol": ["A", "B"], "date": ["2026-09-01", "2026-09-02"],
                           "correct": [1, 0]})
    store.push_feedback(judged, remote=str(remote))
    store.push_feedback(judged.head(1), remote=str(remote))      # replaced, not appended
    out = tmp_path / "fb"
    assert store._branch_checkout(out, store.FEEDBACK_BRANCH, remote=str(remote))
    back = store.load_feedback(out)
    assert back["symbol"].tolist() == ["A"]
    log = subprocess.run(["git", "--git-dir", str(remote), "rev-list", "--count",
                          store.FEEDBACK_BRANCH], capture_output=True, text=True, check=True)
    assert log.stdout.strip() == "1"                             # branch never grows
    assert not store._branch_checkout(tmp_path / "x", "no-such-branch", remote=str(remote))


def test_training_history_merges_cloud_runs(tmp_path):
    db.init_db(tmp_path / "t.db")
    c = db.connect(tmp_path / "t.db")
    T.log_run(c, "intraday", "retrain", "2026-09-24", {"days": 400})
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps({"horizon": "longterm", "version": "2026-09-25T12:00:00",
                                "kind": "tune", "train_to": "2026-09-18",
                                "metrics": {"ic": 0.05, "adopted": True}}) + "\n")
    out = T.load_runs(c, runs)
    assert set(out["horizon"]) == {"intraday", "longterm"}
    assert json.loads(out.loc[out["horizon"] == "longterm", "metrics"].iloc[0])["ic"] == 0.05


def test_tip_headlines_are_ignored():
    tips = ["Buy Jindal Steel; target of Rs 1290: Motilal Oswal", "Stocks to watch: RVNL, Wipro",
            "Top Gainers & Losers, September 18", "Is it a stock to buy, sell or hold after Q3?",
            "NTPC Stock 12-Month Price Target Raised to INR 437"]
    news = ["Tata Power shares fall 4%: arbitration lost", "Adani Ports handles record cargo",
            "PNB Q1 profit jumps over 200%"]
    assert all(N.TIPS.search(t) for t in tips)
    assert not any(N.TIPS.search(t) for t in news)


def test_relevance_weights_news_signals(tmp_path):
    now = pd.Timestamp("2026-09-25 06:00", tz="UTC")
    news = pd.DataFrame({
        "symbol": ["A", "A", "A"], "title": ["A wins big order", "A in list of 10", "A plant fire"],
        "published": [now - pd.Timedelta(hours=h) for h in (1, 2, 3)],
        "sentiment": [0.8, -0.9, -0.9], "relevance": [2.0, 0.1, 0.5]})
    s = N.news_summary(news, now).set_index("symbol").loc["A"]
    assert s["sent_mean_7d"] > 0.4          # the relevant good news dominates the noisy bad ones
    assert s["bad_news_3d"] == 0            # bad headlines below the relevance bar don't count
    uni = pd.DataFrame({"symbol": ["A"], "name": ["A Ltd"]})
    assert (R.add_relevance(news.drop(columns="relevance"), uni, tmp_path)["relevance"] == 1).all()
