import pandas as pd

from stockpredictor.data import fundamentals, news

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Reliance shares jump on strong results - Economic Times</title>
<link>https://x/1</link><pubDate>Thu, 24 Sep 2026 06:30:00 GMT</pubDate>
<source url="https://et">Economic Times</source></item>
<item><title>Reliance faces probe - Mint</title>
<link>https://x/2</link><pubDate>Wed, 23 Sep 2026 10:00:00 GMT</pubDate>
<source url="https://mint">Mint</source></item>
<item><title></title><pubDate>Wed, 23 Sep 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_company_query():
    assert news.company_query("Reliance Industries Ltd.") == '"Reliance Industries" share'
    assert news.company_query("Mahindra & Mahindra Ltd.") == '"Mahindra and Mahindra" share'


def test_parse_rss_strips_source_and_skips_empty():
    items = news.parse_rss(RSS, "RELIANCE")
    assert [i["title"] for i in items] == ["Reliance shares jump on strong results",
                                           "Reliance faces probe"]
    assert items[0]["published"] == "2026-09-24T06:30:00Z"
    assert items[0]["source"] == "Economic Times"


def test_append_dedupes_and_scores(tmp_path):
    rows = news.parse_rss(RSS, "RELIANCE")
    assert len(news.append_news(tmp_path, rows)) == 2
    assert len(news.append_news(tmp_path, rows)) == 0          # duplicates ignored
    assert news.score_missing(tmp_path, lambda t: [-0.9 if "probe" in x else 0.8 for x in t]) == 2
    assert news.score_missing(tmp_path, lambda t: 1 / 0) == 0  # nothing left to score
    df = news.load_news(tmp_path)
    assert sorted(df["sentiment"]) == [-0.9, 0.8]


def test_news_summary_uses_only_past(tmp_path):
    news.append_news(tmp_path, news.parse_rss(RSS, "RELIANCE"))
    news.score_missing(tmp_path, lambda t: [-0.9 if "probe" in x else 0.8 for x in t])
    df = news.load_news(tmp_path)
    s = news.news_summary(df, pd.Timestamp("2026-09-24T00:00:00Z")).iloc[0]
    assert s["news_count_7d"] == 1 and s["strong_negative"]
    s = news.news_summary(df, pd.Timestamp("2026-09-24T12:00:00Z")).iloc[0]
    assert s["news_count_7d"] == 2 and abs(s["sent_mean_7d"] + 0.05) < 1e-9


def test_fundamentals_parse_and_append(tmp_path):
    from datetime import date
    info = {"trailingPE": 25.5, "returnOnEquity": 0.12, "marketCap": 1.7e13, "debtToEquity": "n/a"}
    row = fundamentals.parse_info("RELIANCE", info, date(2026, 9, 25))
    assert row["pe"] == 25.5 and row["market_cap_cr"] == 1700000.0 and row["debt_to_equity"] is None
    fundamentals.append_snapshot(tmp_path, [row])
    fundamentals.append_snapshot(tmp_path, [row])   # same date replaces, not duplicates
    assert len(fundamentals.load_latest(tmp_path)) == 1
