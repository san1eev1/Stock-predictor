"""The US profile (MARKET=us) is chosen per process, so it is checked in a subprocess."""
import os
import subprocess
import sys

US_CHECK = r'''
import numpy as np, pandas as pd
from stockpredictor import config, store, cloud, universe
from stockpredictor.data import daily, news
from stockpredictor.backtest import portfolio as P
from stockpredictor.costs import DEFAULT_COSTS as C
from stockpredictor.features import longterm as F
from stockpredictor.models import longterm as M
assert config.MARKET.name == "us" and config.load_settings().paper_capital_longterm == 1000
assert config.load_settings().db_path.name == "stockpredictor_us.db"
assert daily.yahoo_ticker("BRK.B") == "BRK-B" and daily.INDICES["SP500"] == "^GSPC"
assert F.MARKET_INDEX == "SP500" and "USDINR" not in F.GLOBAL_CUES
assert P.CIRCUIT_BANDS == () and P.shares(3.14159) == 3.1415
assert C.cost("buy", 1000) < 1.0                      # no commission, no ₹5-style minimum
assert store.DATA_BRANCH == "market-data-us" and cloud.STATE_BRANCH == "paper-state-us"
assert M.MODEL_DIR.name == "longterm_us"
assert "gl=US" in news.RSS_URL and news.company_query("Apple Inc.") == '"Apple" stock'
xml = ("<rss><channel><item><title>Apple (NASDAQ: AAPL) raises dividend - X</title>"
       "<source>X</source><pubDate>Mon, 16 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>")
assert len(news.parse_rss(xml, "AAPL")) == 1           # NASDAQ-tagged headlines kept in the US
csv = "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,Date added,CIK,Founded\nAAPL,Apple Inc.,Information Technology,x,y,1982,320193,1977\n"
assert universe.parse_sp500(csv)[0] == {"symbol": "AAPL", "name": "Apple Inc.",
                                         "industry": "Information Technology", "isin": "320193"}
# the $1,000 book: fractional sizing never spends more than the cash
dates = pd.bdate_range("2024-01-01", periods=30)
close = pd.DataFrame({s: np.linspace(400 + i, 450 + i, 30) for i, s in enumerate("ABCDEFG")}, index=dates)
scores = pd.DataFrame([(s, d, float(i)) for d in dates for i, s in enumerate("ABCDEFG")],
                      columns=["symbol", "date", "score"])
res = P.simulate(scores, close, P.Rules(), 1000, "weekly")
assert res.equity.min() > 900 and (res.trades["qty"] % 1 != 0).any()   # fractional shares
print("ok")
'''


def test_us_profile():
    env = {**os.environ, "MARKET": "us"}
    env.pop("PAPER_CAPITAL", None)
    r = subprocess.run([sys.executable, "-c", US_CHECK], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_india_is_the_default():
    from stockpredictor import config
    from stockpredictor.data import daily

    assert config.MARKET.name == "in" and daily.yahoo_ticker("RELIANCE") == "RELIANCE.NS"
