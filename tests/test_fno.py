from datetime import date

from stockpredictor import db
from stockpredictor.paper import fno as FO

ITEMS = [
    {"token": "1", "symbol": "ABC27OCT26FUT", "name": "ABC", "expiry": "27OCT2026", "strike": "-1",
     "lotsize": "100", "instrumenttype": "FUTSTK"},
    {"token": "2", "symbol": "ABC27OCT26500CE", "name": "ABC", "expiry": "27OCT2026",
     "strike": "50000", "lotsize": "100", "instrumenttype": "OPTSTK"},
    {"token": "3", "symbol": "ABC27OCT26520PE", "name": "ABC", "expiry": "27OCT2026",
     "strike": "52000", "lotsize": "100", "instrumenttype": "OPTSTK"},
    {"token": "4", "symbol": "ABC27OCT26500PE", "name": "ABC", "expiry": "27OCT2026",
     "strike": "50000", "lotsize": "100", "instrumenttype": "OPTSTK"},
]


class Client:
    def __init__(self, px):
        self.px = px

    def _require_login(self):
        return self

    def getMarketData(self, mode, req):
        return {"data": {"fetched": [{"symbolToken": t, "ltp": self.px[t]} for t in req["NFO"]]}}


def test_contracts_and_books(tmp_path, monkeypatch):
    today = date(2026, 10, 22)
    monkeypatch.setattr(FO, "instruments", lambda d: ITEMS)
    assert FO.future(ITEMS, "ABC", today).symbol == "ABC27OCT26FUT"
    assert FO.atm_option(ITEMS, "ABC", 503, "PE", today).strike == 500
    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    opened = FO.open_books(conn, Client({"1": 500.0, "2": 12.0, "4": 9.0}),
                           [("ABC", "long")], {"ABC": 503.0}, today, "2026-10-22 09:46")
    assert len(opened) == 2           # 1 future (margin 10k of 100k) + call option
    fut = FO.value(conn, {"ABC27OCT26FUT": 510.0}, FO.FUT_BOOK)
    assert fut["pnl"] > 0             # long future gained Rs 10 x lots x 100
    FO.square_off(conn, {"ABC27OCT26FUT": 510.0, "ABC27OCT26500CE": 15.0}, "2026-10-22 15:15")
    for b in FO.BOOKS:
        v = FO.value(conn, {}, b)
        assert v["positions"] == 0 and v["pnl"] > 0


def test_trade_outcomes_feed_learning(tmp_path):
    from stockpredictor.models import trainer as T

    db.init_db(tmp_path / "t.db")
    conn = db.connect(tmp_path / "t.db")
    conn.execute("INSERT INTO fno_contracts VALUES ('ABC27OCT26FUT', '1', 'ABC', 100)")
    for h, sym, pnl in (("fno_futures", "ABC27OCT26FUT", -50.0), ("intraday_close", "ABC", 20.0),
                        ("intraday", "XYZ", 10.0)):
        conn.execute("INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
                     "costs, status, pnl) VALUES (?, ?, 'long', 1, '2026-10-22 09:46', 1, 0, "
                     "'closed', ?)", (h, sym, pnl))
    out = T.paper_trade_outcomes(conn).set_index(["horizon", "symbol"])["correct"]
    assert out[("intraday_close", "ABC")] == 0      # the futures loss counts for the stock
    assert out[("intraday", "XYZ")] == 1


def test_trial_threshold_rises(tmp_path):
    from stockpredictor.models import trainer as T

    assert T.trial_t_min("x", 3, tmp_path) == T.MIN_T_STAT
    assert T.trial_t_min("x", 500, tmp_path) > 3.4
    T.reset_trials("x", tmp_path)
    assert T.trial_t_min("x", 1, tmp_path) == T.MIN_T_STAT


def test_fiidii_parse_and_update(tmp_path, monkeypatch):
    from stockpredictor.data import fiidii

    js = [{"buyValue": "19232.31", "category": "DII", "date": "23-Oct-2026", "netValue": "3235.36",
           "sellValue": "15996.95"}, {"buyValue": "14445.54", "category": "FII/FPI",
                                      "date": "23-Oct-2026", "netValue": "887.65", "sellValue": "13557.89"}]
    monkeypatch.setattr(fiidii, "fetch", lambda: fiidii.parse(js))
    assert fiidii.update(tmp_path) == 2 and fiidii.update(tmp_path) == 0   # no duplicates
    assert set(fiidii.load(tmp_path)["category"]) == {"FII", "DII"}


def test_ipo_subscription_parse_and_model():
    import numpy as np
    import pandas as pd

    from stockpredictor.data import ipo as IPO
    from stockpredictor.models import ipo as M

    js = {"dataList": [{"category": "Qualified Institutional Buyers(QIBs)", "noOfTotalMeant": "35.5"},
                       {"category": "Non Institutional Investors", "noOfTotalMeant": "158.3"},
                       {"category": "Retail Individual Investors(RIIs)", "noOfTotalMeant": "33.7"},
                       {"category": "Total", "noOfShareOffered": "1.65E7", "noOfTotalMeant": "60.8"}],
          "updateTime": "Updated as on 02-Nov-2026 19:00:00"}
    s = IPO.parse_subscription(js)
    assert s["qib_x"] == 35.5 and s["rii_x"] == 33.7 and s["shares_offered"] == 1.65e7
    rng = np.random.default_rng(0)
    n = 200
    qib = rng.lognormal(2, 1.5, n)
    d = pd.DataFrame({"qib_x": qib, "nii_x": qib * 2, "rii_x": qib / 2, "log_size_cr": 6.0,
                      "nifty_20d": 0.0, "recent_listings": 0.1,
                      "listing_date": pd.date_range("2015-01-01", periods=n, freq="15D")})
    for k in ("qib", "nii", "rii"):
        d[f"log_{k}"] = np.log1p(d[f"{k}_x"])
    d["gain_close"] = np.where(d["qib_x"] > 10, 0.2, -0.05) + rng.normal(0, 0.02, n)
    m, info = M.train_and_report(d)
    hot, cold = d.iloc[d["qib_x"].argmax()], d.iloc[d["qib_x"].argmin()]
    assert m.prob(hot) > 0.8 > 0.2 > m.prob(cold)
    label, p, why = M.verdict(m, hot, closes_today=True)
    assert label == "Apply" and why
    assert M.verdict(m, hot)[0] == "Wait"                 # still open, not the last day


def _zip(text: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("fo.csv", text)
    return buf.getvalue()


def test_fobhav_both_formats_and_features():
    from datetime import date

    import pandas as pd

    from stockpredictor.data import fobhav as FB

    old = ("INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,"
           "CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,\n"
           "FUTIDX,NIFTY,30-Nov-2023,0,XX,1,1,1,1,1,5,1,999,0,10-NOV-2023,\n"
           "FUTSTK,ABC,30-Nov-2023,0,XX,1,1,1,101,1,7,1,1000,0,10-NOV-2023,\n"
           "FUTSTK,ABC,28-Dec-23,0,XX,1,1,1,102,1,2,1,200,0,10-NOV-2023,\n"
           "OPTSTK,ABC,30-Nov-2023,100,CE,1,1,1,3,1,9,1,300,0,10-NOV-2023,\n"
           "OPTSTK,ABC,30-Nov-2023,100,PE,1,1,1,2,1,4,1,150,0,10-NOV-2023,\n")
    a = FB.parse(_zip(old), date(2023, 11, 10)).set_index("symbol")
    assert list(a.index) == ["ABC"]                         # index contracts dropped
    r = a.loc["ABC"]
    assert (r.fut_close, r.fut_oi, r.fut_oi_all, r.call_oi, r.put_oi) == (101, 1000, 1200, 300, 150)
    new = ("TradDt,FinInstrmTp,TckrSymb,XpryDt,OptnTp,ClsPric,OpnIntrst,TtlTradgVol\n"
           "2026-11-13,STF,ABC,2026-11-23,,105,1100,8\n"
           "2026-11-13,STO,ABC,2026-11-23,PE,2,330,4\n"
           "2026-11-13,STO,ABC,2026-11-23,CE,3,300,6\n"
           "2026-11-13,IDF,NIFTY,2026-11-23,,1,1,1\n")
    b = FB.parse(_zip(new), date(2026, 11, 13)).set_index("symbol").loc["ABC"]
    assert (b.fut_oi_all, b.put_oi, b.call_vol) == (1100, 330, 6)
    fo = pd.concat([FB.parse(_zip(old), date(2023, 11, 10)),
                    FB.parse(_zip(old.replace("1000,0", "1500,0")), date(2023, 11, 13))])
    close = pd.DataFrame({"ABC": [100.0, 100.0]}, index=pd.to_datetime(["2023-11-10", "2023-11-13"]))
    f = FB.features(fo, close).set_index("date")
    assert abs(f.loc["2023-11-13", "fo_oi_chg_1"] - (1700 / 1200 - 1)) < 1e-9
    assert abs(f.loc["2023-11-13", "fo_pcr"] - 0.5) < 1e-9
    assert abs(f.loc["2023-11-10", "fo_basis"] - 0.01) < 1e-9
