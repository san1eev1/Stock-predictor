import pandas as pd

from stockpredictor.data.clean import clean_daily


def frame(rows):
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"]) \
        .assign(date=lambda d: pd.to_datetime(d["date"]))


def test_bad_spike_row_is_dropped():
    df = frame([("A", "2005-07-27", 50, 51, 49, 50, 100), ("A", "2005-07-28", 218, 218, 218, 218, 0),
                ("A", "2005-07-29", 50, 51, 49, 50.2, 90)])
    out, rep = clean_daily(df)
    assert list(out["close"]) == [50, 50.2] and rep["empty_rows"] == 1


def test_unadjusted_demerger_is_back_adjusted_but_a_crash_is_kept():
    df = frame([("V", "2026-04-28", 770, 780, 765, 774, 100), ("V", "2026-04-29", 772, 776, 768, 773.6, 100),
                ("V", "2026-04-30", 289.5, 292, 268, 271.55, 400),        # demerger: calm gap
                ("Y", "2020-03-05", 36, 37.5, 35, 36.8, 100),
                ("Y", "2020-03-06", 33.15, 34, 5.55, 16.15, 140)])        # crash: huge range
    out, rep = clean_daily(df)
    v = out[out["symbol"] == "V"]["close"].tolist()
    assert abs(v[1] - 773.6 * 289.5 / 773.6) < 1e-6 and v[2] == 271.55    # continuous now
    assert out[out["symbol"] == "Y"]["close"].tolist() == [36.8, 16.15]    # untouched
    assert [b[0] for b in rep["breaks"]] == ["V"]


def test_intraday_prices_put_on_the_daily_basis():
    from stockpredictor.features.intraday import align_to_daily

    daily = pd.DataFrame({"symbol": "V", "date": pd.to_datetime(["2026-04-27", "2026-04-28"]),
                          "close": [288.0, 289.5]})                  # back-adjusted history
    summ = pd.DataFrame({"symbol": "V", "date": pd.to_datetime(["2026-04-27", "2026-04-28",
                                                                "2026-04-29"]),
                         "open": [770.0, 771.0, 772.0], "c30": [772.0, 773.0, 774.0],
                         "close": [769.6, 773.6, 775.0], "v30": [100.0, 400.0, 300.0]})
    out = align_to_daily(summ, daily)
    f = 288.0 / 769.6                                    # previous day's basis factor
    assert abs(out["close"][1] - 773.6 * f) < 1e-9        # day 2 uses day 1's factor
    assert abs(out["c30"][2] - 774.0 * 289.5 / 773.6) < 1e-9   # live day: latest factor
    assert out["c30"][0] == 772.0                        # first day: nothing known before


def test_nse_delivery_file_is_parsed():
    from datetime import date

    from stockpredictor.data import delivery

    text = """Security Wise Delivery Position - Compulsory Rolling Settlement
10,MTO,24092015,414433471,0001526
Record Type,Sr No,Name of Security,Quantity Traded,Deliverable Quantity,% of Deliverable
20,1,20MICRONS,EQ,27482,15476,56.31
20,2,ABC,BE,100,100,100.00
20,3,TCS,EQ,1000,600,60.00"""
    df = delivery.parse(text, date(2015, 9, 24))
    assert list(df["symbol"]) == ["20MICRONS", "TCS"] and df["deliv_pct"].tolist() == [56.31, 60.0]
    f = delivery.features(pd.concat([df.assign(date=pd.Timestamp("2015-09-24") + pd.Timedelta(days=i))
                                     for i in range(130)]))
    assert abs(f["deliv_pct"].iloc[-1] - 0.60) < 1e-9 and abs(f["deliv_pct_rel"].iloc[-1] - 1) < 1e-9


def test_overnight_cues_have_no_look_ahead():
    from stockpredictor.features.longterm import global_cues

    us = pd.DataFrame({"symbol": "SP500", "date": pd.to_datetime(
        ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]), "close": [100, 101, 99, 110]})
    out = global_cues(us, pd.to_datetime(["2026-09-24"])).iloc[0]
    # long-term decision on the 24th (evening IST): only the US close of the 23rd is known
    assert abs(out["g_sp500_ret1_prev"] - (99 / 101 - 1)) < 1e-12
    # intraday on the 25th uses the 24th's row: the US close of the 24th (known by 9:15)
    assert abs(out["g_sp500_ret1_asof"] - (110 / 99 - 1)) < 1e-12
