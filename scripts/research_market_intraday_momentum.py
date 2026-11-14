"""Experiment E-009 (GitHub, research.yml): market intraday momentum on Nifty 50.
Gao, Han, Li & Zhou (2018, JFE) "Market intraday momentum": the first half-hour return
(previous close -> first 30 minutes) predicts the last half-hour return (S&P 500 ETF).
Test on NSE: r1 = Nifty at 9:45 / previous close - 1; r_last = close (15:30) / 15:00 - 1.
Strategy: at 15:00 go long Nifty if r1 > 0, short if r1 < 0; exit at the close.
Costs: index futures ~0.03% round trip; NIFTYBEES ETF with ₹10k ~0.15%.
Fixed rule: ACCEPT if the mean net (0.03%) return per trade > 0 with t >= 2 over the whole
period AND positive in >= 60% of calendar years. Uses Angel One 5-minute index candles:
never run in market hours (Angel One belongs to the live books then).
"""
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stockpredictor.config import load_settings
from stockpredictor.data import angelone
from stockpredictor.data import intraday as I

now = datetime.now(ZoneInfo("Asia/Kolkata"))
if now.weekday() < 5 and (9, 0) <= (now.hour, now.minute) <= (15, 35):
    sys.exit("market hours: Angel One is reserved for the live books - run after 15:35")
client = angelone.AngelDataClient(load_settings().angel)
client.login()
NIFTY = "99926000"            # Angel One token of the Nifty 50 index (NSE)
bars = I.angel_bars(client, NIFTY, date(2017, 1, 1), date.today(), interval="FIVE_MINUTE",
                    waits=(0, 2, 6, 15))
if bars.empty:
    sys.exit("no Nifty candles returned (token?)")
bars["day"] = bars["ts"].dt.date
bars["t"] = bars["ts"].dt.strftime("%H:%M")       # bar start
print(f"{len(bars):,} five-minute Nifty bars, {bars['day'].nunique()} days "
      f"({bars['day'].min()} .. {bars['day'].max()})", flush=True)


def at(g, start):                                  # price at the end of the bar starting `start`
    x = g.loc[g["t"] == start, "close"]
    return float(x.iloc[0]) if len(x) else np.nan


rows = []
prev_close = None
for d, g in bars.groupby("day"):
    p945, p1500, p1515, close = at(g, "09:40"), at(g, "14:55"), at(g, "15:10"), float(g["close"].iloc[-1])
    if prev_close and not np.isnan(p945) and not np.isnan(p1500):
        rows.append({"day": pd.Timestamp(d), "r1": p945 / prev_close - 1,
                     "r_open30": p945 / float(g["open"].iloc[0]) - 1,
                     "r_last": close / p1500 - 1, "r_last15": close / p1515 - 1,
                     "range": (g["high"].max() - g["low"].min()) / prev_close})
    prev_close = close
r = pd.DataFrame(rows)
r["vol_prev"] = r["range"].shift(1).rolling(20).mean()


def stats(x, name):
    t = x.mean() / x.std() * np.sqrt(len(x)) if x.std() else 0
    print(f"{name:42} mean {x.mean() * 1e4:+6.2f} bp, t {t:5.2f}, hit {(x > 0).mean():.0%}, n {len(x)}",
          flush=True)
    return t


for sig in ("r1", "r_open30"):
    for tgt in ("r_last", "r_last15"):
        slope = np.polyfit(r[sig], r[tgt], 1)[0]
        print(f"{sig} -> {tgt}: corr {r[sig].corr(r[tgt]):+.3f}, slope {slope:+.3f}", flush=True)
gross = np.sign(r["r1"]) * r["r_last"]
stats(gross, "gross: sign(prev close->9:45) x 15:00->close")
t_net = stats(gross - 0.0003, "net of 0.03% (index futures)")
stats(gross - 0.0015, "net of 0.15% (NIFTYBEES, Rs 10k)")
hi = r["vol_prev"] > r["vol_prev"].median()
stats((gross - 0.0003)[hi], "net 0.03%, volatile months only")
by_year = (gross - 0.0003).groupby(r["day"].dt.year).agg(["mean", "count"])
print((by_year["mean"] * 1e4).round(2).to_string(), flush=True)
pos = (by_year["mean"] > 0).mean()
ok = t_net >= 2 and (gross - 0.0003).mean() > 0 and pos >= 0.6
print(f"positive years {pos:.0%} -> {'ACCEPT' if ok else 'REJECT'}", flush=True)
