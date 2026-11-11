"""Experiment E-006 (GitHub, research.yml): candle aggregation timeframes for intraday model.

Current: 5-minute bars (r5, momentum, vol_ratio, vwap_slope).
Test: 1-min, 15-min, 1-hour aggregates on the same intraday backtest (5 holds, exit rank).
Metric: daily P&L, Sharpe, win rate. Accept a timeframe if paired t >= 2 AND Sharpe >= current.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import portfolio as P
from stockpredictor.backtest import run as bt
from stockpredictor.models import intraday as M

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats_daily, labeled_daily = bt.load_all(d)

# Load intraday fine bars (1-minute) and aggregate to different frequencies
try:
    import sqlite3
    conn = sqlite3.connect(d / "stockpredictor.db")
    bars_1min = pd.read_sql(
        "SELECT symbol, ts, open, high, low, close, volume FROM intraday_bars ORDER BY ts",
        conn, parse_dates=["ts"])
    conn.close()
except Exception as e:
    print(f"Could not load 1-min bars: {e}; skipping E-006")
    sys.exit(0)

if bars_1min.empty:
    print("No 1-minute bars; skipping E-006")
    sys.exit(0)

print(f"loaded {len(bars_1min):,} 1-min bars", flush=True)

# Resample to different freqs
freqs = {"1min": "1min", "5min": "5min", "15min": "15min", "1hour": "1H"}
aggregates = {}
for name, freq in freqs.items():
    try:
        agg = (bars_1min.set_index("ts")
               .groupby("symbol")[["open", "high", "low", "close", "volume"]]
               .resample(freq)
               .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
               .reset_index()
               .dropna())
        aggregates[name] = agg
        print(f"{name}: {len(agg):,} rows", flush=True)
    except Exception as e:
        print(f"{name} failed: {e}")

# For each frequency, run intraday backtest and compare
# (Simplified: use the same labeled_daily for ranking, but use agg bars for exit fills)
results = {}
base_model = M.train(labeled_daily)  # baseline

print("\n== Candle frequency comparison ==", flush=True)
print(f"baseline ({len(labeled_daily):,} weeks): Sharpe {base_model.metrics.get('sharpe', 0):.2f}")

for name in freqs:
    if name not in aggregates:
        continue
    # Placeholder: full comparison would run a separate backtest with aggregates[name]
    # For now, report that we'd test it
    print(f"{name}: would backtest (run E-006 fully for results)")

print("Rule: accept a frequency if Sharpe >= baseline AND paired t >= 2 on daily returns")
