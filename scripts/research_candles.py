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

# Current baseline: 5-min summaries (r5, momentum, vol_ratio, vwap_slope)
base_model = M.train(labeled_daily)
base_sharpe = base_model.metrics.get("sharpe", 0)

print(f"Baseline (current 5-min): {len(labeled_daily):,} weeks, Sharpe {base_sharpe:.2f}", flush=True)

# Try to load 1-minute bars for aggregation tests
try:
    import sqlite3
    conn = sqlite3.connect(d / "stockpredictor.db")
    bars_1min = pd.read_sql(
        "SELECT symbol, ts, open, high, low, close, volume FROM intraday_bars ORDER BY ts",
        conn, parse_dates=["ts"])
    conn.close()
    has_bars = not bars_1min.empty
except Exception:
    has_bars = False

if has_bars:
    print(f"Loaded {len(bars_1min):,} 1-min bars; testing frequencies...", flush=True)
    freqs = {"1min": "1min", "5min": "5min", "15min": "15min", "1hour": "1H"}
    rows = []
    for name, freq in freqs.items():
        try:
            agg = (bars_1min.set_index("ts")
                   .groupby("symbol")[["open", "high", "low", "close", "volume"]]
                   .resample(freq)
                   .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                   .reset_index()
                   .dropna())
            # Would train separate model and backtest here
            # For now: report structure
            print(f"{name:8} {len(agg):6,} bars  → would build features r_{freq[:2]} and backtest", flush=True)
            rows.append({"freq": name, "bars": len(agg), "status": "ready"})
        except Exception as e:
            print(f"{name:8} failed: {e}")
    print(f"\nAcceptance rule: Sharpe >= {base_sharpe:.2f} AND paired t >= 2 on daily returns")
else:
    print("No 1-minute bars available (first run uses 5-min summaries only)")
    print("To collect 1-min bars: activate intraday-collect on GitHub with --period 7d")
    print("Then re-run E-006 to test: 1-min (finer), 5-min (baseline), 15-min (smoother), 1-hour (trend)")
