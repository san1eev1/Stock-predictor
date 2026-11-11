"""NSE IPO subscription and live trading. IPOs open at 9:15 on listing day; track and subscribe."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import requests

# For IPO trading to work, you need:
# 1. IPO name (e.g., "ZOMATO"), lot size, IPO date
# 2. Live price at 9:15 (NSE pre-open or first print)
# 3. Quick entry: buy market order at open within first 2 mins
# 4. Exit rules: % gain or time-based (hold 1-5 mins)
#
# What to learn before IPO trading:
# - IPO allocation (oversubscription ratio tells you volatility)
# - GMP (grey market premium) before listing (investor sentiment)
# - Listing day sentiment (FII/DII, retail appetite)
# - First day volume: if volume is 2-5x lot size = liquid
# - Recent listings in same sector (timing, momentum)
# - Quiet period (trading ban before listing; learn during this time)

FILE = "ipo_watchlist.csv"
COLS = ["symbol", "ipo_date", "lot_size", "price_band_low", "price_band_high", "gmp_estimate",
        "listing_date", "alloc_ratio", "first_day_vol", "first_day_return", "notes"]


def load(store_dir: Path) -> pd.DataFrame:
    """Load the IPO watchlist from the store."""
    path = Path(store_dir) / FILE
    if path.exists():
        return pd.read_csv(path, parse_dates=["ipo_date", "listing_date"])
    return pd.DataFrame(columns=COLS)


def save(store_dir: Path, df: pd.DataFrame) -> None:
    """Save the IPO watchlist."""
    path = Path(store_dir) / FILE
    df.to_csv(path, index=False, date_format="%Y-%m-%d")


def add_ipo(store_dir: Path, symbol: str, ipo_date: str, lot_size: int,
            price_band_low: float, price_band_high: float, gmp_estimate: float = 0,
            notes: str = "") -> None:
    """Add an IPO to the watchlist (before listing date)."""
    df = load(store_dir)
    new = pd.DataFrame([{
        "symbol": symbol, "ipo_date": pd.to_datetime(ipo_date), "lot_size": lot_size,
        "price_band_low": price_band_low, "price_band_high": price_band_high,
        "gmp_estimate": gmp_estimate, "listing_date": None, "alloc_ratio": None,
        "first_day_vol": None, "first_day_return": None, "notes": notes
    }])
    df = pd.concat([df, new]).drop_duplicates(subset=["symbol"], keep="last")
    save(store_dir, df)


def record_listing(store_dir: Path, symbol: str, listing_price: float, alloc_ratio: float,
                   first_day_vol: int, first_day_return: float) -> None:
    """Record IPO listing data after the IPO lists."""
    df = load(store_dir)
    mask = df["symbol"] == symbol
    if mask.any():
        df.loc[mask, "listing_date"] = datetime.now().date()
        df.loc[mask, "alloc_ratio"] = alloc_ratio
        df.loc[mask, "first_day_vol"] = first_day_vol
        df.loc[mask, "first_day_return"] = first_day_return
        save(store_dir, df)


def ipos_listing_today(store_dir: Path) -> pd.DataFrame:
    """IPOs that list today (listing_date == today)."""
    df = load(store_dir)
    today = pd.Timestamp.now().date()
    return df[df["listing_date"] == today] if not df.empty else pd.DataFrame(columns=COLS)


def learning_checklist(ipo: dict) -> dict:
    """What to research about an IPO before trading it."""
    return {
        "symbol": ipo.get("symbol", ""),
        "ipo_date": ipo.get("ipo_date", ""),
        "checks": {
            "allocation_ratio": {
                "what": "Oversubscription ratio (Apps / Available lots)",
                "why": "High ratio (>10x) = more retail demand = more first-day volatility",
                "source": "NSE IPO details",
                "ideal": "5-15x (high demand, but not insane)"
            },
            "gmp_grey_market_premium": {
                "what": "GMP estimate (trading price before listing)",
                "why": "GMP > listing price = institutional appetite, likely pop; GMP < listing = caution",
                "source": "IPO forums, Twitter, news",
                "ideal": "5-15% above price band"
            },
            "lot_size": {
                "what": "Lot size in shares",
                "why": "Smaller lot = easier to buy multiple lots; larger = fewer positions",
                "source": "NSE IPO prospectus",
                "ideal": "> 1 (buyable; avoid tiny lots like 1 share)"
            },
            "sector_momentum": {
                "what": "Recent listings in same sector; index performance",
                "why": "Sector rally helps IPO pop; opposite can kill it",
                "source": "NSE news, recent IPO returns",
                "ideal": "Sector positive last 3 months"
            },
            "quiet_period_sentiment": {
                "what": "Investor sentiment during quiet period (no marketing before listing)",
                "why": "During quiet period, learn from comparables: similar IPOs, peer margins",
                "source": "Analyst notes, peer comparisons, news",
                "ideal": "Understand business model, growth, valuation"
            },
            "first_print_volume": {
                "what": "Volume in first 5 minutes after listing",
                "why": "Volume > lot_size * 5 = liquid, easy to exit. Volume < lot_size * 2 = illiquid",
                "source": "NSE live during listing day 09:15-09:20",
                "ideal": "> lot_size * 5 shares in first 5 mins"
            },
        },
        "trading_plan": {
            "entry": "Market order at 09:15 (or within first 30 secs if GMP positive)",
            "size": "1-2 lots (test first lot, add if conviction is high)",
            "exit_gain": "Exit if +5% in first 2 mins (cash in pop)",
            "exit_time": "Exit if 5 mins without +5% (volatility fade, take small gain)",
            "loss_limit": "-3% and exit (IPO fades = time to go)",
        },
        "red_flags": [
            "Allocation ratio < 3x (low demand, likely to fade)",
            "GMP < listing price (institutional selling signal)",
            "Sector in downtrend (headwind)",
            "First-print volume < lot_size * 2 (illiquid, hard to exit at mark)",
            "Large lot size (> 1000 shares) at high price band (illiquid)",
        ]
    }
