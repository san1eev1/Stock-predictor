"""IPO analysis: score incoming IPOs and recommend buy/not buy with reasoning."""

from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IPOScore:
    """IPO tradability score (0-100). High = good to trade."""
    symbol: str
    listing_date: str
    allocation_ratio: float  # subscription times
    gmp_estimate: float      # grey market premium %
    lot_size: int
    sector: str
    sector_momentum: str     # "strong", "neutral", "weak"
    price_band: tuple        # (low, high)

    # Scores (0-10 each)
    demand_score: float      # from alloc ratio
    sentiment_score: float   # from GMP
    liquidity_score: float   # from lot size
    sector_score: float      # from sector momentum

    # Final decision
    overall_score: float     # weighted avg 0-100
    recommendation: str      # "BUY", "CAUTION", "SKIP"
    reasons: list[str]
    red_flags: list[str]


def score_allocation(ratio: float) -> tuple[float, str]:
    """Score alloc ratio. High = demand."""
    if ratio < 3:
        return 2, "Low demand (ratio < 3x)"
    elif ratio < 10:
        return 6, "Moderate demand (3-10x)"
    elif ratio < 30:
        return 8, "High demand (10-30x)"
    else:
        return 10, "Extreme demand (> 30x) — expect volatility"


def score_gmp(gmp_pct: float, price_band: tuple[float, float]) -> tuple[float, str]:
    """Score GMP. Positive GMP = investor appetite."""
    low, high = price_band
    if gmp_pct < -5:
        return 1, f"Negative GMP ({gmp_pct:.1f}%) — institutions selling"
    elif gmp_pct < 0:
        return 3, f"Slightly negative GMP ({gmp_pct:.1f}%)"
    elif gmp_pct == 0:
        return 5, "Neutral GMP — wait for listing sentiment"
    elif gmp_pct < 10:
        return 7, f"Positive GMP ({gmp_pct:.1f}%) — retail interest"
    else:
        return 9, f"Strong GMP ({gmp_pct:.1f}%) — high expectations, possible fade after pop"


def score_lot_size(lot: int, price_band: tuple[float, float]) -> tuple[float, str]:
    """Score lot size. Smaller = more liquid."""
    low, high = price_band
    typical_cost = lot * high  # worst case

    if typical_cost > 200_000:
        return 2, f"Large lot (₹{typical_cost:,.0f} per lot) — illiquid, hard to exit"
    elif typical_cost > 100_000:
        return 5, f"Medium lot (₹{typical_cost:,.0f}) — manageable"
    elif typical_cost > 50_000:
        return 7, f"Small lot (₹{typical_cost:,.0f}) — liquid, easy to scale"
    else:
        return 9, f"Tiny lot (₹{typical_cost:,.0f}) — very liquid"


def score_sector(sector: str, momentum: str) -> tuple[float, str]:
    """Score sector. Trending up = tailwind."""
    scores = {"strong": 8, "neutral": 5, "weak": 2}
    score = scores.get(momentum.lower(), 5)
    reason = {
        "strong": f"{sector} sector rallying — tailwind for IPO",
        "neutral": f"{sector} sector mixed — no strong tailwind",
        "weak": f"{sector} sector declining — headwind for IPO"
    }.get(momentum.lower(), "Sector momentum unknown")
    return float(score), reason


def analyze_ipo(symbol: str, listing_date: str, allocation_ratio: float,
                gmp_estimate: float, lot_size: int, sector: str, sector_momentum: str,
                price_band: tuple[float, float]) -> IPOScore:
    """Score an IPO (0-100) and recommend buy/caution/skip."""

    # Score each factor
    demand_score, demand_reason = score_allocation(allocation_ratio)
    sentiment_score, sentiment_reason = score_gmp(gmp_estimate, price_band)
    liquidity_score, liquidity_reason = score_lot_size(lot_size, price_band)
    sector_score, sector_reason = score_sector(sector, sector_momentum)

    # Weighted average (0-100)
    weights = {"demand": 0.25, "sentiment": 0.30, "liquidity": 0.25, "sector": 0.20}
    overall = (demand_score * weights["demand"] +
               sentiment_score * weights["sentiment"] +
               liquidity_score * weights["liquidity"] +
               sector_score * weights["sector"]) * 10

    # Decision rules
    reasons = [demand_reason, sentiment_reason, liquidity_reason, sector_reason]
    red_flags = []

    if allocation_ratio < 3:
        red_flags.append(f"Low subscription ({allocation_ratio:.1f}x) — may fail to sustain")
    if gmp_estimate < -5:
        red_flags.append(f"Negative GMP ({gmp_estimate:.1f}%) — institutional selling")
    if lot_size * price_band[1] > 200_000:
        red_flags.append(f"Large lot cost (₹{lot_size * price_band[1]:,.0f}) — hard to exit")
    if sector_momentum.lower() == "weak":
        red_flags.append(f"{sector} sector weak — headwind")

    # Recommend
    if overall >= 70 and len(red_flags) == 0:
        rec = "BUY"
    elif overall >= 50 and gmp_estimate > 0:
        rec = "CAUTION"
    else:
        rec = "SKIP"

    return IPOScore(
        symbol=symbol,
        listing_date=listing_date,
        allocation_ratio=allocation_ratio,
        gmp_estimate=gmp_estimate,
        lot_size=lot_size,
        sector=sector,
        sector_momentum=sector_momentum,
        price_band=price_band,
        demand_score=demand_score,
        sentiment_score=sentiment_score,
        liquidity_score=liquidity_score,
        sector_score=sector_score,
        overall_score=overall,
        recommendation=rec,
        reasons=reasons,
        red_flags=red_flags
    )


# Example
if __name__ == "__main__":
    # Hypothetical: Swiggy IPO (if it exists, this is an example)
    ipo = analyze_ipo(
        symbol="SWIGGY",
        listing_date="2024-11-15",
        allocation_ratio=12.5,
        gmp_estimate=8.5,
        lot_size=50,
        sector="Consumer",
        sector_momentum="strong",
        price_band=(371, 412)
    )
    print(f"\n{ipo.symbol} ({ipo.listing_date})")
    print(f"Recommendation: {ipo.recommendation} (Score {ipo.overall_score:.0f}/100)")
    print(f"\nFactors:")
    for reason in ipo.reasons:
        print(f"  • {reason}")
    if ipo.red_flags:
        print(f"\nRed flags:")
        for flag in ipo.red_flags:
            print(f"  ⚠ {flag}")
