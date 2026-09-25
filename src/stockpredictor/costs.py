"""Approximate Indian equity delivery (CNC) trading costs.

Rates change over time; these defaults approximate a discount broker such
as Groww. They are editable so paper P&L can match your contract notes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeliveryCosts:
    brokerage_pct: float = 0.001        # 0.1% ...
    brokerage_min: float = 5.0          # ... min Rs 5
    brokerage_max: float = 20.0         # ... max Rs 20 per order
    stt_pct: float = 0.001              # 0.1% on buy and sell
    exchange_pct: float = 0.0000297     # NSE transaction charge
    sebi_pct: float = 0.000001          # Rs 10 per crore
    stamp_buy_pct: float = 0.00015      # 0.015% on buy
    gst_pct: float = 0.18               # on brokerage + exchange + SEBI
    dp_per_sell: float = 20.0           # depository charge per scrip sold
    slippage_pct: float = 0.0005        # 0.05% execution slippage

    def cost(self, side: str, value: float) -> float:
        """Total charges in rupees for one order of `value` rupees."""
        if value <= 0:
            return 0.0
        brokerage = min(self.brokerage_max, max(self.brokerage_min, value * self.brokerage_pct))
        exchange, sebi = value * self.exchange_pct, value * self.sebi_pct
        total = (brokerage + value * self.stt_pct + exchange + sebi
                 + self.gst_pct * (brokerage + exchange + sebi)
                 + value * self.slippage_pct)
        if side == "buy":
            total += value * self.stamp_buy_pct
        else:
            total += self.dp_per_sell
        return round(total, 2)


DEFAULT_COSTS = DeliveryCosts()


@dataclass(frozen=True)
class IntradayCosts:
    """Approximate Indian intraday (MIS) equity charges per order."""
    brokerage_pct: float = 0.001        # 0.1% ...
    brokerage_min: float = 5.0
    brokerage_max: float = 20.0         # ... max Rs 20 per order
    stt_sell_pct: float = 0.00025       # 0.025% on the sell side
    exchange_pct: float = 0.0000297
    sebi_pct: float = 0.000001
    stamp_buy_pct: float = 0.00003      # 0.003% on the buy side
    gst_pct: float = 0.18
    slippage_pct: float = 0.0005        # 0.05% per order

    def cost(self, side: str, value: float, slippage_pct: float | None = None) -> float:
        """slippage_pct: this order's own slippage (e.g. higher for a volatile stock)."""
        if value <= 0:
            return 0.0
        brokerage = min(self.brokerage_max, max(self.brokerage_min, value * self.brokerage_pct))
        exchange, sebi = value * self.exchange_pct, value * self.sebi_pct
        total = (brokerage + exchange + sebi + self.gst_pct * (brokerage + exchange + sebi)
                 + value * (self.slippage_pct if slippage_pct is None else slippage_pct))
        total += value * (self.stamp_buy_pct if side == "buy" else self.stt_sell_pct)
        return round(total, 2)


DEFAULT_INTRADAY_COSTS = IntradayCosts()
