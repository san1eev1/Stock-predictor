"""Your real trades (entered manually from Groww): holdings, P&L, stop-losses."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

HORIZONS = ("longterm", "intraday")
DEFAULT_STOP_LOSS = 0.15


def add_trade(conn: sqlite3.Connection, horizon: str, symbol: str, side: str, qty: int,
              price: float, trade_date: str, charges: float = 0.0, notes: str = "") -> int:
    if horizon not in HORIZONS or side not in ("buy", "sell"):
        raise ValueError("invalid horizon or side")
    if qty <= 0 or price <= 0:
        raise ValueError("quantity and price must be positive")
    symbol = symbol.strip().upper()
    if side == "sell":
        held = holdings(conn, horizon)
        have = int(held.loc[held["symbol"] == symbol, "qty"].sum()) if not held.empty else 0
        if qty > have:
            raise ValueError(f"cannot sell {qty} {symbol}: only {have} held")
    cur = conn.execute(
        "INSERT INTO portfolio_trades (horizon, symbol, side, qty, price, charges, trade_date, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (horizon, symbol, side, qty, price, charges, trade_date, notes))
    conn.commit()
    return cur.lastrowid


def delete_trade(conn: sqlite3.Connection, trade_id: int) -> None:
    conn.execute("DELETE FROM portfolio_trades WHERE id = ?", (trade_id,))
    conn.commit()


def trades(conn: sqlite3.Connection, horizon: str) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM portfolio_trades WHERE horizon = ? ORDER BY trade_date, id",
                       conn, params=(horizon,))


def holdings(conn: sqlite3.Connection, horizon: str) -> pd.DataFrame:
    """Average-cost holdings. Buy charges are added to cost; sell charges reduce proceeds."""
    state: dict[str, dict] = {}
    for t in trades(conn, horizon).itertuples():
        s = state.setdefault(t.symbol, {"qty": 0, "cost": 0.0, "realized": 0.0, "charges": 0.0})
        s["charges"] += t.charges
        if t.side == "buy":
            s["qty"] += t.qty
            s["cost"] += t.qty * t.price + t.charges
        else:
            avg = s["cost"] / s["qty"] if s["qty"] else 0
            s["realized"] += t.qty * t.price - t.charges - avg * t.qty
            s["cost"] -= avg * t.qty
            s["qty"] -= t.qty
    rows = [{"symbol": k, "qty": v["qty"], "avg_cost": v["cost"] / v["qty"] if v["qty"] else 0,
             "invested": v["cost"], "realized": v["realized"], "charges": v["charges"]}
            for k, v in state.items()]
    return pd.DataFrame(rows, columns=["symbol", "qty", "avg_cost", "invested", "realized", "charges"])


def stop_loss_pct(conn: sqlite3.Connection, horizon: str, symbol: str) -> float:
    for key in (f"pf_sl_{horizon}_{symbol}", f"pf_sl_{horizon}_default"):
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        if row:
            return float(row[0])
    return DEFAULT_STOP_LOSS


def set_stop_loss(conn: sqlite3.Connection, horizon: str, symbol: str | None, pct: float) -> None:
    key = f"pf_sl_{horizon}_{symbol or 'default'}"
    conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, str(pct)))
    conn.commit()


@dataclass
class Summary:
    table: pd.DataFrame
    invested: float
    value: float
    unrealized: float
    realized: float

    @property
    def total_pnl(self) -> float:
        return self.unrealized + self.realized


def summary(conn: sqlite3.Connection, horizon: str, prices: dict[str, float]) -> Summary:
    h = holdings(conn, horizon)
    realized = float(h["realized"].sum()) if not h.empty else 0.0
    h = h[h["qty"] > 0].copy()
    h["price"] = h["symbol"].map(prices).fillna(h["avg_cost"])
    h["value"] = h["qty"] * h["price"]
    h["unrealized"] = h["value"] - h["invested"]
    h["unrealized_pct"] = h["unrealized"] / h["invested"]
    h["stop_loss"] = [r.avg_cost * (1 - stop_loss_pct(conn, horizon, r.symbol))
                      for r in h.itertuples()]
    h["weight"] = h["value"] / h["value"].sum() if len(h) else []
    return Summary(h, float(h["invested"].sum()), float(h["value"].sum()),
                   float(h["unrealized"].sum()), realized)
