"""F&O paper trading (intraday): the 'until close' intraday model's top picks traded as
stock futures and as bought options, each book starting every day with Rs 1 lakh.

- Futures book: near-month stock future, long for buy picks, short for sell picks. Margin
  ~MARGIN_RATE of the contract value is blocked from the book's cash; one lot per pick, best
  picks first, while the margin fits the Rs 1 lakh (usually 1-3 contracts).
- Options book: buy the at-the-money call for buy picks and the at-the-money put for sell
  picks (near-month expiry); the premium is paid from cash, so the loss is limited to it.
Both square off at 15:15 with the cash 'until close' book. Prices: Angel One (NFO).
Paper only - nothing here places real orders.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import requests

from stockpredictor.paper import engine as E

log = logging.getLogger(__name__)

FUT_BOOK = "fno_futures"
OPT_BOOK = "fno_options"
BOOKS = (FUT_BOOK, OPT_BOOK)
N_EACH_SIDE = 5          # top 5 buy picks and top 5 sell picks
MARGIN_RATE = 0.20       # SPAN + exposure margin, roughly, for stock futures
SKIP_EXPIRY_WITHIN = 2   # days: roll to next month this close to expiry (thin, erratic prices)


# --- Instruments ---------------------------------------------------------------------

@dataclass(frozen=True)
class Contract:
    name: str            # underlying, e.g. RELIANCE
    symbol: str          # tradingsymbol, e.g. RELIANCE27OCT26FUT
    token: str
    lot: int
    expiry: date
    kind: str            # FUT / CE / PE
    strike: float = 0.0


def _cache_path() -> Path:
    from stockpredictor.config import DATA_DIR

    return DATA_DIR / "cache" / "nfo_instruments.json"


def instruments(today: date) -> list[dict]:
    """Stock futures and options from Angel One's scrip master (downloaded once a day)."""
    from stockpredictor.data.angelone import SCRIP_MASTER_URL

    path = _cache_path()
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("day") == f"{today}":
            return cached["items"]
    resp = requests.get(SCRIP_MASTER_URL, timeout=60)
    resp.raise_for_status()
    items = [{k: i[k] for k in ("token", "symbol", "name", "expiry", "strike", "lotsize",
                                "instrumenttype")}
             for i in resp.json()
             if i.get("exch_seg") == "NFO" and i.get("instrumenttype") in ("FUTSTK", "OPTSTK")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"day": f"{today}", "items": items}))
    return items


def _expiry(s: str) -> date:
    return datetime.strptime(s, "%d%b%Y").date()


def _near_expiry(rows: list[dict], today: date) -> date | None:
    days = sorted({_expiry(r["expiry"]) for r in rows})
    days = [d for d in days if (d - today).days >= SKIP_EXPIRY_WITHIN]
    return days[0] if days else None


def future(items: list[dict], name: str, today: date) -> Contract | None:
    rows = [r for r in items if r["name"] == name and r["instrumenttype"] == "FUTSTK"]
    exp = _near_expiry(rows, today)
    r = next((r for r in rows if exp and _expiry(r["expiry"]) == exp), None)
    return None if r is None else Contract(name, r["symbol"], r["token"], int(float(r["lotsize"])),
                                           exp, "FUT")


def atm_option(items: list[dict], name: str, spot: float, kind: str,
               today: date) -> Contract | None:
    """The strike nearest to `spot` (Angel One strikes are in paise: /100)."""
    rows = [r for r in items if r["name"] == name and r["instrumenttype"] == "OPTSTK"
            and r["symbol"].endswith(kind)]
    exp = _near_expiry(rows, today)
    rows = [r for r in rows if exp and _expiry(r["expiry"]) == exp]
    if not rows:
        return None
    r = min(rows, key=lambda r: abs(float(r["strike"]) / 100 - spot))
    return Contract(name, r["symbol"], r["token"], int(float(r["lotsize"])), exp, kind,
                    float(r["strike"]) / 100)


# --- Costs (Indian F&O, discount broker) ---------------------------------------------------

def cost(kind: str, side: str, turnover: float) -> float:
    """One order's charges. Futures: STT 0.02% on sell; options: STT 0.1% of premium on
    sell; exchange fees, stamp duty on buy, SEBI fee, Rs 20 brokerage, 18% GST."""
    fut = kind == "FUT"
    stt = turnover * (0.0002 if fut else 0.001) if side == "sell" else 0.0
    exch = turnover * (0.0000173 if fut else 0.0003503)
    stamp = turnover * (0.00002 if fut else 0.00003) if side == "buy" else 0.0
    sebi = turnover * 0.000001
    brokerage = min(20.0, turnover * 0.0003) if fut else 20.0
    return stt + exch + stamp + sebi + (brokerage + exch + sebi) * 0.18 + brokerage


def fill(price: float, kind: str, action: str) -> float:
    """Paper fill price: crossing half the bid-ask spread. Stock options trade in Rs 0.05
    ticks with wide spreads (at least 1 tick or 1% of the premium); futures ~0.02%."""
    half = max(0.05, 0.01 * price) if kind != "FUT" else max(0.05, 0.0002 * price)
    return price + half if action == "buy" else max(0.05, price - half)


# --- Prices ----------------------------------------------------------------------------

def ltp(client, tokens: dict[str, str]) -> dict[str, float]:
    """{tradingsymbol: token} -> {tradingsymbol: last price} (NFO, 50 per request)."""
    import time

    by_token = {t: s for s, t in tokens.items()}
    out, items = {}, list(by_token)
    for i in range(0, len(items), 50):
        resp = client._require_login().getMarketData("LTP", {"NFO": items[i:i + 50]})
        for row in ((resp or {}).get("data") or {}).get("fetched", []):
            s = by_token.get(str(row.get("symbolToken")))
            if s and row.get("ltp"):
                out[s] = float(row["ltp"])
        time.sleep(0.4)
    return out


# --- Books -----------------------------------------------------------------------------

def _start_day(conn: sqlite3.Connection, capital: float, book: str) -> None:
    E.ensure_account(conn, capital, book)
    if not open_trades(conn, book):
        conn.execute("UPDATE paper_accounts SET capital = ?, cash = ? WHERE horizon = ?",
                     (capital, capital, book))


def open_trades(conn: sqlite3.Connection, book: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'open'",
                        (book,)).fetchall()


def _blocked(t) -> float:
    """Cash a position holds: margin for futures, the premium paid for options."""
    return t["qty"] * t["entry_price"] * (MARGIN_RATE if t["horizon"] == FUT_BOOK else 1.0)


def _insert(conn, book, c: Contract, side, lots, price, stamp, reason) -> str:
    qty = lots * c.lot
    price = fill(price, c.kind, "buy" if side == "long" else "sell")
    fee = cost(c.kind, "buy" if side == "long" else "sell", qty * price)
    blocked = qty * price * (MARGIN_RATE if book == FUT_BOOK else 1.0)
    conn.execute(
        "INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, costs, "
        "status, reason) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)",
        (book, c.symbol, side, qty, stamp, price, fee, reason))
    conn.execute("UPDATE paper_accounts SET cash = cash - ? WHERE horizon = ?",
                 (blocked + fee, book))
    conn.execute("INSERT OR REPLACE INTO fno_contracts VALUES (?, ?, ?, ?)",
                 (c.symbol, c.token, c.name, c.lot))
    return f"{side.upper():5} {c.symbol} {lots} lot(s) x {c.lot} @ {price:.2f}"


def open_books(conn: sqlite3.Connection, client, picks: list[tuple[str, str]],
               spot: dict[str, float], today: date, stamp: str,
               capital: float = 100_000) -> list[str]:
    """`picks`: [(symbol, 'long'|'short')] best first. Opens the futures and options books."""
    items = instruments(today)
    for b in BOOKS:
        _start_day(conn, capital, b)
    slot = capital / max(1, len(picks))
    plans = []
    for sym, side in picks:
        if sym not in spot:
            continue
        f = future(items, sym, today)
        if f:
            plans.append((FUT_BOOK, f, side))
        o = atm_option(items, sym, spot[sym], "CE" if side == "long" else "PE", today)
        if o:
            plans.append((OPT_BOOK, o, "long"))          # options are always bought
    prices = ltp(client, {c.symbol: c.token for _, c, _ in plans}) if plans else {}
    opened = []
    free = capital            # futures: one lot's margin is often Rs 50k+, so the best picks
    for book, c, side in plans:     # take one lot each while the margin fits the Rs 1 lakh
        p = prices.get(c.symbol)
        if not p:
            continue
        per_lot = c.lot * p * (MARGIN_RATE if book == FUT_BOOK else 1.0)
        if book == FUT_BOOK:
            lots = 1 if per_lot + cost("FUT", "buy", c.lot * p) <= free else 0
            free -= lots * per_lot
        else:
            lots = int(slot // per_lot)
        if lots <= 0:
            continue        # doesn't fit (option: one lot's premium > this pick's share)
        opened.append(f"[{book}] " + _insert(conn, book, c, side, lots, p, stamp,
                                             f"9:45 pick ({c.name} {side})"))
    conn.commit()
    return opened


def close_trade(conn, t, price: float, reason: str, stamp: str) -> str:
    kind = "FUT" if t["horizon"] == FUT_BOOK else "OPT"
    sign = 1 if t["side"] == "long" else -1
    price = fill(price, kind, "sell" if sign > 0 else "buy")
    fee = cost(kind, "sell" if sign > 0 else "buy", t["qty"] * price)
    gross = sign * (price - t["entry_price"]) * t["qty"]
    pnl = gross - t["costs"] - fee
    conn.execute("UPDATE paper_trades SET exit_time = ?, exit_price = ?, costs = costs + ?, "
                 "pnl = ?, status = 'closed', exit_reason = ? WHERE id = ?",
                 (stamp, price, fee, pnl, reason, t["id"]))
    conn.execute("UPDATE paper_accounts SET cash = cash + ? WHERE horizon = ?",
                 (_blocked(t) + gross - fee, t["horizon"]))
    return f"{t['side']} {t['symbol']} closed @ {price:.2f} ({reason}), P&L {pnl:+.0f}"


def tokens_for_open(conn) -> dict[str, str]:
    syms = [t["symbol"] for b in BOOKS for t in open_trades(conn, b)]
    if not syms:
        return {}
    q = ",".join("?" * len(syms))
    return dict(conn.execute(f"SELECT symbol, token FROM fno_contracts WHERE symbol IN ({q})",
                             syms).fetchall())


def square_off(conn, prices: dict[str, float], stamp: str) -> list[str]:
    out = [close_trade(conn, t, prices.get(t["symbol"], t["entry_price"]), "15:15 square-off",
                       stamp) for b in BOOKS for t in open_trades(conn, b)]
    conn.commit()
    return out


def value(conn, prices: dict[str, float], book: str) -> dict:
    acct = conn.execute("SELECT capital, cash FROM paper_accounts WHERE horizon = ?",
                        (book,)).fetchone()
    if acct is None:
        return {"capital": 0, "cash": 0, "equity": 0, "pnl": 0, "positions": 0}
    trades = open_trades(conn, book)
    held = sum(_blocked(t) + (1 if t["side"] == "long" else -1)
               * (prices.get(t["symbol"], t["entry_price"]) - t["entry_price"]) * t["qty"]
               for t in trades)
    eq = acct["cash"] + held
    return {"capital": acct["capital"], "cash": acct["cash"], "equity": eq,
            "pnl": eq - acct["capital"], "positions": len(trades)}
