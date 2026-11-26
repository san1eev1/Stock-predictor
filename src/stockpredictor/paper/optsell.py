"""Option selling paper books: several hedged Nifty weekly strategies run side by side on
paper, each with its own capital, so live results show which approach actually earns.

Evidence behind them (see the Learn page and research/technique_catalog.md):
- Index options are priced for more volatility than usually arrives (the volatility risk
  premium: Coval & Shumway 2001, Bakshi & Kapadia 2003; Nifty VIX ~2-4 pts above realized).
- The edge is small and fragile: costs and tail days can erase it; same-day condors entered
  in the first two hours did worst (Option Alpha, 25k trades); a few days decide the year.
So every book is hedged (bought wings cap the worst loss) and differs in one idea:

  fly_920      AlgoTest's 9:20 straddle as an iron fly: 25% stop per short leg, stop to cost
  fly_1100     the same, entered at 11:00 (do the opening hours hurt?)
  condor       short strikes ~1% away, wings 2.5%: take profit at 50% of the credit, close
               if the loss reaches the credit (tastytrade-style management)
  condor_vrp   the condor only when India VIX is above Nifty's 20-day realized volatility
  fly_expiry   iron fly on expiry day (Tuesday) only, at 9:45 (most time decay)

Every book: Nifty weekly options, nearest expiry; as many lots as the capital covers (max
loss + 10%); skip when India VIX > 22; everything closes at 15:15. Prices from Angel One,
paper fills cross half the spread, Indian option costs per order (paper/fno.cost).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import requests

from stockpredictor.paper import engine as E
from stockpredictor.paper import fno as FO

log = logging.getLogger(__name__)

UNDERLYING = "NIFTY"
INDEX_TOKEN = "99926000"           # Nifty 50 on Angel One (NSE)
STRIKE_STEP = 50
EXIT = time(15, 15)
MAX_VIX = 22.0
MAX_LOTS = 4


@dataclass(frozen=True)
class Strategy:
    book: str
    label: str
    entry: time
    short_otm: float = 0.0          # 0 = at the money (straddle / fly); 0.01 = 1% away
    wing: float = 0.02              # bought wings this far from spot
    leg_stop: float | None = 0.25   # stop each short leg this much above its sell price
    to_cost: bool = True            # after a leg stops, the other short's stop moves to cost
    loss_stop: float | None = None  # close all when the loss reaches this x the credit
    take_profit: float | None = None  # close all at this share of the credit
    expiry_only: bool = False
    vrp_filter: bool = False        # only when India VIX > 20-day realized vol


STRATEGIES = (
    Strategy("os_fly_920", "9:20 iron fly", time(9, 20)),
    Strategy("os_fly_1100", "11:00 iron fly", time(11, 0)),
    Strategy("os_condor", "Iron condor, 50% profit target", time(9, 30), short_otm=0.01,
             wing=0.025, leg_stop=None, to_cost=False, loss_stop=1.0, take_profit=0.5),
    Strategy("os_condor_vrp", "Iron condor when VIX > realized", time(9, 30), short_otm=0.01,
             wing=0.025, leg_stop=None, to_cost=False, loss_stop=1.0, take_profit=0.5,
             vrp_filter=True),
    Strategy("os_fly_expiry", "Expiry-day iron fly", time(9, 45), expiry_only=True),
)
BOOKS = tuple(s.book for s in STRATEGIES)
ENTRY_WINDOW = 20                   # minutes after a strategy's entry time it may still enter


@dataclass
class Leg:
    contract: FO.Contract
    side: str                      # "short" (sold) or "long" (bought wing)


def _cache_path() -> Path:
    from stockpredictor.config import DATA_DIR

    return DATA_DIR / "cache" / "nifty_options.json"


def instruments(today: date) -> list[dict]:
    """Nifty index options from Angel One's scrip master (downloaded once a day)."""
    from stockpredictor.data.angelone import SCRIP_MASTER_URL

    path = _cache_path()
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("day") == f"{today}":
            return cached["items"]
    resp = requests.get(SCRIP_MASTER_URL, timeout=60)
    resp.raise_for_status()
    items = [{k: i[k] for k in ("token", "symbol", "name", "expiry", "strike", "lotsize")}
             for i in resp.json()
             if i.get("exch_seg") == "NFO" and i.get("instrumenttype") == "OPTIDX"
             and i.get("name") == UNDERLYING]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"day": f"{today}", "items": items}))
    return items


def option(items: list[dict], strike: float, kind: str, today: date) -> FO.Contract | None:
    """Nearest weekly expiry (today included: expiry-day selling is part of the strategy)."""
    rows = [r for r in items if r["symbol"].endswith(kind)]
    exps = sorted({FO._expiry(r["expiry"]) for r in rows if FO._expiry(r["expiry"]) >= today})
    if not exps:
        return None
    rows = [r for r in rows if FO._expiry(r["expiry"]) == exps[0]]
    r = min(rows, key=lambda r: abs(float(r["strike"]) / 100 - strike), default=None)
    if r is None or abs(float(r["strike"]) / 100 - strike) > STRIKE_STEP:
        return None
    return FO.Contract(UNDERLYING, r["symbol"], r["token"], int(float(r["lotsize"])), exps[0],
                       kind, float(r["strike"]) / 100)


def _strike(x: float) -> float:
    return round(x / STRIKE_STEP) * STRIKE_STEP


def legs_for(items: list[dict], spot: float, today: date,
             st: Strategy = STRATEGIES[0]) -> list[Leg] | None:
    """Short call + short put (at the money, or st.short_otm away) and bought wings."""
    atm = _strike(spot)
    off = _strike(spot * st.short_otm) if st.short_otm else 0
    w = max(STRIKE_STEP, _strike(spot * st.wing))
    want = [(atm + off, "CE", "short"), (atm - off, "PE", "short"),
            (atm + max(w, off + STRIKE_STEP), "CE", "long"),
            (atm - max(w, off + STRIKE_STEP), "PE", "long")]
    legs = []
    for strike, kind, side in want:
        c = option(items, strike, kind, today)
        if c is None:
            return None
        legs.append(Leg(c, side))
    return legs


def max_loss_per_lot(legs: list[Leg], prices: dict[str, float]) -> float:
    """Wing distance from the short strike minus the net credit, per lot (worse side)."""
    lot = legs[0].contract.lot
    width = max(abs(legs[2].contract.strike - legs[0].contract.strike),
                abs(legs[1].contract.strike - legs[3].contract.strike))
    credit = sum((1 if l.side == "short" else -1) * prices[l.contract.symbol] for l in legs)
    return max(0.0, (width - credit) * lot)


# --- Books ------------------------------------------------------------------------------

def open_legs(conn, book: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'open'",
                        (book,)).fetchall()


def entered(conn, book: str, day: str) -> bool:
    return conn.execute("SELECT 1 FROM paper_trades WHERE horizon = ? AND entry_time LIKE ? "
                        "LIMIT 1", (book, f"{day}%")).fetchone() is not None


def enter(conn, st: Strategy, legs: list[Leg], prices: dict[str, float], stamp: str,
          capital: float) -> list[str]:
    if any(l.contract.symbol not in prices for l in legs):
        return []
    E.ensure_account(conn, capital, st.book)
    risk = max_loss_per_lot(legs, prices) * 1.10
    lots = min(MAX_LOTS, int(capital // risk)) if risk > 0 else 0
    if lots <= 0:
        log.info("%s: one lot's risk %.0f exceeds the capital", st.label, risk)
        return []
    out = []
    for l in legs:
        c, qty = l.contract, lots * l.contract.lot
        px = FO.fill(prices[c.symbol], c.kind, "sell" if l.side == "short" else "buy")
        fee = FO.cost(c.kind, "sell" if l.side == "short" else "buy", qty * px)
        stop = px * (1 + st.leg_stop) if (l.side == "short" and st.leg_stop) else None
        conn.execute(
            "INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
            "costs, status, reason, stop_loss) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
            (st.book, c.symbol, l.side, qty, stamp, px, fee, f"{st.label} {c.kind}", stop))
        conn.execute("INSERT OR REPLACE INTO fno_contracts VALUES (?, ?, ?, ?)",
                     (c.symbol, c.token, c.name, c.lot))
        out.append(f"[{st.label}] {'SELL' if l.side == 'short' else 'BUY '} {c.symbol} "
                   f"{qty} @ {px:.2f}")
    conn.execute("UPDATE paper_accounts SET capital = ?, cash = ? WHERE horizon = ?",
                 (capital, capital, st.book))
    conn.commit()
    return out


def close_leg(conn, t, price: float, reason: str, stamp: str) -> str:
    short = t["side"] == "short"
    px = FO.fill(price, t["symbol"][-2:], "buy" if short else "sell")
    fee = FO.cost(t["symbol"][-2:], "buy" if short else "sell", t["qty"] * px)
    gross = (t["entry_price"] - px if short else px - t["entry_price"]) * t["qty"]
    pnl = gross - t["costs"] - fee
    conn.execute("UPDATE paper_trades SET exit_time = ?, exit_price = ?, costs = costs + ?, "
                 "pnl = ?, status = 'closed', exit_reason = ? WHERE id = ?",
                 (stamp, px, fee, pnl, reason, t["id"]))
    return f"{t['symbol']} closed @ {px:.2f} ({reason}), P&L {pnl:+.0f}"


def check_stops(conn, st: Strategy, prices: dict[str, float], stamp: str) -> list[str]:
    """Per-leg stops (with the stopped leg's wing, then the other short's stop to cost), or
    whole-position rules: take profit / loss stop as a share of the credit."""
    legs = open_legs(conn, st.book)
    if not legs:
        return []
    out: list[str] = []
    if st.loss_stop or st.take_profit:
        credit = sum((1 if t["side"] == "short" else -1) * t["entry_price"] * t["qty"] for t in legs)
        now = sum((1 if t["side"] == "short" else -1) * prices.get(t["symbol"], t["entry_price"])
                  * t["qty"] for t in legs)
        gain = credit - now                              # before costs
        reason = ("profit target" if st.take_profit and gain >= st.take_profit * credit else
                  "loss stop" if st.loss_stop and -gain >= st.loss_stop * credit else None)
        if reason and credit > 0:
            out = [close_leg(conn, t, prices.get(t["symbol"], t["entry_price"]), reason, stamp)
                   for t in legs]
            conn.commit()
            return out
    closed: set = set()
    for t in legs:
        p = prices.get(t["symbol"])
        if (t["id"] in closed or t["side"] != "short" or p is None or t["stop_loss"] is None
                or p < t["stop_loss"]):
            continue
        kind = t["symbol"][-2:]
        out.append(close_leg(conn, t, p, "stop-loss", stamp))
        closed.add(t["id"])
        for w in legs:                                   # its wing goes with it
            if w["side"] == "long" and w["symbol"][-2:] == kind and w["id"] not in closed:
                out.append(close_leg(conn, w, prices.get(w["symbol"], w["entry_price"]),
                                     "wing of a stopped leg", stamp))
                closed.add(w["id"])
        if st.to_cost:                                   # the other short: stop to cost
            for o in legs:
                if o["side"] == "short" and o["symbol"][-2:] != kind and o["id"] not in closed:
                    conn.execute("UPDATE paper_trades SET stop_loss = ? WHERE id = ?",
                                 (o["entry_price"], o["id"]))
    conn.commit()
    return out


def square_off(conn, book: str, prices: dict[str, float], stamp: str) -> list[str]:
    out = [close_leg(conn, t, prices.get(t["symbol"], t["entry_price"]), "15:15 square-off",
                     stamp) for t in open_legs(conn, book)]
    conn.commit()
    return out


def value(conn, book: str, prices: dict[str, float], day: str) -> dict:
    """The day's result: closed legs' P&L plus open legs at the latest prices (after costs)."""
    rows = conn.execute("SELECT * FROM paper_trades WHERE horizon = ? AND entry_time LIKE ?",
                        (book, f"{day}%")).fetchall()
    pnl, credit = 0.0, 0.0
    for t in rows:
        short = t["side"] == "short"
        credit += (1 if short else -1) * t["entry_price"] * t["qty"]
        if t["status"] == "closed":
            pnl += t["pnl"]
        else:
            p = prices.get(t["symbol"], t["entry_price"])
            pnl += (t["entry_price"] - p if short else p - t["entry_price"]) * t["qty"] - t["costs"]
    return {"legs": len(rows), "open": sum(t["status"] == "open" for t in rows),
            "credit": credit, "pnl": pnl}


def history(conn) -> "pd.DataFrame":
    """Daily P&L per strategy (closed days), for the leaderboard."""
    import pandas as pd

    q = ",".join("?" * len(BOOKS))
    d = pd.read_sql(f"SELECT horizon AS book, substr(entry_time, 1, 10) AS day, SUM(pnl) AS pnl "
                    f"FROM paper_trades WHERE horizon IN ({q}) AND status = 'closed' "
                    f"GROUP BY 1, 2 ORDER BY 2", conn, params=BOOKS)
    return d


# --- Live --------------------------------------------------------------------------------

def index_spot(client) -> float | None:
    resp = client._require_login().getMarketData("LTP", {"NSE": [INDEX_TOKEN]})
    rows = ((resp or {}).get("data") or {}).get("fetched", [])
    return float(rows[0]["ltp"]) if rows and rows[0].get("ltp") else None


def tokens_for_open(conn) -> dict[str, str]:
    syms = [t["symbol"] for b in BOOKS for t in open_legs(conn, b)]
    if not syms:
        return {}
    q = ",".join("?" * len(syms))
    return dict(conn.execute(f"SELECT symbol, token FROM fno_contracts WHERE symbol IN ({q})",
                             syms).fetchall())


def tick(conn, client, now: datetime, capital: float, vix: float | None,
         realized: float | None = None) -> list[str]:
    """Called every market minute by the live monitor: entries, stops, 15:15 exit for every
    strategy. realized: Nifty's 20-day realized volatility in VIX points."""
    day, stamp, t = f"{now:%Y-%m-%d}", now.strftime("%Y-%m-%d %H:%M"), now.time()
    out: list[str] = []
    items, spot = None, None
    for st in STRATEGIES:
        end = (datetime.combine(now.date(), st.entry) + timedelta(minutes=ENTRY_WINDOW)).time()
        if not (st.entry <= t <= end) or entered(conn, st.book, day) or \
                conn.execute("SELECT 1 FROM app_settings WHERE key = ?",
                             (f"os_skip_{st.book}_{day}",)).fetchone():
            continue
        why = ("India VIX above %g" % MAX_VIX if vix is not None and vix > MAX_VIX else
               "VIX not above realized volatility" if st.vrp_filter and (
                   vix is None or realized is None or vix <= realized) else None)
        if items is None:
            items = instruments(now.date())
            spot = index_spot(client)
        exps = sorted({FO._expiry(r["expiry"]) for r in items if FO._expiry(r["expiry"]) >= now.date()})
        if st.expiry_only and (not exps or exps[0] != now.date()):
            why = "not an expiry day"
        if why:
            conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                         (f"os_skip_{st.book}_{day}", why))
            out.append(f"{day} [{st.label}] no trade today: {why}")
            continue
        legs = legs_for(items, spot, now.date(), st) if spot else None
        if legs:
            prices = FO.ltp(client, {l.contract.symbol: l.contract.token for l in legs})
            out += enter(conn, st, legs, prices, stamp, capital)
    tokens = tokens_for_open(conn)
    if tokens:
        prices = FO.ltp(client, tokens)
        conn.executemany("INSERT OR REPLACE INTO live_prices VALUES (?, ?, ?, ?)",
                         [(s, p, stamp, "angelone") for s, p in prices.items()])
        for st in STRATEGIES:
            out += square_off(conn, st.book, prices, stamp) if t >= EXIT else \
                check_stops(conn, st, prices, stamp)
        conn.commit()
    return out
