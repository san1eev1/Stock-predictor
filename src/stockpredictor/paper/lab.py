"""Strategy lab: rule-based intraday strategies traded on paper side by side with the model's
books, each in its own book, so live results show which ideas earn (dashboard: Strategy lab).

Cash books (₹10,000 each, like the model's books; picks at 9:46, square-off 15:15, up to 3 a
side; every candidate is saved and judged like the model's picks):

  lab_news     News drift. Stocks with fresh company news since the last close (relevant
               headlines, learned relevance x FinBERT tone, plus a smaller share of their
               sector's news), traded in the news direction when the first 30 minutes agree.
               Moves that come with news tend to continue (Chan 2003; Savor 2012; Tetlock,
               Saar-Tsechansky & Macskassy 2008).
  lab_fade     Gap fade without news. Large opening gaps (vs the stock's normal daily range)
               with no company news in 36 hours, traded back towards yesterday's close:
               moves without news tend to reverse (Savor 2012). Target = yesterday's close.
  lab_inplay   Stocks in play. The highest relative volume in the first 30 minutes, traded in
               the direction of that move, stop at the other end of the opening range
               (Zarattini, Barbon & Aziz 2024, opening-range breakout on "stocks in play").
  lab_reversal Weekly reversal. Last week's biggest losers bought and winners sold
               (Lehmann 1990; Jegadeesh 1990). Mostly a baseline: costs usually eat it.

Futures book (₹2 lakh, one Nifty future):
  lab_mim      Market intraday momentum (Gao, Han, Li & Zhou 2018, JFE). The Nifty's move from
               yesterday's close to 9:45 predicts its last half hour: at 14:55, if that move was
               at least 0.25%, one Nifty future in its direction, closed at 15:25.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from stockpredictor.costs import DEFAULT_INTRADAY_COSTS
from stockpredictor.data import news as N
from stockpredictor.paper import engine as E
from stockpredictor.paper import intraday as PI

log = logging.getLogger(__name__)

PER_SIDE = 3                  # traded per side; up to CANDIDATES per side are judged
CANDIDATES = 10
MIM_BOOK = "lab_mim"
MIM_CAPITAL = 200_000
MIM_MIN_MOVE = 0.0025
MIM_ENTRY, MIM_EXIT = time(14, 55), time(15, 25)
SECTOR_WEIGHT = 0.3           # a stock's sector news counts this much of its own news


@dataclass(frozen=True)
class Strategy:
    book: str
    label: str
    idea: str


CASH = (
    Strategy("lab_news", "News drift", "Fresh company or sector news, traded in its direction "
             "when the first 30 minutes agree"),
    Strategy("lab_fade", "Gap fade without news", "Big opening gaps with no news, traded back "
             "towards yesterday's close"),
    Strategy("lab_inplay", "Stocks in play", "Highest morning volume, traded with the first "
             "30 minutes, stop at the other end of the range"),
    Strategy("lab_reversal", "Weekly reversal", "Last week's losers bought, winners sold"),
)
FUTURES = Strategy(MIM_BOOK, "Nifty last-half-hour momentum", "Nifty futures 14:55-15:25 in "
                   "the direction of the morning move")
CASH_BOOKS = tuple(s.book for s in CASH)
BOOKS = CASH_BOOKS + (MIM_BOOK,)
LABELS = {s.book: s.label for s in (*CASH, FUTURES)}


# --- news --------------------------------------------------------------------------------

def news_scores(news: pd.DataFrame, universe: pd.DataFrame, since: pd.Timestamp,
                until: pd.Timestamp) -> pd.DataFrame:
    """Per stock: company news since `since` (UTC) and its sector's theme news.
    Columns: symbol, n (company headlines), tone (relevance-weighted), sector_tone, score."""
    cols = ["symbol", "n", "tone", "sector_tone", "score", "headline"]
    if news is None or news.empty:
        return pd.DataFrame(columns=cols)
    w = news[(news["published"] > since) & (news["published"] <= until)
             & news["sentiment"].notna()].copy()
    w = w[~w["title"].astype(str).str.contains(N.PRICE_MOVE)]
    w["rel"] = w["relevance"] if "relevance" in w else 1.0
    comp = w[~w["symbol"].astype(str).str.startswith(N.THEME_PREFIX)]
    comp = comp[comp["rel"] >= N.RELEVANT * 0.75]
    g = comp.groupby("symbol")
    out = pd.DataFrame({
        "n": g.size(),
        "tone": g.apply(lambda x: float((x["sentiment"] * x["rel"]).sum() / x["rel"].sum())),
        "headline": g.apply(lambda x: x.loc[(x["sentiment"] * x["rel"]).abs().idxmax(), "title"]),
    }) if len(comp) else pd.DataFrame(columns=["n", "tone", "headline"])
    sector = N.sector_tone(w, universe)                 # industry -> tone
    ind = universe.set_index("symbol")["industry"]
    syms = sorted(set(out.index) | set(ind[ind.isin(sector.index)].index))
    out = out.reindex(syms)
    out["n"] = out["n"].fillna(0).astype(int)
    out["sector_tone"] = ind.reindex(syms).map(sector).astype(float).values
    out["score"] = out["tone"].fillna(0) + SECTOR_WEIGHT * out["sector_tone"].fillna(0)
    return out.reset_index(names="symbol")[cols]


def last_close_utc(ctx: E.MarketContext, today: pd.Timestamp) -> pd.Timestamp:
    prev = ctx.daily.loc[ctx.daily["date"] < today, "date"].max()
    prev = prev if pd.notna(prev) else today - pd.Timedelta(days=1)
    return (pd.Timestamp(prev).normalize() + pd.Timedelta(hours=15, minutes=30)) \
        .tz_localize("Asia/Kolkata").tz_convert("UTC")


# --- picks -------------------------------------------------------------------------------

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, df.index)


def candidates(book: str, feats: pd.DataFrame, ns: pd.DataFrame,
               recent: set[str]) -> pd.DataFrame:
    """Ranked candidates for one cash book: symbol, side, strength, why, stop, target
    (stop/target as prices; None = the book's default %)."""
    f = feats.copy()
    for c in ("gap", "gap_atr", "r_prev", "r30", "vol30_adv", "ret_5", "c30", "h30", "l30",
              "prev_close"):
        f[c] = _num(f, c)
    rows = []
    if book == "lab_news":
        m = f.merge(ns, on="symbol")
        m = m[(m["score"].abs() >= 0.3) & (np.sign(m["score"]) == np.sign(m["r_prev"]))]
        for r in m.itertuples():
            side = "long" if r.score > 0 else "short"
            why = (f"{int(r.n)} fresh headline{'s' if r.n != 1 else ''}, tone {r.tone:+.2f}"
                   if r.n else "") + (f"; sector news {r.sector_tone:+.2f}"
                                      if pd.notna(r.sector_tone) else "")
            if isinstance(r.headline, str):
                why += f': "{r.headline[:90]}"'
            rows.append((r.symbol, side, abs(r.score), why.strip("; "), None, None))
    elif book == "lab_fade":
        m = f[(f["gap_atr"].abs() >= 1.0) & ~f["symbol"].isin(recent)
              & (np.sign(f["r30"]) != np.sign(f["gap"]))]       # first 30 min already turning
        for r in m.itertuples():
            side = "short" if r.gap > 0 else "long"
            rows.append((r.symbol, side, abs(r.gap_atr),
                         f"gap {r.gap:+.1%} ({abs(r.gap_atr):.1f}x its normal day), no news, "
                         "turning in the first 30 minutes", None, r.prev_close))
    elif book == "lab_inplay":
        m = f[f["vol30_adv"].notna() & (f["r30"].abs() > 0.003)]
        m = m.sort_values("vol30_adv", ascending=False).head(2 * CANDIDATES)
        for r in m.itertuples():
            side = "long" if r.r30 > 0 else "short"
            rows.append((r.symbol, side, r.vol30_adv,
                         f"morning volume {r.vol30_adv:.0%} of a normal day, first 30 min "
                         f"{r.r30:+.1%}", r.l30 if side == "long" else r.h30, None))
    elif book == "lab_reversal":
        m = f[f["ret_5"].notna()].sort_values("ret_5")
        for r in m.head(CANDIDATES).itertuples():
            rows.append((r.symbol, "long", -r.ret_5, f"last week {r.ret_5:+.1%}", None, None))
        for r in m.tail(CANDIDATES).iloc[::-1].itertuples():
            rows.append((r.symbol, "short", r.ret_5, f"last week {r.ret_5:+.1%}", None, None))
    out = pd.DataFrame(rows, columns=["symbol", "side", "strength", "why", "stop", "target"])
    out = out.sort_values("strength", ascending=False)
    return pd.concat([out[out["side"] == s].head(CANDIDATES) for s in ("long", "short")])


DEFAULT_STOP = {"lab_news": 0.01, "lab_fade": 0.012, "lab_inplay": 0.02, "lab_reversal": 0.015}


def open_book(conn: sqlite3.Connection, book: str, picks: pd.DataFrame, feats: pd.DataFrame,
              prices: dict[str, float], today: pd.Timestamp, capital: float,
              at: str = "09:46") -> list[str]:
    """Save every candidate as a judged prediction; open paper trades for the best PER_SIDE."""
    E.ensure_account(conn, capital, book)
    PI.start_day(conn, capital, book)
    stamp = f"{today:%Y-%m-%d}"
    c30 = dict(zip(feats["symbol"], pd.to_numeric(feats["c30"], errors="coerce")))
    conn.executemany("INSERT OR REPLACE INTO intraday_open VALUES (?, ?, ?)",
                     [(stamp, s, float(v)) for s, v in c30.items() if pd.notna(v)])
    n_trades = min(PER_SIDE, (picks["side"] == "long").sum()) + \
        min(PER_SIDE, (picks["side"] == "short").sum())
    slot, budget, opened = capital / max(1, n_trades), capital, []
    for side in ("long", "short"):
        sub = picks[picks["side"] == side]
        for rank, p in enumerate(sub.itertuples(), 1):
            entry = prices.get(p.symbol) or c30.get(p.symbol)
            if not entry or not np.isfinite(entry):
                continue
            sign = 1 if side == "long" else -1
            stop = p.stop if p.stop is not None and pd.notna(p.stop) and sign * (entry - p.stop) > 0 \
                else entry * (1 - sign * DEFAULT_STOP[book])
            target = p.target if p.target is not None and pd.notna(p.target) \
                and sign * (p.target - entry) > 0 else None
            conn.execute(
                "INSERT OR REPLACE INTO predictions (horizon, date, symbol, direction, confidence, "
                "rank, entry_price, stop_loss, target, reasons, model_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (book, stamp, p.symbol, "up" if sign > 0 else "down", None, rank, entry, stop,
                 target, json.dumps([p.why]), "rules"))
            if rank > PER_SIDE:
                continue
            qty = int(slot // entry) or (1 if entry <= 2 * slot else 0)
            qty = min(qty, int(budget // entry))
            if qty <= 0:
                continue
            budget -= qty * entry
            c = DEFAULT_INTRADAY_COSTS.cost("buy" if sign > 0 else "sell", qty * entry)
            conn.execute(
                "INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, "
                "costs, status, reason, stop_loss, target) VALUES (?,?,?,?,?,?,?,'open',?,?,?)",
                (book, p.symbol, side, qty, f"{stamp} {at}", entry, c, p.why[:200], stop, target))
            conn.execute("UPDATE paper_accounts SET cash = cash - ? WHERE horizon = ?",
                         (qty * entry + c, book))
            opened.append(f"{side.upper():5} {p.symbol} {qty} @ {entry:.2f}")
    conn.commit()
    return opened


def run_picks(conn: sqlite3.Connection, ctx: E.MarketContext, feats: pd.DataFrame,
              prices: dict[str, float], now: datetime, capital: float) -> dict[str, list[str]]:
    """All cash lab books for today (once a day, after the model's 9:46 picks)."""
    today = pd.Timestamp(now.date())
    until = pd.Timestamp(now).tz_convert("UTC")
    ns = news_scores(ctx.news, ctx.universe, last_close_utc(ctx, today), until)
    recent_news = news_scores(ctx.news, ctx.universe, until - pd.Timedelta(hours=36), until)
    recent = set(recent_news.loc[recent_news["n"] > 0, "symbol"])
    done = {}
    for st in CASH:
        try:
            picks = candidates(st.book, feats, ns, recent)
            done[st.book] = open_book(conn, st.book, picks, feats, prices, today, capital,
                                      at=now.strftime("%H:%M"))
        except Exception:
            log.exception("strategy lab %s failed", st.book)
    return done


# --- Nifty futures: market intraday momentum -----------------------------------------------

def nifty_future(today) -> dict | None:
    from stockpredictor.paper import fno as FO
    from stockpredictor.paper import optsell as OS

    rows = [r for r in OS.instruments(today) if r["symbol"].endswith("FUT")]
    rows = [r for r in rows if FO._expiry(r["expiry"]) >= today]
    return min(rows, key=lambda r: FO._expiry(r["expiry"]), default=None)


def mim_tick(conn: sqlite3.Connection, client, now: datetime, prev_close: float | None) -> list[str]:
    """9:45: note the Nifty's move since yesterday's close; 14:55: one future in its direction
    if at least MIM_MIN_MOVE; 15:25: close."""
    from stockpredictor.paper import fno as FO
    from stockpredictor.paper import optsell as OS

    day, t = f"{now:%Y-%m-%d}", now.time()
    key = f"lab_mim_move_{day}"
    log_lines = []
    E.ensure_account(conn, MIM_CAPITAL, MIM_BOOK)
    if time(9, 45) <= t < MIM_ENTRY and _get(conn, key) is None and prev_close:
        spot = OS.index_spot(client)
        if spot:
            _put(conn, key, f"{spot / prev_close - 1:.5f}")
    if MIM_ENTRY <= t < MIM_EXIT and _get(conn, f"lab_mim_done_{day}") is None:
        _put(conn, f"lab_mim_done_{day}", "1")
        move = _get(conn, key)
        fut = nifty_future(now.date())
        if move is None or fut is None:
            _put(conn, f"os_skip_{MIM_BOOK}_{day}", "no 9:45 reading" if move is None
                 else "no Nifty future found")
            return log_lines
        move = float(move)
        if abs(move) < MIM_MIN_MOVE:
            _put(conn, f"os_skip_{MIM_BOOK}_{day}", f"morning move {move:+.2%} too small")
            return log_lines
        px = FO.ltp(client, {fut["symbol"]: fut["token"]}).get(fut["symbol"])
        if not px:
            return log_lines
        side = "long" if move > 0 else "short"
        qty = int(float(fut["lotsize"]))
        c = FO.cost("FUT", "buy" if side == "long" else "sell", qty * px)
        PI.start_day(conn, MIM_CAPITAL, MIM_BOOK)
        conn.execute(
            "INSERT INTO paper_trades (horizon, symbol, side, qty, entry_time, entry_price, costs, "
            "status, reason, stop_loss, target) VALUES (?,?,?,?,?,?,?,'open',?,?,NULL)",
            (MIM_BOOK, fut["symbol"], side, qty, f"{day} {t:%H:%M}", px, c,
             f"Nifty {move:+.2%} by 9:45", px * (1 - 0.006 if side == "long" else 1 + 0.006)))
        conn.execute("INSERT OR REPLACE INTO fno_contracts (symbol, token, name, lot) VALUES (?, ?, ?, ?)",
                     (fut["symbol"], fut["token"], "NIFTY", qty))
        conn.commit()
        log_lines.append(f"{side} {fut['symbol']} {qty} @ {px:.2f} (Nifty {move:+.2%} by 9:45)")
    open_ = PI.open_trades(conn, MIM_BOOK)
    if open_ and t >= MIM_ENTRY:
        toks = {r["symbol"]: r["token"] for r in conn.execute(
            "SELECT symbol, token FROM fno_contracts WHERE symbol IN (%s)"
            % ",".join("?" * len(open_)), [o["symbol"] for o in open_])}
        px = FO.ltp(client, toks)
        stamp = f"{day} {t:%H:%M}"
        for o in open_:
            p = px.get(o["symbol"])
            if p is None:
                continue
            stop = (o["side"] == "long" and p < o["stop_loss"]) or \
                   (o["side"] == "short" and p > o["stop_loss"])
            if stop or t >= MIM_EXIT:
                log_lines.append(_close_future(conn, o, p, "stop-loss" if stop else "15:25 exit",
                                               stamp))
        conn.commit()
    return log_lines


def _close_future(conn, t, price: float, reason: str, stamp: str) -> str:
    from stockpredictor.paper import fno as FO

    sign = 1 if t["side"] == "long" else -1
    c = FO.cost("FUT", "sell" if sign > 0 else "buy", t["qty"] * price)
    pnl = sign * (price - t["entry_price"]) * t["qty"] - t["costs"] - c
    conn.execute("UPDATE paper_trades SET exit_time = ?, exit_price = ?, costs = costs + ?, "
                 "pnl = ?, status = 'closed', exit_reason = ? WHERE id = ?",
                 (stamp, price, c, pnl, reason, t["id"]))
    return f"{t['side']} {t['symbol']} closed @ {price:.2f} ({reason}), P&L {pnl:+.0f}"


def _get(conn, key):
    r = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return r[0] if r else None


def _put(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


# --- results -----------------------------------------------------------------------------

def history(conn: sqlite3.Connection, books=None) -> pd.DataFrame:
    """Daily closed P&L per book: day, book, pnl, trades."""
    books = books or BOOKS
    q = ",".join("?" * len(books))
    return pd.read_sql(f"SELECT substr(entry_time, 1, 10) AS day, horizon AS book, "
                       f"SUM(pnl) AS pnl, COUNT(*) AS trades FROM paper_trades "
                       f"WHERE horizon IN ({q}) AND status = 'closed' GROUP BY 1, 2 ORDER BY 1",
                       conn, params=list(books))


def leaderboard(hist: pd.DataFrame, capital: dict[str, float], labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for book, label in labels.items():
        h = hist.loc[hist["book"] == book, "pnl"]
        eq = h.cumsum()
        t = float(h.mean() / h.std() * np.sqrt(len(h))) if len(h) > 2 and h.std() > 0 else None
        rows.append({"Strategy": label, "Days traded": len(h), "Total": float(h.sum()),
                     "Average day": float(h.mean()) if len(h) else None,
                     "Winning days": float((h > 0).mean()) if len(h) else None,
                     "Worst day": float(h.min()) if len(h) else None,
                     "Deepest fall": float((eq - eq.cummax()).min()) if len(h) else None,
                     "t-stat": t,
                     "On capital": float(h.sum()) / capital.get(book, 1)})
    return pd.DataFrame(rows).sort_values("Total", ascending=False)
