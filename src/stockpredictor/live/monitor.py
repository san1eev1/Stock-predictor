"""Live monitor: one long-running loop on the Mac.

Market hours (Mon-Fri 9:15-15:30 IST, skipping holidays):
  every minute   live prices -> fill queued paper orders, stop-loss checks
                 (paper: sell; real portfolio: alert)
  every 15 min   sync news from git -> negative-news alerts / paper exits,
                 provisional live re-scoring of all stocks
After the close (from 17:15 IST): sync data -> official daily decision.
Weekends: retrain the model if it is more than 6 days old.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from stockpredictor import store
from stockpredictor.data import news as N
from stockpredictor.features import longterm as F
from stockpredictor.live.prices import IST, LivePrices, in_market_hours, now_ist
from stockpredictor.models import longterm as M
from stockpredictor.live import intraday_bars
from stockpredictor.models import intraday as MI
from stockpredictor.paper import daily as D
from stockpredictor.paper import engine as E
from stockpredictor.paper import intraday as PI
from stockpredictor.portfolio import real as R

log = logging.getLogger(__name__)
AFTER_CLOSE = time(17, 15)
INTRADAY_PICKS = time(9, 46)       # first 30 minutes complete
INTRADAY_LATEST = time(10, 15)     # too late to act on a 9:45 signal after this
INTRADAY_SQUARE_OFF = time(15, 15)
FIRST_FILL = time(9, 20)       # skip the opening minutes' noise
LIVE_HISTORY_DAYS = 400        # enough history for 12-month features


def _setting(conn, key, default=None):
    row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def _set(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()


def alert(conn: sqlite3.Connection, source: str, kind: str, symbol: str | None, message: str) -> bool:
    cur = conn.execute("INSERT OR IGNORE INTO alerts (source, kind, symbol, message) VALUES (?, ?, ?, ?)",
                       (source, kind, symbol, message))
    conn.commit()
    if cur.rowcount:
        log.warning("ALERT %s %s %s", kind, symbol or "", message)
    return bool(cur.rowcount)


class Monitor:
    def __init__(self, conn: sqlite3.Connection, store_dir: Path, prices: LivePrices,
                 capital: float, clock=now_ist, sync=store.sync,
                 capital_intraday: float = 100_000):
        self.conn, self.store_dir, self.prices = conn, store_dir, prices
        self.capital, self.clock, self.sync = capital, clock, sync
        self._ctx: E.MarketContext | None = None
        self._model: M.LongTermModel | None = None
        self._last_quarter: datetime | None = None
        self._live_day: tuple | None = None   # (date, is_trading_day)
        self._imodel: MI.IntradayModel | None = None
        self.capital_intraday = capital_intraday
        E.ensure_account(conn, capital)
        E.ensure_account(conn, capital_intraday, PI.HORIZON)

    # --- helpers -----------------------------------------------------------
    def ctx(self, reload: bool = False) -> E.MarketContext:
        if self._ctx is None or reload:
            self._ctx = E.MarketContext.load(self.store_dir)
        return self._ctx

    def model(self) -> M.LongTermModel:
        if self._model is None:
            self._model = D.load_or_train(self.ctx())
        return self._model

    def intraday_model(self) -> MI.IntradayModel | None:
        if self._imodel is None and (MI.MODEL_DIR / "model.txt").exists():
            self._imodel = MI.IntradayModel.load()
        return self._imodel

    def watched_symbols(self) -> set[str]:
        syms = set(E.holdings(self.conn))
        syms |= {t["symbol"] for t in PI.open_trades(self.conn)}
        syms |= {r[0] for r in self.conn.execute(
            "SELECT symbol FROM paper_orders WHERE status = 'pending'")}
        for h in R.HORIZONS:
            held = R.holdings(self.conn, h)
            syms |= set(held.loc[held["qty"] > 0, "symbol"]) if not held.empty else set()
        return syms

    def trading_today(self, now: datetime) -> bool:
        if self._live_day is None or self._live_day[0] != now.date():
            self._live_day = (now.date(), self.prices.market_is_live())
        return self._live_day[1]

    # --- jobs --------------------------------------------------------------
    def tick(self) -> list[str]:
        now = self.clock()
        done = []
        _set(self.conn, "monitor_heartbeat", now.isoformat(timespec="seconds"))
        if in_market_hours(now) and self.trading_today(now):
            day = f"{now:%Y-%m-%d}"
            if INTRADAY_PICKS <= now.time() <= INTRADAY_LATEST \
                    and _setting(self.conn, "id_last_picks") != day:
                self.intraday_picks_job(now)
                done.append("intraday-picks")
            self.minute_job(now)
            done.append("minute")
            if now.time() >= INTRADAY_SQUARE_OFF and _setting(self.conn, "id_last_squareoff") != day:
                self.square_off_job(now)
                done.append("square-off")
            if self._last_quarter is None or (now - self._last_quarter).total_seconds() >= 900:
                self.quarter_job(now)
                self._last_quarter = now
                done.append("quarter")
        elif now.weekday() < 5 and now.time() >= AFTER_CLOSE \
                and _setting(self.conn, "lt_after_close_day") != f"{now:%Y-%m-%d}":
            if self._last_quarter is None or (now - self._last_quarter).total_seconds() >= 900:
                self._last_quarter = now
                if self.after_close_job(now):
                    done.append("after-close")
        elif now.weekday() >= 5:
            if self.weekly_retrain(now):
                done.append("retrain")
        return done

    def minute_job(self, now: datetime) -> None:
        symbols = self.watched_symbols()
        if not symbols:
            return
        prices = self.prices.get(sorted(symbols))
        stamp = now.strftime("%Y-%m-%d %H:%M")
        self.conn.executemany("INSERT OR REPLACE INTO live_prices VALUES (?, ?, ?, ?)",
                              [(s, p, stamp, self.prices.source) for s, p in prices.items()])
        _set(self.conn, "live_source", self.prices.source)

        rules, _ = D.get_rules(self.conn)
        if now.time() >= FIRST_FILL:
            for line in E.fill_pending(self.conn, prices, stamp, rules):
                alert(self.conn, "paper-longterm", "fill", line.split()[1], f"{stamp} {line}")

        # Paper stop-loss: sell immediately at the live price.
        for sym, pos in E.holdings(self.conn).items():
            p = prices.get(sym)
            if p is not None and p <= pos.entry_price * (1 - rules.stop_loss):
                E.queue_orders(self.conn, [(sym, "stop-loss")], [], stamp)
                E.fill_pending(self.conn, {sym: p}, stamp, rules)
                alert(self.conn, "paper-longterm", "stop-loss", sym,
                      f"{now:%Y-%m-%d} paper stop-loss: sold {sym} at {p:.2f}")

        # Intraday paper: stop-loss / target exits.
        for line in PI.check_exits(self.conn, prices, stamp):
            kind = "stop-loss" if "stop-loss" in line else "target"
            alert(self.conn, "paper-intraday", kind, line.split()[1], f"{stamp} {line}")

        # Real portfolio: alerts only (you place trades yourself on Groww).
        for h in R.HORIZONS:
            s = R.summary(self.conn, h, prices)
            for r in s.table.itertuples():
                if r.symbol in prices and r.price <= r.stop_loss:
                    alert(self.conn, f"portfolio-{h}", "stop-loss", r.symbol,
                          f"{now:%Y-%m-%d} {r.symbol} at {r.price:.2f} hit your stop-loss "
                          f"{r.stop_loss:.2f} (avg cost {r.avg_cost:.2f})")

    def intraday_picks_job(self, now: datetime) -> None:
        day = f"{now:%Y-%m-%d}"
        _set(self.conn, "id_last_picks", day)       # at most one attempt per day
        rules, enabled = PI.get_rules(self.conn)
        model = self.intraday_model()
        if not enabled:
            return
        if model is None:
            alert(self.conn, "paper-intraday", "info", None,
                  f"{day} no intraday model yet - run `train-intraday` (after intraday-backfill)")
            return
        ctx = self.ctx()
        active = store.tradable(ctx.universe)
        f30 = intraday_bars.first30(self.prices, active, now.date())
        if len(f30) < 0.8 * len(active):
            alert(self.conn, "paper-intraday", "info", None,
                  f"{day} intraday skipped: first-30-minute data for only {len(f30)} stocks")
            return
        feats = PI.todays_features(ctx, f30, pd.Timestamp(now.date()), self.store_dir)
        prices = self.prices.get(sorted(feats["symbol"]))
        summary = N.news_summary(ctx.news, pd.Timestamp(now).tz_convert("UTC"))
        negative = set(summary.loc[summary["strong_negative"].astype(bool), "symbol"])
        strengths = PI.recent_strengths(model, self.store_dir, ctx, rules) \
            if rules.skip_quantile > 0 else None
        r = PI.run_picks(self.conn, feats, model, prices, pd.Timestamp(now.date()), rules,
                         self.capital_intraday, negative, strengths)
        msg = "skip today (weak signal)" if r["skipped"] else "; ".join(r["picks"])
        alert(self.conn, "paper-intraday", "decision", None, f"{day} 9:45 intraday: {msg}")

    def square_off_job(self, now: datetime) -> None:
        day = f"{now:%Y-%m-%d}"
        _set(self.conn, "id_last_squareoff", day)
        stamp = now.strftime("%Y-%m-%d %H:%M")
        opens = [r[0] for r in self.conn.execute(
            "SELECT symbol FROM intraday_open WHERE date = ?", (day,))]
        symbols = sorted(set(opens) | {t["symbol"] for t in PI.open_trades(self.conn)})
        if not symbols:
            return
        prices = self.prices.get(symbols)
        for line in PI.square_off(self.conn, prices, stamp):
            alert(self.conn, "paper-intraday", "fill", line.split()[1], f"{stamp} {line}")
        PI.evaluate_day(self.conn, day, prices)

    def quarter_job(self, now: datetime) -> None:
        try:
            self.sync(self.store_dir)
            self._ctx = None
        except Exception as exc:
            log.warning("sync failed: %s", exc)
        ctx = self.ctx()
        summary = N.news_summary(ctx.news, pd.Timestamp(now).tz_convert("UTC"))
        negative = set(summary.loc[summary["strong_negative"].astype(bool), "symbol"])
        severe = set(summary.loc[summary["severe_negative"].astype(bool), "symbol"])
        rules, _ = D.get_rules(self.conn)
        paper = set(E.holdings(self.conn))
        for sym in negative & self.watched_symbols():
            head = N.latest_headlines(ctx.news, sym, 1)
            title = head["title"].iloc[0] if len(head) else ""
            alert(self.conn, "news", "negative-news", sym, f"{now:%Y-%m-%d} {sym}: {title}")
            if sym in paper and sym in severe and rules.news_exit:
                p = self.prices.get([sym]).get(sym)
                if p:
                    E.queue_orders(self.conn, [(sym, "negative news")], [], f"{now:%Y-%m-%d %H:%M}")
                    E.fill_pending(self.conn, {sym: p}, f"{now:%Y-%m-%d %H:%M}", rules)
        self.live_scores(now)

    def live_scores(self, now: datetime) -> None:
        """Provisional scores: today's live prices appended as a temporary daily candle."""
        ctx = self.ctx()
        active = store.tradable(ctx.universe)
        live = self.prices.get(active)
        if not live:
            return
        today = pd.Timestamp(now.date())
        start = sorted(ctx.daily["date"].unique())[-LIVE_HISTORY_DAYS]
        hist = ctx.daily[(ctx.daily["date"] >= start) & (ctx.daily["date"] < today)]
        prov = pd.DataFrame([{"symbol": s, "date": today, "open": p, "high": p, "low": p,
                              "close": p, "adj_close": p, "volume": float("nan"), "source": "live"}
                             for s, p in live.items()])
        idx = ctx.indices[ctx.indices["date"] >= start]
        feats = F.build_features(pd.concat([hist, prov], ignore_index=True), idx, ctx.universe)
        now_rows = feats[feats["date"] == today]
        if now_rows.empty:
            return
        scores = self.model().score(now_rows)
        ranks = scores.rank(ascending=False, method="first").astype(int)
        official: dict[str, int] = {}   # ranks at the last official decision
        last_day = self.conn.execute(
            "SELECT MAX(date) FROM predictions WHERE horizon = 'longterm'").fetchone()[0]
        if last_day:
            off = ctx.feats[(ctx.feats["date"] == pd.Timestamp(last_day))
                            & ctx.feats["symbol"].isin(active)]
            if not off.empty:
                osc = self.model().score(off)
                official = dict(zip(off["symbol"], osc.rank(ascending=False, method="first").astype(int)))
        stamp = now.strftime("%Y-%m-%d %H:%M")
        self.conn.execute("DELETE FROM live_scores")
        self.conn.executemany(
            "INSERT INTO live_scores VALUES (?, ?, ?, ?, ?, ?)",
            [(s, stamp, float(sc), int(rk), official.get(s), live.get(s))
             for s, sc, rk in zip(now_rows["symbol"], scores, ranks)])
        self.conn.commit()

    def after_close_job(self, now: datetime) -> bool:
        try:
            self.sync(self.store_dir)
        except Exception as exc:
            log.warning("sync failed: %s", exc)
            return False
        ctx = self.ctx(reload=True)
        latest = ctx.daily["date"].max()
        waited_long = now.time() >= time(21, 0)
        if latest.date() < now.date() and not waited_long:
            return False   # today's data not published yet; try again in 15 minutes
        results = D.run_daily(self.conn, ctx, self.capital, self.model())
        for r in results:
            buys, sells = ", ".join(r["buys"]) or "none", ", ".join(s for s, _ in r["sells"]) or "none"
            alert(self.conn, "paper-longterm", "decision", None,
                  f"{r['date']:%Y-%m-%d} decision - buy: {buys}; sell: {sells}")
        _set(self.conn, "lt_after_close_day", f"{now:%Y-%m-%d}")
        return True

    def weekly_retrain(self, now: datetime) -> bool:
        model = self.model()
        age = (now.replace(tzinfo=None) - datetime.fromisoformat(model.trained_at)).days
        if age < 6:
            return False
        ctx = self.ctx(reload=True)
        labeled = M.add_labels(F.weekly_snapshots(ctx.feats), ctx.daily, ctx.indices)
        self._model = M.LongTermModel.train(labeled)
        self._model.save(M.MODEL_DIR)
        alert(self.conn, "model", "info", None,
              f"{now:%Y-%m-%d} model retrained on data up to {self._model.train_to}")
        self.retrain_intraday(ctx, now)
        return True

    def retrain_intraday(self, ctx: E.MarketContext, now: datetime) -> None:
        from stockpredictor.data import intraday as I
        from stockpredictor.features import intraday as FI

        feats = FI.build(I.load_summaries(self.store_dir), ctx.daily, ctx.feats,
                         store.load_actions(self.store_dir))
        if feats.empty or feats["date"].nunique() < MI.MIN_TRAIN_DAYS:
            return
        self._imodel = MI.IntradayModel.train(feats)
        self._imodel.save()
        alert(self.conn, "model", "info", None,
              f"{now:%Y-%m-%d} intraday model retrained on {self._imodel.train_days} days")


def run_forever(monitor: Monitor, interval: int = 60) -> None:
    import time as _time

    log.info("Live monitor started (Ctrl+C to stop)")
    while True:
        started = _time.monotonic()
        try:
            done = monitor.tick()
            if done:
                log.info("%s ran: %s", monitor.clock().strftime("%H:%M"), ", ".join(done))
        except Exception:
            log.exception("monitor tick failed; continuing")
        _time.sleep(max(1.0, interval - (_time.monotonic() - started)))
