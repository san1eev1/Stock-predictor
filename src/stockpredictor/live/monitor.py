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

from stockpredictor import config, store
from stockpredictor.data import news as N
from stockpredictor.features import longterm as F
from stockpredictor.live.prices import IST, LivePrices, in_market_hours, now_ist
from stockpredictor.models import longterm as M
from stockpredictor.live import intraday_bars
from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T
from stockpredictor.paper import daily as D
from stockpredictor.paper import engine as E
from stockpredictor.paper import intraday as PI
from stockpredictor.paper import scoreboard as SB
from stockpredictor.portfolio import real as R

log = logging.getLogger(__name__)
AFTER_CLOSE = time(17, 15)
LOCAL_FETCH_AFTER = time(17, 45)   # fetch prices ourselves if GitHub hasn't by then
INTRADAY_PICKS = time(9, 46)       # first 30 minutes complete
INTRADAY_LATEST = time(14, 30)     # late start: still pick (at live prices) until 14:30
INTRADAY_SQUARE_OFF = time(12, 30)   # first intraday book closes here (data.intraday.TRADE_EXIT)
CLOSE_SQUARE_OFF = time(15, 15)      # second book ("until the close") closes here
LIVE_LEARN = time(15, 32)         # market closed: learn from today's full live session
MAC_TRAIN_AT = time(16, 0)        # the Mac's one daily training (GitHub trains at 21:00 IST)
PRE_MARKET = time(8, 30)          # no background tuning from here until the day's work is done
TUNE_EVERY_MIN = 60               # market closed: one tuning round on history per hour
TUNE_CANDIDATES = 3               # new settings tried per model in each round
SCHEDULED_TUNE_ROUNDS = 10        # background job (monitor off): at most 10 rounds / 30 min
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
        self._cmodel: MI.IntradayModel | None = None   # 9:45 -> close (learning model)
        self._models_rev: str | None = None      # commit of the cloud-trained models in use
        self.background = None                   # live/background.py BackgroundTrainer
        self._started = False
        self._last_pick_try: datetime | None = None
        self.capital_intraday = capital_intraday
        E.ensure_account(conn, capital)
        for book in PI.BOOKS:
            E.ensure_account(conn, capital_intraday, book)

    # --- helpers -----------------------------------------------------------
    def ctx(self, reload: bool = False) -> E.MarketContext:
        if self._ctx is None or reload:
            self._ctx = E.MarketContext.load(self.store_dir)
        return self._ctx

    def model(self) -> M.LongTermModel:
        if self._model is None:
            self._model = D.load_or_train(self.ctx(), M.model_path())
        return self._model

    def intraday_model(self) -> MI.IntradayModel | None:
        if self._imodel is None and MI.model_path(MI.TRADE) is not None:
            self._imodel = MI.IntradayModel.load(MI.model_path(MI.TRADE))
        return self._imodel

    def close_model(self) -> MI.IntradayModel | None:
        if self._cmodel is None and MI.model_path(MI.CLOSE) is not None:
            self._cmodel = MI.IntradayModel.load(MI.model_path(MI.CLOSE))
        return self._cmodel

    def watched_symbols(self) -> set[str]:
        syms = set(E.holdings(self.conn)) | set(E.holdings(self.conn, E.SHORT_HORIZON))
        syms |= {t["symbol"] for h in PI.BOOKS for t in PI.open_trades(self.conn, h)}
        syms |= {r[0] for r in self.conn.execute(
            "SELECT symbol FROM paper_orders WHERE status = 'pending'")}
        for h in R.HORIZONS:
            held = R.holdings(self.conn, h)
            syms |= set(held.loc[held["qty"] > 0, "symbol"]) if not held.empty else set()
        return syms

    def trading_today(self, now: datetime) -> bool:
        """Is NSE open today? A definite answer is kept for the day; an unknown one
        (network hiccup) counts as open and is checked again after 30 minutes."""
        if self._live_day is None or self._live_day[0] != now.date() or (
                self._live_day[1] is None and (now - self._live_day[2]).total_seconds() > 1800):
            self._live_day = (now.date(), self.prices.market_is_live(), now)
        return self._live_day[1] is not False

    # --- jobs --------------------------------------------------------------
    def tick(self) -> list[str]:
        now = self.clock()
        done = []
        _set(self.conn, "monitor_heartbeat", now.isoformat(timespec="seconds"))
        if self.background is not None and self.background.model_changed.is_set():
            self.background.model_changed.clear()
            self._imodel = self._cmodel = None   # background training improved them: reload
        if not self._started:
            self._started = True
            self.startup_job(now)
            done.append("startup")
            if self.background is not None:
                self.background.ready.set()
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
            if now.time() >= CLOSE_SQUARE_OFF and _setting(self.conn, "id_close_squareoff") != day:
                self.square_off_job(now, PI.CLOSE_HORIZON)
                done.append("square-off-close")

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
        # After the close: today's whole session (trading stopped at 12:30, but the model
        # keeps learning from the market until 15:30) goes into the intraday history.
        day = f"{now:%Y-%m-%d}"
        if now.weekday() < 5 and now.time() >= LIVE_LEARN \
                and _setting(self.conn, "id_last_squareoff") == day \
                and _setting(self.conn, "live_learn_day") != day:
            _set(self.conn, "live_learn_day", day)
            if self.background is not None:
                self.background.learn_from_today(now.date())
                done.append("live-learn")
        # The Mac's one training of the day: today's Angel One session, then long-term and
        # both intraday models (config.MAC_DAILY_TRAINING; GitHub trains again at 21:00).
        if not config.MAC_TRAINING and config.MAC_DAILY_TRAINING and now.weekday() < 5 \
                and now.time() >= MAC_TRAIN_AT and _setting(self.conn, "mac_train_day") != day:
            _set(self.conn, "mac_train_day", day)
            self.angel_topup(now)
            self.daily_mac_train(self.ctx())
            done.append("mac-train")
        # Without the background thread, tune in the monitor itself (only when idle).
        if self.background is None and config.MAC_TRAINING and self.market_idle(now) \
                and self.idle_tune(now):
            done.append("tune")
        return done

    def market_idle(self, now: datetime) -> bool:
        """Nothing to do for the market: nights, weekends, holidays, and weekday evenings
        once the after-close decision is made."""
        t = now.time()
        if now.weekday() >= 5 or t < PRE_MARKET:
            return True
        if in_market_hours(now):
            return not self.trading_today(now)
        return t >= AFTER_CLOSE and _setting(self.conn, "lt_after_close_day") == f"{now:%Y-%m-%d}"

    def idle_tune(self, now: datetime, force: bool = False) -> bool:
        """Keep learning from history while the market is closed: every hour, try new model
        settings (walk-forward on all history) and keep them only if they test better."""
        last = _setting(self.conn, "last_idle_tune")
        if not force and last and (now.replace(tzinfo=None) - datetime.fromisoformat(last)
                                   ).total_seconds() < TUNE_EVERY_MIN * 60:
            return False
        _set(self.conn, "last_idle_tune", now.replace(tzinfo=None).isoformat(timespec="seconds"))
        self._tune_round(now, TUNE_CANDIDATES, "background")
        return True

    def _tune_round(self, now: datetime, n: int, label: str) -> None:
        try:
            ctx = self.ctx()
            msg = "long-term: trained on GitHub"
            if not config.CLOUD_TRAINING:
                lt = T.tune_longterm(ctx, self.conn, n_candidates=n)
                msg = f"long-term IC {lt['ic']:.3f}"
                if lt["adopted"]:
                    self._model = None
                    alert(self.conn, "model", "info", None,
                          f"{now:%Y-%m-%d %H:%M} {label} tuning improved long-term IC "
                          f"{lt['previous_ic']:.3f} -> {lt['ic']:.3f}")
                    msg += " (improved - new settings adopted)"
            for tgt in MI.TARGETS[1:]:
                T.tune_intraday(ctx, self.store_dir, self.conn, n_candidates=n, target=tgt)
            self._cmodel = None
            it = T.tune_intraday(ctx, self.store_dir, self.conn, n_candidates=n)
            if it:
                msg += f", intraday IC {it['ic']:.3f}"
                if it["adopted"]:
                    self._imodel = None
                    alert(self.conn, "model", "info", None,
                          f"{now:%Y-%m-%d %H:%M} {label} tuning improved intraday IC "
                          f"{it['previous_ic']:.3f} -> {it['ic']:.3f}")
                    msg += " (improved - new settings adopted)"
            log.info("Training on history (%s round, %d new settings per model): %s",
                     label, n, msg)
        except Exception:
            log.exception("%s tuning failed", label)

    def scheduled_run(self, now: datetime) -> list[str]:
        """One headless pass for the daily background job (when the live monitor is off)."""
        beat = _setting(self.conn, "monitor_heartbeat")
        if beat and (now - datetime.fromisoformat(beat)).total_seconds() < 300:
            return []   # the live monitor is running and does all of this itself
        done = []
        if now.weekday() < 5 and now.time() >= AFTER_CLOSE \
                and _setting(self.conn, "lt_after_close_day") != f"{now:%Y-%m-%d}":
            if self.after_close_job(now):
                done.append("after-close")
        elif now.weekday() >= 5 and self.weekly_retrain(now):
            done.append("retrain")
        if self.market_idle(now):
            # The live monitor is off: spend up to 30 minutes learning from history.
            import time as _time
            end = _time.monotonic() + 30 * 60
            for _ in range(SCHEDULED_TUNE_ROUNDS):
                if _time.monotonic() >= end or not self.idle_tune(now, force=True):
                    break
                done.append("tune")
        return done

    def minute_job(self, now: datetime) -> None:
        # Every tradable stock gets a fresh price each minute (Angel One: 50 per request),
        # so the dashboard's picks move live, not only the stocks we hold.
        symbols = self.watched_symbols() | set(store.tradable(self.ctx().universe))
        if not symbols:
            return
        prices = self.prices.get(sorted(symbols))
        stamp = now.strftime("%Y-%m-%d %H:%M")
        self.conn.executemany("INSERT OR REPLACE INTO live_prices VALUES (?, ?, ?, ?)",
                              [(s, p, stamp, self.prices.source) for s, p in prices.items()])
        _set(self.conn, "live_source", self.prices.source)
        sample = next((f"{s} Rs {prices[s]:,.2f}" for s in ("RELIANCE", "HDFCBANK", "TCS")
                       if s in prices), "")
        log.info("Live prices: %d/%d stocks from %s  %s", len(prices), len(symbols),
                 self.prices.source, sample)
        if len(prices) < len(symbols) * 0.9:
            log.warning("Live prices missing for %d stocks (%s)", len(symbols) - len(prices),
                        getattr(self.prices, "last_error", None) or "no data returned")

        rules, _ = D.get_rules(self.conn)
        if now.time() >= FIRST_FILL:
            for h in (E.HORIZON, E.SHORT_HORIZON):
                for line in E.fill_pending(self.conn, prices, stamp, rules, horizon=h):
                    alert(self.conn, f"paper-{h}", "fill", line.split()[1], f"{stamp} {line}")

        # Paper stop-loss (buy book: price falls; virtual short book: price rises).
        for h, side in ((E.HORIZON, "long"), (E.SHORT_HORIZON, "short")):
            for sym, pos in E.holdings(self.conn, h).items():
                p = prices.get(sym)
                if p is None:
                    continue
                hit = p <= pos.entry_price * (1 - rules.stop_loss) if side == "long" \
                    else p >= pos.entry_price * (1 + rules.stop_loss)
                if hit:
                    E.queue_orders(self.conn, [(sym, "stop-loss")], [], stamp, h)
                    E.fill_pending(self.conn, {sym: p}, stamp, rules, horizon=h)
                    alert(self.conn, f"paper-{h}", "stop-loss", sym,
                          f"{now:%Y-%m-%d} paper stop-loss ({'buy' if side == 'long' else 'sell'} "
                          f"book): closed {sym} at {p:.2f}")

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

    def startup_job(self, now: datetime) -> None:
        """Whenever the program starts: newest data, train if not yet today, catch up on
        missed decisions, judge predictions, and make today's intraday picks if the market
        is open. Then print the accuracy scoreboard."""
        log.info("Starting up: getting data, training and catching up...")
        try:
            self.sync(self.store_dir)
        except Exception as exc:
            log.warning("sync failed (%s) - using the data already on this Mac", exc)
        try:
            ctx = self.ctx(reload=True)
            last_close = ctx.daily["date"].max().date()
            if (now.date() - last_close).days > 1 and now.weekday() < 5 \
                    and now.time() >= LOCAL_FETCH_AFTER or (now.date() - last_close).days > 3:
                if self.local_catchup(now):
                    ctx = self.ctx(reload=True)
            lt = self.retrain_longterm(ctx)
            if lt is not None:
                self._model = lt
                log.info("Long-term model retrained on data up to %s", lt.train_to)
            if config.MAC_TRAINING:
                it = T.retrain_intraday(ctx, self.store_dir, self.conn)
                if it is not None:
                    self._imodel = it
                    log.info("Intraday model retrained on %s days", it.train_days)
                if T.retrain_intraday(ctx, self.store_dir, self.conn, target=MI.CLOSE) is not None:
                    self._cmodel = None
            results = D.run_daily(self.conn, ctx, self.capital, self.model())
            for r in results:
                log.info("Decision for %s: buy %s; sell %s", f"{r['date']:%Y-%m-%d}",
                         ", ".join(r["buys"]) or "none",
                         ", ".join(s_ for s_, _ in r["sells"]) or "none")
            E.evaluate_predictions(self.conn, ctx)
        except Exception:
            log.exception("startup training/decision failed")
        if in_market_hours(now) and self.trading_today(now) \
                and INTRADAY_PICKS <= now.time() <= INTRADAY_LATEST \
                and _setting(self.conn, "id_last_picks") != f"{now:%Y-%m-%d}":
            self.intraday_picks_job(now)
        self.print_scoreboard(now)

    def print_scoreboard(self, now: datetime) -> None:
        try:
            for line in SB.lines(SB.compute(self.conn, now)):
                log.info(line)
        except Exception:
            log.exception("scoreboard failed")

    def intraday_picks_job(self, now: datetime) -> None:
        day = f"{now:%Y-%m-%d}"
        if self._last_pick_try and (now - self._last_pick_try).total_seconds() < 300:
            return                                   # retry at most every 5 minutes
        self._last_pick_try = now
        _set(self.conn, "id_last_picks", day)       # set now; cleared again if data is missing
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
            log.warning("Intraday: first-30-minute data for only %s of %s stocks - retrying in "
                        "5 minutes", len(f30), len(active))
            _set(self.conn, "id_last_picks", "")    # not done: allow a retry
            return
        feats = PI.todays_features(ctx, f30, pd.Timestamp(now.date()), self.store_dir)
        prices = self.prices.get(sorted(feats["symbol"]))
        summary = N.news_summary(ctx.news, pd.Timestamp(now).tz_convert("UTC"))
        negative = set(summary.loc[summary["strong_negative"].astype(bool), "symbol"])
        strengths = PI.recent_strengths(model, self.store_dir, ctx, rules) \
            if rules.skip_quantile > 0 else None
        # Each model may blend in the other's ranking (learning from each other).
        close_model = self.close_model()
        trade_view = MI.Blended(model, close_model,
                                MI.current_params(MI.TRADE).get("peer_weight") or 0)
        r = PI.run_picks(self.conn, feats, trade_view, prices, pd.Timestamp(now.date()), rules,
                         self.capital_intraday, negative, strengths)
        if close_model is not None:              # second book: 9:45 -> 15:15, its own rules
            crules, _ = PI.get_rules(self.conn, PI.CLOSE_HORIZON)
            cstrengths = PI.recent_strengths(close_model, self.store_dir, ctx, crules) \
                if crules.skip_quantile > 0 else None
            close_view = MI.Blended(close_model, model,
                                    MI.current_params(MI.CLOSE).get("peer_weight") or 0)
            rc = PI.run_picks(self.conn, feats, close_view, prices, pd.Timestamp(now.date()),
                              crules, self.capital_intraday, negative, cstrengths,
                              horizon=PI.CLOSE_HORIZON)
            log.info("Intraday picks (until close): %s", "skip today (weak signal)"
                     if rc["skipped"] else "; ".join(rc["picks"]) or "none")
        late = now.time() > time(10, 15)
        msg = "skip today (weak signal)" if r["skipped"] else "; ".join(r["picks"])
        alert(self.conn, "paper-intraday", "decision", None,
              f"{day} intraday picks{' (late start, entered at ' + now.strftime('%H:%M') + ' prices)' if late else ''}: {msg}")
        log.info("Intraday picks made%s: %s", " (late start)" if late else "", msg)

    def square_off_job(self, now: datetime, book: str = PI.HORIZON) -> None:
        """Close `book`'s trades (12:30 book, or the 15:15 'until close' book) and judge its
        picks at these prices."""
        day = f"{now:%Y-%m-%d}"
        _set(self.conn, "id_last_squareoff" if book == PI.HORIZON else "id_close_squareoff", day)
        stamp = now.strftime("%Y-%m-%d %H:%M")
        opens = [r[0] for r in self.conn.execute(
            "SELECT symbol FROM intraday_open WHERE date = ?", (day,))]
        symbols = sorted(set(opens) | {t["symbol"] for t in PI.open_trades(self.conn, book)})
        if not symbols:
            return
        prices = self.prices.get(symbols)
        for line in PI.square_off(self.conn, prices, stamp, book):
            alert(self.conn, f"paper-{book}", "fill", line.split()[1], f"{stamp} {line}")
        PI.evaluate_day(self.conn, day, prices, book)
        self.print_scoreboard(now)

    def quarter_job(self, now: datetime) -> None:
        try:
            self.sync(self.store_dir)
            self._ctx = None
        except Exception as exc:
            log.warning("sync failed: %s", exc)
        self.refresh_models()
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
        self.print_scoreboard(now)

    def live_scores(self, now: datetime) -> None:
        """Provisional scores: today's live prices appended as a temporary daily candle."""
        ctx = self.ctx()
        active = store.tradable(ctx.universe)
        live = self.prices.get(active)
        if not live:
            return
        stamp_all = now.strftime("%Y-%m-%d %H:%M")
        self.conn.executemany("INSERT OR REPLACE INTO live_prices VALUES (?, ?, ?, ?)",
                              [(s_, p_, stamp_all, self.prices.source) for s_, p_ in live.items()])
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
        if latest.date() < now.date() and now.time() >= LOCAL_FETCH_AFTER:
            # GitHub's data job is late or didn't run: fetch today's prices ourselves.
            if self.local_catchup(now):
                ctx = self.ctx(reload=True)
                latest = ctx.daily["date"].max()
        waited_long = now.time() >= time(21, 0)
        if latest.date() < now.date() and not waited_long:
            return False   # today's data not published yet; try again in 15 minutes
        # Add today's real Angel One intraday data (more accurate than Yahoo).
        self.angel_topup(now)
        # Keep learning: retrain on the newest data before today's decision.
        try:
            lt = self.retrain_longterm(ctx)
            if lt is not None:
                self._model = lt
            if config.MAC_TRAINING:
                it = T.retrain_intraday(ctx, self.store_dir, self.conn)
                if it is not None:
                    self._imodel = it
                if T.retrain_intraday(ctx, self.store_dir, self.conn,
                                      target=MI.CLOSE) is not None:
                    self._cmodel = None
            else:
                self._model = None      # use the freshest model (Mac 16:00 or GitHub)
        except Exception:
            log.exception("daily retraining failed; using current models")
        results = D.run_daily(self.conn, ctx, self.capital, self.model())
        # Learn from paper trading: switch strategy if a variant clearly wins live.
        try:
            sel = T.live_selection(self.conn)
            if sel and sel["switched"]:
                self._model = None
                alert(self.conn, "model", "info", None,
                      f"{now:%Y-%m-%d} live paper results: switched long-term strategy from "
                      f"'{sel['current']}' to '{sel['best']}'")
        except Exception:
            log.exception("live strategy selection failed")
        for r in results:
            buys, sells = ", ".join(r["buys"]) or "none", ", ".join(s for s, _ in r["sells"]) or "none"
            alert(self.conn, "paper-longterm", "decision", None,
                  f"{r['date']:%Y-%m-%d} decision - buy: {buys}; sell: {sells}")
        self.push_feedback()
        _set(self.conn, "lt_after_close_day", f"{now:%Y-%m-%d}")
        if config.MAC_TRAINING:
            self.evening_tune(now)
        return True

    # --- cloud training (GitHub Actions) ---------------------------------------------
    def retrain_longterm(self, ctx) -> M.LongTermModel | None:
        """On the Mac only when cloud training is off; otherwise fetch GitHub's newest model."""
        if not config.CLOUD_TRAINING:
            return T.retrain_longterm(ctx, self.conn)
        self.refresh_models()
        return None

    def refresh_models(self) -> None:
        """Download the newest long-term and news models trained on GitHub (if changed)."""
        if not config.CLOUD_TRAINING:
            return
        try:
            rev = store.sync_models()
        except Exception as exc:
            log.warning("could not download models from GitHub: %s", exc)
            return
        if rev and rev != self._models_rev:
            if self._models_rev is not None:
                log.info("New models from GitHub training (%s)", rev[:7])
            self._models_rev = rev
            self._model = None
            self._ctx = None          # reload so the new news-relevance model is applied

    def push_feedback(self) -> None:
        """Send judged paper predictions to GitHub so cloud training keeps learning from them."""
        if not config.CLOUD_TRAINING:
            return
        judged = T.judged_predictions(self.conn)
        intraday = pd.read_sql("SELECT horizon, symbol, date, correct FROM predictions WHERE "
                               "horizon LIKE 'intraday%' AND correct IS NOT NULL", self.conn)
        if judged.empty and intraday.empty:
            return
        key = f"{len(judged)}:{judged['date'].max()}:{len(intraday)}:{intraday['date'].max()}"
        if _setting(self.conn, "feedback_pushed") == key:
            return
        try:
            store.push_feedback(judged, intraday=intraday)
            _set(self.conn, "feedback_pushed", key)
            log.info("Sent %d long-term and %d intraday judged paper predictions to GitHub "
                     "for training", len(judged), len(intraday))
        except Exception as exc:
            log.warning("could not send paper feedback to GitHub: %s", exc)

    def evening_tune(self, now: datetime) -> None:
        """Keep training: a light self-tuning round every weekday evening (2 new settings
        per model); the full round (5 settings) runs at the weekend."""
        if _setting(self.conn, "last_light_tune") == f"{now:%Y-%m-%d}":
            return
        _set(self.conn, "last_light_tune", f"{now:%Y-%m-%d}")
        self._tune_round(now, 2, "evening")

    def local_catchup(self, now: datetime) -> bool:
        """Download recent daily prices (and intraday summaries) directly from Yahoo."""
        try:
            from stockpredictor.data import daily as DD
            from stockpredictor.data import intraday as I

            ctx = self.ctx()
            symbols = ctx.universe.loc[ctx.universe["active"] == 1, "symbol"].tolist()
            stocks, idx = DD.fetch_recent(symbols)
            if stocks.empty:
                return False
            git_last = f"{ctx.daily['date'].max():%Y-%m-%d}"
            new_stocks = stocks[stocks["date"] > git_last]
            store.save_local_overlay("daily", new_stocks)
            store.save_local_overlay("indices", idx[idx["date"] > git_last])
            bars = I.yahoo_bars(symbols, period="5d")
            I.upsert_backfill([r for s_, b in bars.items() for r in I.summarize(b, s_, "yahoo")])
            log.info("Fetched %s new daily rows from Yahoo (GitHub data was late)", len(new_stocks))
            return not new_stocks.empty
        except Exception:
            log.exception("local data catch-up failed")
            return False

    def daily_mac_train(self, ctx) -> None:
        """Once a day after the close: retrain the long-term model (on the Mac's recent
        history) and both intraday models (with today's session), using GitHub's tuned
        settings. No tuning, a few minutes. GitHub's evening run then trains again on all
        history since 2005; the freshest model is always used."""
        try:
            lt = T.retrain_longterm(ctx, self.conn, model_dir=M.MAC_MODEL_DIR)
            if lt is not None:
                log.info("Daily Mac training: long-term model retrained up to %s", lt.train_to)
        except Exception:
            log.exception("daily Mac long-term training failed")
        for target in MI.TARGETS:
            model = T.retrain_intraday(ctx, self.store_dir, self.conn, target=MI.on_mac(target))
            if model is not None:
                log.info("Daily Mac training: %s model retrained on %s days up to %s",
                         target.label, model.train_days, model.train_to)
        self._imodel = self._cmodel = self._model = None

    def angel_topup(self, now: datetime) -> None:
        settings = getattr(self.prices, "settings", None)
        if settings is None or not settings.angel.is_complete \
                or _setting(self.conn, "angel_topup_day") == f"{now:%Y-%m-%d}":
            return
        try:
            from stockpredictor.data import intraday as I

            client = self.prices._angel_client()
            ctx = self.ctx()
            symbols = ctx.universe.loc[ctx.universe["active"] == 1, "symbol"].tolist()
            n = I.angel_backfill(client, self.prices._tokens, symbols, now.date(), now.date(),
                                 progress=lambda m: None)
            from stockpredictor.data import fine

            nf = fine.backfill(client, self.prices._tokens, symbols, now.date(), now.date(),
                               progress=lambda m: None)
            _set(self.conn, "angel_topup_day", f"{now:%Y-%m-%d}")
            log.info("Angel One: saved today's intraday summaries for %s stocks (1-minute "
                     "features for %s)", n, nf)
        except Exception:
            log.exception("Angel One daily top-up failed")

    def weekly_retrain(self, now: datetime) -> bool:
        """Weekend self-tuning: try new model settings, keep them only if they test better."""
        if not config.MAC_TRAINING:
            return False
        last = _setting(self.conn, "last_tune")
        if last and (now.replace(tzinfo=None) - datetime.fromisoformat(last)).days < 6:
            return False
        _set(self.conn, "last_tune", now.replace(tzinfo=None).isoformat(timespec="seconds"))
        ctx = self.ctx(reload=True)
        if not config.CLOUD_TRAINING:        # otherwise GitHub tunes the long-term model
            lt = T.tune_longterm(ctx, self.conn)
            self._model = None
            msg = (f"{now:%Y-%m-%d} long-term tuning: prediction quality (IC) {lt['ic']:.3f}"
                   + (f", improved from {lt['previous_ic']:.3f} - new settings adopted"
                      if lt["adopted"] else " - current settings kept"))
            alert(self.conn, "model", "info", None, msg)
        it = T.tune_intraday(ctx, self.store_dir, self.conn)
        self._imodel = None
        if it:
            alert(self.conn, "model", "info", None,
                  f"{now:%Y-%m-%d} intraday tuning: IC {it['ic']:.3f}"
                  + (" - new settings adopted" if it["adopted"] else " - current settings kept"))
        return True


def run_forever(monitor: Monitor, interval: int = 60, until: time | None = None) -> None:
    import time as _time

    log.info("Live monitor started (Ctrl+C to stop)")
    while True:
        if until is not None and monitor.clock().time() >= until \
                and _setting(monitor.conn, "lt_after_close_day") == f"{monitor.clock():%Y-%m-%d}":
            log.info("Day's work done and it is past %s - stopping.", until.strftime("%H:%M"))
            return
        started = _time.monotonic()
        try:
            done = monitor.tick()
            if done:
                log.info("%s ran: %s", monitor.clock().strftime("%H:%M"), ", ".join(done))
        except Exception:
            log.exception("monitor tick failed; continuing")
        _time.sleep(max(1.0, interval - (_time.monotonic() - started)))
