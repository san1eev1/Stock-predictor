"""Background training on the Mac, alongside the live monitor.

A separate thread (own database connection; LightGBM trains in C++ outside Python's lock)
so live prices keep updating every minute while it works:

  * all day, market hours included: tuning rounds for the intraday model on its history
    (the long-term and news models are trained on GitHub; see cloud-train), keeping new
    settings only when they test better out-of-sample
  * after the 15:30 close: today's whole live Angel One 5-minute session is added to the
    history and the intraday model retrains on it (trading stops at 12:30, learning doesn't)
"""

from __future__ import annotations

import logging
import threading
from datetime import date
from pathlib import Path

from stockpredictor import config, db
from stockpredictor.models import intraday as MI
from stockpredictor.models import trainer as T

log = logging.getLogger(__name__)
PAUSE_MIN = 5            # rest between tuning rounds
CANDIDATES = 3           # new settings tried per round


class BackgroundTrainer(threading.Thread):
    def __init__(self, db_path: Path, store_dir: Path, settings=None,
                 pause_min: float = PAUSE_MIN):
        super().__init__(daemon=True, name="background-training")
        self.db_path, self.store_dir, self.settings = db_path, store_dir, settings
        self.pause = pause_min * 60
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.model_changed = threading.Event()     # the monitor reloads the intraday model
        self.ready = threading.Event()             # set by the monitor once startup is done
        self._live_day: date | None = None
        self.rounds = 0

    # --- called from the monitor thread -------------------------------------------------
    def learn_from_today(self, day: date) -> None:
        """Ask for today's live session to be added and the intraday model retrained."""
        self._live_day = day
        self.wake.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake.set()

    # --- the thread ---------------------------------------------------------------------
    def run(self) -> None:
        conn = db.connect(self.db_path)
        self.ready.wait(timeout=30 * 60)           # let the monitor's startup sync finish first
        log.info("Background training started (keeps learning from history all day)")
        if not T.COMPARE_PATH.exists():
            self._compare(conn)
        while not self.stop_event.is_set():
            if self._live_day is not None:
                day, self._live_day = self._live_day, None
                self._learn_live(conn, day)
                self._compare(conn)                # daily: which exit is better so far
            else:
                self._tune(conn)
            self.wake.wait(self.pause)
            self.wake.clear()

    def _ctx(self):
        from stockpredictor import store
        from stockpredictor.paper.engine import MarketContext

        with store.DATA_LOCK:                      # not while the monitor refreshes the data
            return MarketContext.load(self.store_dir)

    def _tune(self, conn) -> None:
        try:
            ctx = self._ctx()
            # Alternate: the 12:30 (traded) model, then the model that learns until the close.
            target = MI.TARGETS[self.rounds % len(MI.TARGETS)]
            with MI.MODEL_LOCK:
                it = T.tune_intraday(ctx, self.store_dir, conn, n_candidates=CANDIDATES,
                                     log_all=False, target=target)
            self.rounds += 1
            if it is None:
                log.info("Background training: not enough intraday history (%s) to tune yet",
                         target.label)
                return
            log.info("Background training round %d (%s): IC %.4f (current %.4f)%s",
                     self.rounds, target.label, it["ic"], it["previous_ic"],
                     " -> new settings adopted" if it["adopted"] else "")
            if it["adopted"]:
                self.model_changed.set()
            if not config.CLOUD_TRAINING:
                lt = T.tune_longterm(ctx, conn, n_candidates=CANDIDATES, log_all=False)
                log.info("Background training: long-term IC %.4f%s", lt["ic"],
                         " -> new settings adopted" if lt["adopted"] else "")
        except Exception:
            log.exception("background training round failed")

    def _compare(self, conn) -> None:
        """Refresh the 12:30-vs-close comparison (walk-forward, same days, costs)."""
        try:
            from stockpredictor.paper import intraday as PI

            rules, _ = PI.get_rules(conn)
            r = T.compare_exits(self._ctx(), self.store_dir, rules)
            if r:
                a, b = r[MI.TRADE.horizon], r[MI.CLOSE.horizon]
                log.info("Exit comparison (%s): 12:30 avg day Rs %+.0f, accuracy %.0f%% | close "
                         "avg day Rs %+.0f, accuracy %.0f%%", r["period"], a["avg_day_pnl"],
                         a["direction_accuracy"] * 100, b["avg_day_pnl"],
                         b["direction_accuracy"] * 100)
        except Exception:
            log.exception("exit comparison failed")

    def _learn_live(self, conn, day: date) -> None:
        """Today's 5-minute session from Angel One -> intraday history -> retrain."""
        try:
            if self.settings is not None and self.settings.angel.is_complete:
                from stockpredictor import store
                from stockpredictor.data import angelone
                from stockpredictor.data import intraday as I

                client = angelone.AngelDataClient(self.settings.angel)
                client.login()
                uni = store.load_universe(self.store_dir)
                symbols = uni.loc[uni["active"] == 1, "symbol"].tolist()
                n = I.angel_backfill(client, angelone.fetch_nse_equity_tokens(), symbols,
                                     day, day, progress=lambda m: None)
                log.info("Live learning: added today's session for %d stocks", n)
            ctx = self._ctx()
            for target in MI.TARGETS:
                with MI.MODEL_LOCK:
                    model = T.retrain_intraday(ctx, self.store_dir, conn, force=True,
                                               target=target)
                if model is not None:
                    self.model_changed.set()
                    log.info("Live learning: %s model retrained with today's session "
                             "(%d days, up to %s)", target.label, model.train_days, model.train_to)
        except Exception:
            log.exception("live learning failed")
