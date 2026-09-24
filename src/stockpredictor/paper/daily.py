"""The daily after-close job for the long-term side, with catch-up."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from stockpredictor.backtest.portfolio import Rules
from stockpredictor.models import longterm as M
from stockpredictor.paper import engine as E

MAX_CATCHUP_DAYS = 30


def load_or_train(ctx: E.MarketContext, model_dir: Path = M.MODEL_DIR) -> M.LongTermModel:
    if (model_dir / "model.txt").exists():
        return M.LongTermModel.load(model_dir)
    from stockpredictor.features import longterm as F

    model = M.LongTermModel.train(M.add_labels(F.weekly_snapshots(ctx.feats), ctx.daily, ctx.indices))
    model.save(model_dir)
    return model


def get_rules(conn: sqlite3.Connection) -> tuple[Rules, str]:
    s = dict(conn.execute("SELECT key, value FROM app_settings").fetchall())
    rules = Rules(n_hold=int(s.get("lt_n_hold", 10)), exit_rank=int(s.get("lt_exit_rank", 30)),
                  stop_loss=float(s.get("lt_stop_loss", 0.15)),
                  news_exit=s.get("lt_news_exit", "1") == "1")
    return rules, s.get("lt_rebalance", "weekly")


def run_daily(conn: sqlite3.Connection, ctx: E.MarketContext, capital: float,
              model: M.LongTermModel | None = None) -> list[dict]:
    """Run decisions for every trading day since the last one (catch-up), then evaluate."""
    E.ensure_account(conn, capital)
    model = model or load_or_train(ctx)
    rules, mode = get_rules(conn)
    last = conn.execute("SELECT value FROM app_settings WHERE key = 'lt_last_decision'").fetchone()
    days = sorted(ctx.feats["date"].unique())
    if last:
        todo = [d for d in days if d > pd.Timestamp(last[0])][-MAX_CATCHUP_DAYS:]
    else:
        todo = days[-1:]
    results = [E.run_decision(conn, ctx, model, d, rules, mode) for d in todo]
    E.evaluate_predictions(conn, ctx)
    return results
