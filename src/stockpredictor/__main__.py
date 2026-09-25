"""Command line entry point: python -m stockpredictor <command>."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from stockpredictor import config, db, universe
from stockpredictor.config import load_settings


def cmd_init(settings) -> None:
    db.init_db(settings.db_path)
    print(f"Database ready: {settings.db_path}")


def cmd_universe(settings) -> None:
    stocks = universe.fetch_universe()
    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as conn:
        universe.save_universe(conn, stocks)
    n_trade = sum(s["tradable"] for s in stocks)
    print(f"Saved {len(stocks)} stocks for training, {n_trade} tradable (Nifty LargeMidcap 250).")


def cmd_tokens(settings) -> None:
    from stockpredictor.data import angelone

    tokens = angelone.fetch_nse_equity_tokens()
    with db.connect(settings.db_path) as conn:
        angelone.save_tokens(conn, tokens)
        missing = [r["symbol"] for r in conn.execute(
            "SELECT symbol FROM stocks WHERE active = 1 AND angel_token IS NULL")]
    print(f"Mapped Angel One tokens. Missing: {missing or 'none'}")


def cmd_check_angel(settings) -> None:
    from stockpredictor.data import angelone

    client = angelone.AngelDataClient(settings.angel)
    client.login()
    with db.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT symbol, angel_token FROM stocks WHERE angel_token IS NOT NULL "
            "ORDER BY symbol LIMIT 1").fetchone()
    if row is None:
        print("Login OK. Run `tokens` first to test a live price.")
        return
    print(f"Login OK. {row['symbol']} LTP: {client.ltp(row['symbol'], row['angel_token'])}")


def cmd_prices(settings, args) -> None:
    from stockpredictor.data import daily

    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as conn:
        symbols = args.symbols or universe.active_symbols(conn)
        if not symbols:
            print("No stocks yet — run `universe` first.")
            return
        results = daily.update_all(conn, symbols, date.fromisoformat(args.start),
                                   extend=args.extend_back)
    failed = {k: v for k, v in results.items() if v.startswith("error")}
    print(f"Done. {len(results) - len(failed)} updated, {len(failed)} failed.")
    for name, err in failed.items():
        print(f"  {name}: {err}")


def cmd_intraday_collect(settings, args) -> None:
    """Yahoo 5-minute bars -> daily summaries in the git store (run by GitHub Actions)."""
    from stockpredictor import store
    from stockpredictor.data import intraday

    d = Path(args.dir)
    uni = store.load_universe(d)
    symbols = uni.loc[uni["active"] == 1, "symbol"].tolist()
    bars = intraday.yahoo_bars(symbols, period=args.period)
    rows = [r for s, b in bars.items() for r in intraday.summarize(b, s, "yahoo")]
    n = intraday.upsert_store(d, rows)
    print(f"Intraday summaries: {n} stock-days from {len(bars)}/{len(symbols)} stocks")


def cmd_delivery_update(settings, args) -> None:
    """NSE delivery share per stock and day -> delivery/<year>.csv in the git store."""
    from stockpredictor import store
    from stockpredictor.data import delivery

    d = Path(args.dir)
    uni = store.load_universe(d)
    n = delivery.update(d, set(uni["symbol"]), date.fromisoformat(args.start),
                        minutes=args.minutes)
    print(f"Delivery data: {n} new days")


def cmd_earnings_update(settings, args) -> None:
    """Quarterly results dates (Yahoo) -> earnings.csv in the git store."""
    from stockpredictor import store
    from stockpredictor.data import earnings

    d = Path(args.dir)
    n = earnings.update(d, store.load_universe(d)["symbol"].tolist())
    print(f"Results dates: {n} announcements stored")


def cmd_preopen_update(settings, args) -> None:
    """Today's NSE pre-open auction -> preopen/<year>.csv in the git store."""
    from stockpredictor import store
    from stockpredictor.data import preopen

    d = Path(args.dir)
    n = preopen.update(d, set(store.load_universe(d)["symbol"]))
    print(f"Pre-open auction: {n} stocks saved")


def _delivery_args(p) -> None:
    _dir_arg(p)
    p.add_argument("--start", default="2005-01-01", help="Oldest day to download")
    p.add_argument("--minutes", type=float, default=20, help="Time budget (resumes next run)")


def cmd_intraday_backfill(settings, args) -> None:
    """Angel One 5-minute history -> local summaries (one-time, on the Mac)."""
    from stockpredictor import store
    from stockpredictor.data import angelone, intraday

    d = Path(args.dir)
    uni = store.load_universe(d)
    symbols = args.symbols or uni.loc[uni["active"] == 1, "symbol"].tolist()
    client = angelone.AngelDataClient(settings.angel)
    tokens = angelone.fetch_nse_equity_tokens()
    start, end = date.today() - timedelta(days=args.days), date.today() - timedelta(days=1)
    total = intraday.angel_backfill(client, tokens, symbols, start, end,
                                    progress=lambda m: print(m, flush=True))
    print(f"Saved {total} stock-days to {intraday.BACKFILL_PATH}")


def cmd_data_check(settings, args) -> None:
    from stockpredictor.data import quality

    with db.connect(settings.db_path) as conn:
        symbols = args.symbols or universe.active_symbols(conn)
        print(quality.format_report(quality.daily_report(conn, symbols)))


def cmd_export_store(settings, args) -> None:
    from stockpredictor import store

    with db.connect(settings.db_path) as conn:
        store.export_store(conn, Path(args.dir))
    print(f"Exported market data to {args.dir}")


def cmd_import_store(settings, args) -> None:
    from stockpredictor import store

    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as conn:
        store.import_store(conn, Path(args.dir))
    print(f"Imported market data from {args.dir}")


def cmd_sync_data(settings, args) -> None:
    from stockpredictor import store

    store.sync(Path(args.dir))
    daily = store.load_daily(Path(args.dir))
    print(f"Synced {daily['symbol'].nunique()} stocks, {len(daily):,} daily rows, "
          f"latest {daily['date'].max():%Y-%m-%d}")


def cmd_features(settings, args) -> None:
    from stockpredictor import store
    from stockpredictor.features import longterm

    d = Path(args.dir)
    feats = longterm.build_features(store.load_daily(d), store.load_indices(d),
                                    store.load_universe(d))
    cols = longterm.feature_columns(feats)
    print(f"{feats['symbol'].nunique()} stocks, {len(feats):,} rows, {len(cols)} features, "
          f"{feats['date'].min():%Y-%m-%d} to {feats['date'].max():%Y-%m-%d}")
    print(f"Weekly training rows: {len(longterm.weekly_snapshots(feats)):,}")
    coverage = feats[cols].notna().mean().sort_values()
    print("Lowest coverage:", ", ".join(f"{c} {v:.0%}" for c, v in coverage.head(5).items()))
    if args.symbol:
        row = longterm.latest(feats).set_index("symbol").loc[args.symbol]
        print(f"\nLatest features for {args.symbol} ({row['date']:%Y-%m-%d}):")
        for c in cols:
            print(f"  {c:<22}{row[c]:.4f}")


def cmd_news_update(settings, args) -> None:
    from stockpredictor import store
    from stockpredictor.data import news

    d = Path(args.dir)
    uni = store.load_universe(d)
    rows, errors = news.fetch_all(uni[uni["symbol"].isin(store.tradable(uni))])
    new = news.append_news(d, rows)
    print(f"Fetched {len(rows)} headlines, {len(new)} new, {len(errors)} feed errors")
    for e in errors[:10]:
        print("  ", e)
    if not args.no_score:
        from stockpredictor.nlp.sentiment import FinBertScorer

        print(f"Scored {news.score_missing(d, FinBertScorer())} headlines with FinBERT")


def cmd_fundamentals_update(settings, args) -> None:
    from stockpredictor import store
    from stockpredictor.data import fundamentals

    d = Path(args.dir)
    uni = store.load_universe(d)
    rows, errors = fundamentals.fetch_snapshot(store.tradable(uni))
    fundamentals.append_snapshot(d, rows)
    print(f"Fundamentals snapshot: {len(rows)} stocks, {len(errors)} errors")


def cmd_train(settings, args) -> None:
    from stockpredictor.models import trainer as T
    from stockpredictor.paper.engine import MarketContext

    ctx = MarketContext.load(Path(args.dir))
    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as conn:
        model = T.retrain_longterm(ctx, conn, force=True)
    print(f"Long-term model trained on outcomes known up to {model.train_to}")
    print("Top features:", ", ".join(model.importance().head(8).index))


def cmd_improve(settings, args) -> None:
    """Get the newest data, retrain both models and (optionally) self-tune them."""
    from stockpredictor import store
    from stockpredictor.models import trainer as T
    from stockpredictor.paper.engine import MarketContext

    d = Path(args.dir)
    if not args.no_sync:
        store.sync(d)
    ctx = MarketContext.load(d)
    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as conn:
        print("Retraining on the newest data...")
        T.retrain_longterm(ctx, conn, force=True)
        it = T.retrain_intraday(ctx, d, conn)
        print("  long-term: done;", "intraday: done" if it else "intraday: up to date / not enough data")
        if args.tune:
            print(f"Self-tuning with {args.candidates} candidate settings each "
                  "(walk-forward, can take several minutes)...")
            lt = T.tune_longterm(ctx, conn, n_candidates=args.candidates)
            print(f"  long-term: IC {lt['ic']:.3f} (was {lt['previous_ic']:.3f}), top-10 beat "
                  f"Nifty {lt['top10_hit']:.0%} -> {'ADOPTED new settings' if lt['adopted'] else 'kept'}")
            it = T.tune_intraday(ctx, d, conn, n_candidates=args.candidates)
            if it:
                print(f"  intraday:  IC {it['ic']:.3f} (was {it['previous_ic']:.3f}), direction "
                      f"accuracy {it['direction_accuracy']:.0%} -> "
                      f"{'ADOPTED new settings' if it['adopted'] else 'kept'}")
            else:
                print("  intraday:  not enough data to tune yet")


def cmd_backtest(settings, args) -> None:
    from stockpredictor.backtest import run as bt
    from stockpredictor.models import longterm as M

    report = bt.run(Path(args.dir), start_year=args.start_year,
                    capital=settings.paper_capital_longterm)
    print(bt.format_report(report))
    bt.save(report, M.MODEL_DIR / "backtest.json")
    print(f"\nSaved report to {M.MODEL_DIR / 'backtest.json'}")


def cmd_daily(settings, args) -> None:
    from stockpredictor import store
    from stockpredictor.paper import daily, engine

    d = Path(args.dir)
    if not args.no_sync:
        store.sync(d)
    ctx = engine.MarketContext.load(d)
    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as conn:
        results = daily.run_daily(conn, ctx, settings.paper_capital_longterm)
        if not results:
            print("Already up to date.")
        for r in results:
            print(f"\n== {r['date']:%Y-%m-%d} "
                  f"{'(rebalance)' if r.get('rebalance') else ''}")
            for f in r["fills"]:
                print("  filled:", f)
            for sym, why in r["sells"]:
                print(f"  queued SELL {sym} ({why})")
            for sym in r["buys"]:
                print(f"  queued BUY  {sym}")
        if results and "value" in results[-1]:
            v = results[-1]["value"]
            print(f"\nPaper long-term: equity Rs {v['equity']:,.0f} "
                  f"(P&L Rs {v['pnl']:+,.0f}), {v['positions']} positions, cash Rs {v['cash']:,.0f}")
        top = conn.execute(
            "SELECT symbol, confidence, reasons FROM predictions WHERE horizon = 'longterm' "
            "AND direction = 'up' AND date = (SELECT MAX(date) FROM predictions) ORDER BY rank"
        ).fetchall()
        print("\nTop picks:", ", ".join(r["symbol"] for r in top))


def cmd_run(settings, args) -> None:
    import logging

    from stockpredictor.live import background, monitor, prices

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db.init_db(settings.db_path)
    conn = db.connect(settings.db_path)
    mon = monitor.Monitor(conn, Path(args.dir), prices.LivePrices(settings),
                          settings.paper_capital_longterm,
                          capital_intraday=settings.paper_capital_intraday)
    if config.MAC_TRAINING:
        mon.background = background.BackgroundTrainer(settings.db_path, Path(args.dir), settings)
        mon.background.start()
    try:
        monitor.run_forever(mon)
    except KeyboardInterrupt:
        print("Stopped.")


def cmd_auto(settings, args) -> None:
    """One background pass (run daily by `schedule install`): decide + retrain, tune at weekends."""
    import logging
    from datetime import datetime

    from stockpredictor.live import monitor, prices

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as conn:
        mon = monitor.Monitor(conn, Path(args.dir), prices.LivePrices(settings),
                              settings.paper_capital_longterm,
                              capital_intraday=settings.paper_capital_intraday)
        done = mon.scheduled_run(mon.clock())
    print(f"{datetime.now():%Y-%m-%d %H:%M} auto: {', '.join(done) or 'nothing to do'}", flush=True)


def cmd_cloud_train(settings, args) -> None:
    """GitHub Actions: retrain the long-term and news models on all history (2005 onwards),
    then keep self-tuning until the time budget is used. Output goes to trained-models/,
    which the workflow publishes to the `models` branch for the Mac to download."""
    import json
    import time
    from datetime import datetime

    from stockpredictor import store
    from stockpredictor.backtest import run as bt
    from stockpredictor.config import SHARED_MODELS_DIR
    from stockpredictor.models import intraday as MI
    from stockpredictor.models import longterm as M
    from stockpredictor.models import trainer as T
    from stockpredictor.nlp import relevance as R
    from stockpredictor.paper.engine import MarketContext

    started = time.monotonic()
    budget = args.minutes * 60
    d = Path(args.dir)
    T.RUN_LOG = SHARED_MODELS_DIR / "runs.jsonl"
    if args.feedback:
        T.FEEDBACK = store.load_feedback(Path(args.feedback))
    print(f"Paper feedback: {0 if T.FEEDBACK is None else len(T.FEEDBACK)} judged predictions",
          flush=True)
    ctx = MarketContext.load(d)
    print(f"Data: {ctx.daily['symbol'].nunique()} stocks, {len(ctx.daily):,} daily rows, "
          f"{ctx.daily['date'].min():%Y-%m-%d} to {ctx.daily['date'].max():%Y-%m-%d}", flush=True)

    model = T.retrain_longterm(ctx, None, force=True)
    print(f"Long-term model retrained on outcomes up to {model.train_to}", flush=True)
    news = R.train(ctx.news, ctx.universe, ctx.daily, ctx.indices)
    print("News relevance: " + (f"{news['headlines']} headlines, out-of-sample corr "
                                f"{news['test_corr']:.3f} (tone only {news['baseline_sentiment_corr']:.3f})"
                                if news else "not enough headlines yet"), flush=True)

    bt_path = M.MODEL_DIR / "backtest.json"
    # Long-term paper trading on history (walk-forward since 2015, the live paper rules:
    # the 5 best held), once a day on the first run after the 15:30 close.
    from stockpredictor.live.prices import now_ist

    ist = now_ist()
    close = ist.replace(hour=15, minute=30, second=0, microsecond=0)
    last_close = close if ist >= close else close - timedelta(days=1)
    try:     # the file's own time is the clone time on GitHub: use the time saved inside it
        done = datetime.fromisoformat(json.loads(bt_path.read_text())["updated"])
    except (OSError, KeyError, ValueError):
        done = None
    if args.backtest or done is None or done < last_close:
        report = bt.run(d, capital=settings.paper_capital_longterm)
        report["updated"] = ist.isoformat(timespec="seconds")
        bt.save(report, bt_path)
        s = report["strategies"]
        print("Long-term historical paper trading: " + "; ".join(
            f"{k}: {v['cagr']:.1%}/yr, Sharpe {v['sharpe']:.2f}, worst fall {v['max_drawdown']:.1%}"
            for k, v in s.items()), flush=True)

    conn = _cloud_intraday(settings, d, ctx, args, started + budget * 0.4)

    rounds = adopted = 0
    scores: dict = {}               # settings already scored this run are not re-tested
    while time.monotonic() - started < budget:
        rounds += 1
        # Take turns: long-term, then each intraday model (when its history is available).
        turn = ["longterm", *([t.horizon for t in MI.TARGETS] if conn is not None else [])]
        which = turn[(rounds - 1) % len(turn)]
        if which == "longterm":
            r = T.tune_longterm(ctx, None, n_candidates=args.candidates, log_all=False,
                                cache=scores)
            print(f"Tuning round {rounds} (long-term): IC {r['ic']:.4f} (current "
                  f"{r['previous_ic']:.4f}), paper check: the 5 stocks bought beat Nifty "
                  f"{r['top5_hit']:.1%} of weeks, avg {r['top5_excess']:+.2%}/week "
                  f"(top 10: {r['top10_hit']:.1%})"
                  + (" -> new settings adopted" if r["adopted"] else ""), flush=True)
        else:
            target = next(t for t in MI.TARGETS if t.horizon == which)
            r = T.tune_intraday(ctx, d, conn, n_candidates=args.candidates, log_all=False,
                                target=target)
            if r is None:
                continue
            p = r.get("paper") or {}
            print(f"Tuning round {rounds} (intraday {target.label}): IC {r['ic']:.4f} (current "
                  f"{r['previous_ic']:.4f}), paper check: Rs {p.get('avg_day_pnl', 0):+.0f}/day, "
                  f"{(p.get('accuracy') or 0):.0%} picks right"
                  + (" -> new settings adopted" if r["adopted"] else ""), flush=True)
        adopted += int(bool(r["adopted"]))
    if conn is not None:
        _export_history(conn, SHARED_MODELS_DIR / "history")

    if T.RUN_LOG.exists():   # keep the log small
        lines = T.RUN_LOG.read_text().splitlines()[-2000:]
        T.RUN_LOG.write_text("\n".join(lines) + "\n")
    status = {"updated": datetime.now().isoformat(timespec="seconds"),
              "data_to": f"{ctx.daily['date'].max():%Y-%m-%d}", "tuning_rounds": rounds,
              "adopted": adopted, "minutes": round((time.monotonic() - started) / 60, 1)}
    (SHARED_MODELS_DIR / "status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status), flush=True)


def _cloud_intraday(settings, d: Path, ctx, args, deadline: float):
    """GitHub: Angel One 5-minute history (kept in GitHub's private cache, never committed),
    then both intraday models: retrain on all of it + judged paper picks, choose the trading
    rules by profit, 3 rounds of historical paper-trading replays, 12:30-vs-close comparison.
    Returns a database connection holding the run's results (None without Angel One keys)."""
    import tempfile

    import pandas as pd

    from stockpredictor import store
    from stockpredictor.data import intraday as I
    from stockpredictor.models import intraday as MI
    from stockpredictor.models import trainer as T

    if not settings.angel.is_complete:
        print("Intraday: no Angel One keys (GitHub Secrets) - intraday models not trained",
              flush=True)
        return None
    if args.feedback:
        T.INTRADAY_FEEDBACK = store.load_intraday_feedback(Path(args.feedback))
    n_fb = 0 if T.INTRADAY_FEEDBACK is None else len(T.INTRADAY_FEEDBACK)
    print(f"Intraday paper feedback: {n_fb} judged buy/sell picks", flush=True)
    active = ctx.universe.loc[ctx.universe["active"] == 1, "symbol"].tolist()
    try:
        from stockpredictor.data import angelone

        client = angelone.AngelDataClient(settings.angel)
        client.login()
        tokens = angelone.fetch_nse_equity_tokens()
        have = I.backfill_symbols()
        # Stocks with no or incomplete history: 2 years. Everyone: the days since the last run.
        todo = sorted((set(active) - have) | (set(active) & (I.backfill_missing_exit()
                                                              | I.backfill_incomplete())))
        today = date.today()
        if todo:
            n = I.angel_backfill(client, tokens, todo, today - timedelta(days=730), today,
                                 progress=lambda m: None, deadline=deadline)
            print(f"Angel One history: {n} stock-days for {len(todo)} stocks with little or no "
                  "history", flush=True)
        last = I.load_summaries(d).query("source == 'angelone'")["date"].max()
        if pd.notna(last) and last.date() < today:
            rest = sorted(set(active) - set(todo))
            n = I.angel_backfill(client, tokens, rest, last.date() + timedelta(days=1), today,
                                 progress=lambda m: None, deadline=deadline + 20 * 60)
            print(f"Angel One history: {n} new stock-days since {last:%Y-%m-%d}", flush=True)
        # 1-minute opening features (data/fine.py): 2 years for stocks without them, then the
        # days since the last run; resumes next run when the time budget is used.
        from stockpredictor.data import fine

        cov = fine.coverage()
        need = [s for s in active if cov.get(s, 0) < 0.8 * I.backfill_days()]
        if need:
            n = fine.backfill(client, tokens, need, today - timedelta(days=730), today,
                              progress=lambda m: None, deadline=deadline + 20 * 60)
            print(f"1-minute history: {n} stock-days for {len(need)} stocks", flush=True)
        last_fine = fine.load()["date"].max()
        if pd.notna(last_fine) and last_fine.date() < today:
            rest = [s for s in active if s not in need]
            n = fine.backfill(client, tokens, rest, last_fine.date() + timedelta(days=1), today,
                              progress=lambda m: None, deadline=deadline + 30 * 60)
            print(f"1-minute history: {n} new stock-days", flush=True)
    except Exception as exc:
        print(f"Angel One download failed ({exc}); training on the history already cached",
              flush=True)
    have = len(I.backfill_symbols())
    print(f"Intraday history: {I.backfill_days()} days, {have} stocks", flush=True)
    if have < 0.8 * len(active):
        # A model trained on a fraction of the stocks is worse than the Mac's own: publish
        # none (the Mac then uses its models) until the download is complete.
        import shutil

        for t in MI.TARGETS:
            shutil.rmtree(t.model_dir, ignore_errors=True)
        print(f"Intraday: history for only {have} of {len(active)} stocks so far - intraday "
              "training postponed (download continues next run)", flush=True)
        return None

    path = Path(tempfile.mkdtemp()) / "cloud.db"
    db.init_db(path)
    conn = db.connect(path)
    for target in MI.TARGETS:
        feats = T.intraday_feedback(None, T.intraday_feats(ctx, d, target), target.horizon)
        model = T.retrain_intraday(ctx, d, conn, force=True, target=target)
        if model is None:
            print(f"Intraday ({target.label}): not enough history yet", flush=True)
            continue
        print(f"Intraday ({target.label}): retrained on {model.train_days} days up to "
              f"{model.train_to}", flush=True)
        rep = T.tune_rules(feats, target)
        if rep:
            r = rep["rules"]
            print(f"Intraday ({target.label}) trading rules by profit: {r['n_long']} buys + "
                  f"{r['n_short']} sells, stop {r['stop_loss']}%, target {r['target'] or 'none'}"
                  f"%, skip weakest {r['skip_quantile']:.0%} of days -> Rs "
                  f"{rep['day_profit_recent']:+.0f}/day recently, Rs {rep['day_profit_before']:+.0f}"
                  f"/day before" + (" (new)" if rep["adopted"] else " (kept)"), flush=True)
    out = T.replay_training(ctx, d, conn)
    for target in MI.TARGETS:
        r = out.get(target.horizon)
        if r:
            print(f"Historical replay ({target.label}): " + " -> ".join(
                f"round {x['round']}: Rs {x['avg_day_pnl']:+.0f}/day, {x['accuracy']:.0%} right"
                for x in r["rounds"]) + (" -> learning kept" if r["adopted"] else
                                         " -> not used"), flush=True)
            if r["adopted"]:
                T.retrain_intraday(ctx, d, conn, force=True, target=target)
    cmp = T.compare_exits(ctx, d)
    if cmp:
        a, b = cmp[MI.TRADE.horizon], cmp[MI.CLOSE.horizon]
        print(f"Exit comparison: 12:30 Rs {a['avg_day_pnl']:+.0f}/day, close Rs "
              f"{b['avg_day_pnl']:+.0f}/day", flush=True)
    return conn


def _export_history(conn, folder: Path, keep: int = 5000) -> None:
    """Append this run's replay and paper-check results to the published CSVs (dashboard)."""
    import pandas as pd

    folder.mkdir(parents=True, exist_ok=True)
    for table in ("replay_runs", "tune_checks"):
        new = pd.read_sql(f"SELECT * FROM {table}", conn).drop(columns="id")
        path = folder / f"{table}.csv"
        old = pd.read_csv(path) if path.exists() else new.iloc[:0]
        pd.concat([old, new], ignore_index=True).tail(keep).to_csv(path, index=False)


def cmd_schedule(settings, args) -> None:
    from stockpredictor import scheduler

    if args.action == "install":
        scheduler.install()
        times = " and ".join(f"{h:02d}:{m:02d}" for h, m in scheduler.RUN_TIMES)
        print(f"Daily training scheduled at {times} (log: {scheduler.LOG_PATH}).")
    elif args.action == "remove":
        scheduler.remove()
        print("Daily training schedule removed.")
    else:
        print("Daily training schedule:", "ON" if scheduler.is_installed() else "OFF")


WEB_PORT = 8501


def _web_server(port: int = WEB_PORT):
    """Start the web dashboard (a local website) in the background."""
    import subprocess

    page = Path(__file__).parent / "app" / "main.py"
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(page),
                             "--server.headless", "true", "--server.port", str(port),
                             "--browser.gatherUsageStats", "false",
                             "--client.toolbarMode", "minimal"])


def cmd_app(settings) -> None:
    """Web dashboard only."""
    import webbrowser

    proc = _web_server()
    url = f"http://localhost:{WEB_PORT}"
    print(f"Web dashboard: {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("Stopped.")


BACKFILL_ATTEMPTS = 8        # Angel One history: tries, 30 minutes apart


def _angel_startup(settings, store_dir: Path) -> None:
    """Check the Angel One login; download intraday history in the background if missing."""
    import threading

    from stockpredictor.data import intraday

    if not settings.angel.is_complete:
        print("  Angel One: keys not set in .env - using Yahoo (prices may lag a few minutes)\n")
        return
    try:
        from stockpredictor.data import angelone

        client = angelone.AngelDataClient(settings.angel)
        client.login()
        print("  Angel One: login OK - real-time prices enabled")
    except Exception as exc:
        print(f"  Angel One: login FAILED ({exc}) - using Yahoo. Check the keys in .env.\n")
        return
    from stockpredictor import store

    uni = store.load_universe(store_dir)
    active = uni.loc[uni["active"] == 1, "symbol"].tolist()
    have = intraday.backfill_days()
    # Full download if there is little history; otherwise only stocks new to the universe.
    # (and again for stocks whose history lacks the 12:30 exit prices; resumes if interrupted)
    todo = active if have < 200 else sorted(
        (set(active) - intraday.backfill_symbols())
        | (set(active) & (intraday.backfill_missing_exit() | intraday.backfill_incomplete())))
    if not todo:
        print(f"  Angel One intraday history: {have} days available\n")
        return
    print(f"  Angel One intraday history: downloading ~2 years for {len(todo)} stocks in the "
          f"background (~{max(1, len(todo) * 15 // 200)} min); the intraday model retrains when it "
          "finishes.\n")

    def job():
        import logging
        import time as _time

        from stockpredictor.models import intraday as MI
        from stockpredictor.models import trainer as T
        from stockpredictor.paper.engine import MarketContext

        log = logging.getLogger("backfill")
        remaining, total = list(todo), 0
        for attempt in range(1, BACKFILL_ATTEMPTS + 1):
            try:
                tokens = angelone.fetch_nse_equity_tokens()
                failed = []
                total += intraday.angel_backfill(
                    client, tokens, remaining, date.today() - timedelta(days=730),
                    date.today() - timedelta(days=1),
                    progress=lambda m: failed.append(m) if "error" in m else log.debug(m))
            except Exception:
                log.exception("Angel One backfill failed")
            missing = (set(active) - intraday.backfill_symbols()) | (set(active) & (
                intraday.backfill_missing_exit() | intraday.backfill_incomplete()))
            remaining = [s_ for s_ in remaining if s_ in missing]
            if not remaining:
                break
            log.warning("Angel One history: %d stocks still missing (Angel One throttling); "
                        "retrying in 30 minutes (attempt %d of %d)", len(remaining), attempt,
                        BACKFILL_ATTEMPTS)
            _time.sleep(30 * 60)
        if not total:
            log.info("Angel One backfill: nothing new downloaded (will retry next start)")
            return
        log.info("Angel One backfill: %s stock-days saved", total)
        if not config.MAC_TRAINING:
            return
        try:
            with db.connect(settings.db_path) as conn:
                ctx = MarketContext.load(store_dir)
                for target in MI.TARGETS:
                    with MI.MODEL_LOCK:
                        T.retrain_intraday(ctx, store_dir, conn, force=True, target=target)
            log.info("Intraday models retrained with Angel One history")
        except Exception:
            log.exception("retraining after the Angel One backfill failed")

    threading.Thread(target=job, daemon=True, name="angel-backfill").start()


def cmd_start(settings, args) -> None:
    """Everything in one go: web dashboard + live monitor (+ continuous training)."""
    import logging
    import time
    import webbrowser

    from stockpredictor.live import background, monitor, prices

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db.init_db(settings.db_path)
    proc = _web_server()
    url = f"http://localhost:{WEB_PORT}"
    time.sleep(3)
    print(f"\n  Web dashboard:  {url}")
    print("  Open it in your browser, or in VS Code: Cmd+Shift+P -> 'Simple Browser: Show'")
    print("  Live monitor and continuous training are running. Ctrl+C stops everything.\n")
    if not args.no_browser:
        webbrowser.open(url)
    _angel_startup(settings, Path(args.dir))
    conn = db.connect(settings.db_path)
    mon = monitor.Monitor(conn, Path(args.dir), prices.LivePrices(settings),
                          settings.paper_capital_longterm,
                          capital_intraday=settings.paper_capital_intraday)
    until = None
    if args.until:
        from datetime import time as _t
        until = _t(*map(int, args.until.split(":")))
    if config.MAC_TRAINING:
        mon.background = background.BackgroundTrainer(settings.db_path, Path(args.dir), settings)
        mon.background.start()
    try:
        monitor.run_forever(mon, until=until)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        proc.terminate()


def _intraday_features(d: Path):
    from stockpredictor import store
    from stockpredictor.data import intraday
    from stockpredictor.features import intraday as FI
    from stockpredictor.paper.engine import MarketContext

    ctx = MarketContext.load(d)
    return FI.build_for(intraday.load_summaries(d), ctx, d)


def cmd_train_intraday(settings, args) -> None:
    from stockpredictor.models import intraday as MI

    feats = _intraday_features(Path(args.dir))
    days = feats["date"].nunique()
    if days < MI.MIN_TRAIN_DAYS:
        print(f"Only {days} days of intraday data; need {MI.MIN_TRAIN_DAYS}. "
              "Run intraday-backfill (Angel One) or wait for daily collection.")
        return
    model = MI.IntradayModel.train(feats)
    model.save()
    print(f"Intraday model trained on {model.train_days} days up to {model.train_to}")
    print("Top features:", ", ".join(model.importance().head(8).index))


def cmd_backtest_intraday(settings, args) -> None:
    from stockpredictor.backtest import intraday as B
    from stockpredictor.models import intraday as MI

    feats = _intraday_features(Path(args.dir))
    scores = MI.walk_forward(feats)
    if scores.empty:
        print(f"Not enough intraday history ({feats['date'].nunique()} days).")
        return
    report = B.run(feats, scores, capital=settings.paper_capital_intraday)
    print(B.format_report(report))
    B.save(report, MI.MODEL_DIR / "backtest.json")
    print(f"\nSaved report to {MI.MODEL_DIR / 'backtest.json'}")


def cmd_status(settings) -> None:
    print(f"Database:        {settings.db_path} ({'exists' if settings.db_path.exists() else 'missing'})")
    print(f"Angel One keys:  {'set' if settings.angel.is_complete else 'not set'}")
    print(f"Paper capital:   intraday ₹{settings.paper_capital_intraday:,.0f}, "
          f"long-term ₹{settings.paper_capital_longterm:,.0f}")
    if settings.db_path.exists():
        db.init_db(settings.db_path)  # applies schema updates to older databases
        with db.connect(settings.db_path) as conn:
            n, t = conn.execute("SELECT COUNT(*), SUM(tradable) FROM stocks WHERE active = 1").fetchone()
            print(f"Stocks:          {n} for training, {t or 0} tradable")
            for table in ("daily_prices", "index_prices"):
                cnt, last = conn.execute(f"SELECT COUNT(*), MAX(date) FROM {table}").fetchone()
                print(f"{table + ':':<17}{cnt:,} rows, latest {last or '-'}")


# Commands that take extra arguments: name -> (handler, help, argument setup)
def _symbols_arg(p):
    p.add_argument("--symbols", nargs="+", help="Only these NSE symbols (default: all Nifty 250)")


def _prices_args(p):
    _symbols_arg(p)
    p.add_argument("--start", default="2005-01-01", help="First date for new stocks (YYYY-MM-DD)")
    p.add_argument("--extend-back", action="store_true",
                   help="Also fetch history older than what is stored, back to --start")


def _collect_args(p):
    _dir_arg(p)
    p.add_argument("--period", default="7d", help="Yahoo lookback, up to 60d (default 7d)")


def _backfill_args(p):
    _dir_arg(p)
    _symbols_arg(p)
    p.add_argument("--days", type=int, default=730, help="How many days back (default 730)")


def _dir_arg(p):
    from stockpredictor.store import DEFAULT_STORE_DIR
    p.add_argument("--dir", default=str(DEFAULT_STORE_DIR), help="Market data folder")


def _features_args(p):
    _dir_arg(p)
    p.add_argument("--symbol", help="Show the latest feature values for one stock")


def _news_args(p):
    _dir_arg(p)
    p.add_argument("--no-score", action="store_true", help="Skip FinBERT scoring")


def _backtest_args(p):
    _dir_arg(p)
    p.add_argument("--start-year", type=int, default=2015, help="First out-of-sample year")


def _daily_args(p):
    _dir_arg(p)
    p.add_argument("--no-sync", action="store_true", help="Use the local data snapshot as is")


def _improve_args(p):
    _dir_arg(p)
    p.add_argument("--tune", action="store_true", help="Also try new model settings")
    p.add_argument("--candidates", type=int, default=5, help="Settings to try per model")
    p.add_argument("--no-sync", action="store_true")


def _cloud_train_args(p):
    _dir_arg(p)
    p.add_argument("--minutes", type=float, default=40, help="Total time budget incl. tuning")
    p.add_argument("--candidates", type=int, default=3, help="New settings per tuning round")
    p.add_argument("--feedback", help="Folder with longterm_judged.csv (paper-feedback branch)")
    p.add_argument("--backtest", action="store_true", help="Refresh the backtest now")


def _schedule_args(p):
    p.add_argument("action", choices=["install", "remove", "status"], nargs="?", default="status")


def _start_args(p):
    _dir_arg(p)
    p.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    p.add_argument("--until", help="Stop after this time (HH:MM, IST) once the day's "
                                   "after-close work is done, e.g. 21:30")


def cmd_today(settings, args) -> None:
    """One-shot daily run for when the monitor can't stay on: sync, learn, decide, show picks."""
    import logging

    from stockpredictor.live import monitor, prices
    from stockpredictor.live.prices import in_market_hours, now_ist
    from stockpredictor.paper import engine as E

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    db.init_db(settings.db_path)
    conn = db.connect(settings.db_path)
    mon = monitor.Monitor(conn, Path(args.dir), prices.LivePrices(settings),
                          settings.paper_capital_longterm,
                          capital_intraday=settings.paper_capital_intraday)
    now = now_ist()
    print("1/3 Getting the latest data...")
    mon.sync(Path(args.dir))
    ctx = mon.ctx(reload=True)
    if in_market_hours(now) and monitor.INTRADAY_PICKS <= now.time() <= monitor.INTRADAY_LATEST:
        print("    Market is open: making today's intraday picks...")
        mon.intraday_picks_job(now)
    print("2/3 Learning from the newest data and paper results, then deciding...")
    from stockpredictor.models import trainer as T
    from stockpredictor.paper import daily as D
    T.retrain_longterm(ctx, conn)
    T.retrain_intraday(ctx, Path(args.dir), conn)
    mon._model = None
    D.run_daily(conn, ctx, settings.paper_capital_longterm, mon.model())
    print("3/3 Today's long-term picks\n")
    last = conn.execute("SELECT MAX(date) FROM predictions WHERE horizon = 'longterm'").fetchone()[0]
    for direction, title in (("up", "▲ 10 BUY candidates"),):
        rows = conn.execute("SELECT symbol, confidence, entry_price FROM predictions WHERE horizon = "
                            "'longterm' AND date = ? AND direction = ? ORDER BY confidence DESC",
                            (last, direction)).fetchall()
        print(f"  {title} ({last})")
        for i, r in enumerate(rows, 1):
            print(f"   {i:2}. {r['symbol']:<12} confidence {r['confidence']:.0%}   Rs {r['entry_price']:,.2f}")
        print()
    for h, name in ((E.HORIZON, "Buy book"),):
        v = E.value(conn, ctx.closes_on(ctx.daily["date"].max()), h)
        print(f"  Paper {name}: Rs {v['equity']:,.0f} (P&L Rs {v['pnl']:+,.0f}), {v['positions']} positions")
    print("\nOpen the dashboard for details: python -m stockpredictor app")


def cmd_autostart(settings, args) -> None:
    """Start automatically at 09:00 on weekdays (macOS launchd); stops by itself at night."""
    import plistlib
    import subprocess

    from stockpredictor.config import PROJECT_ROOT

    label = "com.stockpredictor.daily"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    if args.off:
        subprocess.run(["launchctl", "unload", str(plist)], check=False)
        plist.unlink(missing_ok=True)
        print("Autostart turned off.")
        return
    logs = PROJECT_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    python = PROJECT_ROOT / ".venv" / "bin" / "python"
    spec = {
        "Label": label,
        "ProgramArguments": ["/usr/bin/caffeinate", "-i", str(python), "-m", "stockpredictor",
                             "start", "--no-browser", "--until", "21:30"],
        "WorkingDirectory": str(PROJECT_ROOT),
        "StartCalendarInterval": [{"Weekday": d, "Hour": 9, "Minute": 0} for d in range(1, 6)],
        "StandardOutPath": str(logs / "daily.log"),
        "StandardErrorPath": str(logs / "daily.log"),
    }
    plist.parent.mkdir(parents=True, exist_ok=True)
    with open(plist, "wb") as f:
        plistlib.dump(spec, f)
    subprocess.run(["launchctl", "unload", str(plist)], check=False, capture_output=True)
    subprocess.run(["launchctl", "load", str(plist)], check=True)
    print("Autostart on: Mon-Fri at 09:00 (or when the Mac wakes, if it was asleep).")
    print("It runs in the background and stops after 21:30 once the day's work is done.")
    print(f"Log: {logs / 'daily.log'}  ·  Dashboard: python -m stockpredictor app")
    print("Turn off with: python -m stockpredictor autostart --off")


def _autostart_args(p):
    p.add_argument("--off", action="store_true", help="Turn autostart off")


ARG_COMMANDS = {
    "today": (cmd_today, "One-shot daily run: sync, learn, decide, print picks", _dir_arg),
    "autostart": (cmd_autostart, "Start automatically at 09:00 on weekdays (macOS)",
                  _autostart_args),
    "start": (cmd_start, "Start everything: web dashboard + live monitor + training", _start_args),
    "improve": (cmd_improve, "Sync data, retrain both models, optionally self-tune",
                _improve_args),
    "auto": (cmd_auto, "One background pass: after-close decision, retrain, weekend tuning",
             _dir_arg),
    "cloud-train": (cmd_cloud_train, "GitHub Actions: train on all history and self-tune",
                    _cloud_train_args),
    "schedule": (cmd_schedule, "Install/remove the daily background training job (macOS)",
                 _schedule_args),
    "train-intraday": (cmd_train_intraday, "Train the intraday model", _dir_arg),
    "backtest-intraday": (cmd_backtest_intraday, "Walk-forward backtest of the intraday model",
                          _dir_arg),
    "run": (cmd_run, "Start the live monitor (keep running during market hours)", _dir_arg),
    "daily": (cmd_daily, "After-close job: sync, decide, paper-trade, evaluate", _daily_args),
    "train": (cmd_train, "Train the long-term model on all data", _dir_arg),
    "backtest": (cmd_backtest, "Walk-forward backtest of the long-term model", _backtest_args),
    "news-update": (cmd_news_update, "Fetch company news and score sentiment", _news_args),
    "fundamentals-update": (cmd_fundamentals_update, "Save a fundamentals snapshot", _dir_arg),
    "features": (cmd_features, "Build long-term features and show a summary", _features_args),
    "sync-data": (cmd_sync_data, "Fetch the latest market data from git (for training)",
                  _dir_arg),
    "export-store": (cmd_export_store, "Write database market data to CSV files", _dir_arg),
    "import-store": (cmd_import_store, "Load CSV market data into the database", _dir_arg),
    "prices": (cmd_prices, "Download/update daily prices for stocks and indices", _prices_args),
    "intraday-collect": (cmd_intraday_collect, "Yahoo 5-min bars -> intraday summaries (git)",
                         _collect_args),
    "intraday-backfill": (cmd_intraday_backfill, "Angel One 5-min history -> local summaries",
                          _backfill_args),
    "data-check": (cmd_data_check, "Report data coverage and gaps", _symbols_arg),
    "delivery-update": (cmd_delivery_update, "NSE delivery share per stock (git store)",
                        _delivery_args),
    "earnings-update": (cmd_earnings_update, "Quarterly results dates (git store)", _dir_arg),
    "preopen-update": (cmd_preopen_update, "Today's NSE pre-open auction (git store)", _dir_arg),
}

COMMANDS = {
    "init": (cmd_init, "Create the local SQLite database"),
    "universe": (cmd_universe, "Download the Nifty LargeMidcap 250 constituents from NSE"),
    "tokens": (cmd_tokens, "Map stocks to Angel One instrument tokens"),
    "check-angel": (cmd_check_angel, "Test Angel One login and fetch one live price"),
    "status": (cmd_status, "Show configuration and database status"),
    "app": (cmd_app, "Open the web dashboard only"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stockpredictor")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    for name, (_, help_text, setup) in ARG_COMMANDS.items():
        setup(sub.add_parser(name, help=help_text))
    args = parser.parse_args(argv)
    if args.command in ARG_COMMANDS:
        ARG_COMMANDS[args.command][0](load_settings(), args)
    else:
        COMMANDS[args.command][0](load_settings())
    return 0


if __name__ == "__main__":
    sys.exit(main())
