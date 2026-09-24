"""Command line entry point: python -m stockpredictor <command>."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from stockpredictor import db, universe
from stockpredictor.config import load_settings


def cmd_init(settings) -> None:
    db.init_db(settings.db_path)
    print(f"Database ready: {settings.db_path}")


def cmd_universe(settings) -> None:
    stocks = universe.fetch_nifty100()
    with db.connect(settings.db_path) as conn:
        universe.save_universe(conn, stocks)
    print(f"Saved {len(stocks)} Nifty 100 stocks.")


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
        results = daily.update_all(conn, symbols, date.fromisoformat(args.start))
    failed = {k: v for k, v in results.items() if v.startswith("error")}
    print(f"Done. {len(results) - len(failed)} updated, {len(failed)} failed.")
    for name, err in failed.items():
        print(f"  {name}: {err}")


def cmd_prices_intraday(settings, args) -> None:
    from stockpredictor.data import angelone, intraday

    db.init_db(settings.db_path)
    client = angelone.AngelDataClient(settings.angel)
    start = date.today() - timedelta(days=args.days)
    with db.connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT symbol, angel_token FROM stocks WHERE active = 1 "
            "AND angel_token IS NOT NULL ORDER BY symbol").fetchall()
        if args.symbols:
            rows = [r for r in rows if r["symbol"] in args.symbols]
        if not rows:
            print("No stocks with Angel One tokens — run `tokens` first.")
            return
        for i, r in enumerate(rows, 1):
            try:
                n = intraday.update_symbol(conn, client, r["symbol"], r["angel_token"],
                                           args.interval, start)
                print(f"[{i}/{len(rows)}] {r['symbol']}: {n} candles", flush=True)
            except Exception as exc:
                print(f"[{i}/{len(rows)}] {r['symbol']}: error: {exc}", flush=True)


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
    rows, errors = news.fetch_all(uni[uni["active"] == 1])
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
    rows, errors = fundamentals.fetch_snapshot(uni.loc[uni["active"] == 1, "symbol"].tolist())
    fundamentals.append_snapshot(d, rows)
    print(f"Fundamentals snapshot: {len(rows)} stocks, {len(errors)} errors")


def cmd_train(settings, args) -> None:
    from stockpredictor.backtest import run as bt
    from stockpredictor.models import longterm as M

    _, _, _, labeled = bt.load_all(Path(args.dir))
    model = M.LongTermModel.train(labeled)
    model.save(M.MODEL_DIR)
    print(f"Trained on data up to {model.train_to}; saved to {M.MODEL_DIR}")
    print("Top features:", ", ".join(model.importance().head(8).index))


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

    from stockpredictor.live import monitor, prices

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db.init_db(settings.db_path)
    conn = db.connect(settings.db_path)
    mon = monitor.Monitor(conn, Path(args.dir), prices.LivePrices(settings),
                          settings.paper_capital_longterm)
    try:
        monitor.run_forever(mon)
    except KeyboardInterrupt:
        print("Stopped.")


def cmd_app(settings) -> None:
    import subprocess

    app = Path(__file__).parent / "app" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
                    "--server.headless", "false", "--browser.gatherUsageStats", "false",
                    "--client.toolbarMode", "minimal"])


def cmd_status(settings) -> None:
    print(f"Database:        {settings.db_path} ({'exists' if settings.db_path.exists() else 'missing'})")
    print(f"Angel One keys:  {'set' if settings.angel.is_complete else 'not set'}")
    print(f"Telegram:        {'set' if settings.telegram_bot_token else 'not set'}")
    print(f"Paper capital:   intraday ₹{settings.paper_capital_intraday:,.0f}, "
          f"long-term ₹{settings.paper_capital_longterm:,.0f}")
    if settings.db_path.exists():
        db.init_db(settings.db_path)  # applies schema updates to older databases
        with db.connect(settings.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM stocks WHERE active = 1").fetchone()[0]
            print(f"Active stocks:   {n}")
            for table in ("daily_prices", "index_prices", "intraday_prices"):
                cnt, last = conn.execute(
                    f"SELECT COUNT(*), MAX({'ts' if table == 'intraday_prices' else 'date'}) "
                    f"FROM {table}").fetchone()
                print(f"{table + ':':<17}{cnt:,} rows, latest {last or '-'}")


# Commands that take extra arguments: name -> (handler, help, argument setup)
def _symbols_arg(p):
    p.add_argument("--symbols", nargs="+", help="Only these NSE symbols (default: all Nifty 100)")


def _prices_args(p):
    _symbols_arg(p)
    p.add_argument("--start", default="2010-01-01", help="First date for new stocks (YYYY-MM-DD)")


def _intraday_args(p):
    _symbols_arg(p)
    p.add_argument("--interval", default="FIVE_MINUTE",
                   help="ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ... (default FIVE_MINUTE)")
    p.add_argument("--days", type=int, default=365, help="How many days back (default 365)")


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


ARG_COMMANDS = {
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
    "prices-intraday": (cmd_prices_intraday, "Download intraday candles from Angel One",
                        _intraday_args),
    "data-check": (cmd_data_check, "Report data coverage and gaps", _symbols_arg),
}

COMMANDS = {
    "init": (cmd_init, "Create the local SQLite database"),
    "universe": (cmd_universe, "Download the Nifty 100 constituents from NSE"),
    "tokens": (cmd_tokens, "Map stocks to Angel One instrument tokens"),
    "check-angel": (cmd_check_angel, "Test Angel One login and fetch one live price"),
    "status": (cmd_status, "Show configuration and database status"),
    "app": (cmd_app, "Open the dashboard in your browser"),
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
