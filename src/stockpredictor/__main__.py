"""Command line entry point: python -m stockpredictor <command>."""

from __future__ import annotations

import argparse
import sys

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


def cmd_status(settings) -> None:
    print(f"Database:        {settings.db_path} ({'exists' if settings.db_path.exists() else 'missing'})")
    print(f"Angel One keys:  {'set' if settings.angel.is_complete else 'not set'}")
    print(f"Telegram:        {'set' if settings.telegram_bot_token else 'not set'}")
    print(f"Paper capital:   intraday ₹{settings.paper_capital_intraday:,.0f}, "
          f"long-term ₹{settings.paper_capital_longterm:,.0f}")
    if settings.db_path.exists():
        with db.connect(settings.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM stocks WHERE active = 1").fetchone()[0]
        print(f"Active stocks:   {n}")


COMMANDS = {
    "init": (cmd_init, "Create the local SQLite database"),
    "universe": (cmd_universe, "Download the Nifty 100 constituents from NSE"),
    "tokens": (cmd_tokens, "Map stocks to Angel One instrument tokens"),
    "check-angel": (cmd_check_angel, "Test Angel One login and fetch one live price"),
    "status": (cmd_status, "Show configuration and database status"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stockpredictor")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    args = parser.parse_args(argv)
    COMMANDS[args.command][0](load_settings())
    return 0


if __name__ == "__main__":
    sys.exit(main())
