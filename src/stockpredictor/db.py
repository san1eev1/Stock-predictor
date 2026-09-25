"""SQLite database: schema and connection helper."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
-- Stock universe (Nifty 100)
CREATE TABLE IF NOT EXISTS stocks (
    symbol        TEXT PRIMARY KEY,          -- NSE symbol, e.g. RELIANCE
    name          TEXT,
    industry      TEXT,
    isin          TEXT,
    angel_token   TEXT,                      -- Angel One instrument token
    active        INTEGER NOT NULL DEFAULT 1,
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Daily OHLCV candles
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,                   -- YYYY-MM-DD
    open    REAL, high REAL, low REAL, close REAL,
    adj_close REAL,                          -- also dividend-adjusted
    volume  INTEGER,
    source  TEXT,
    PRIMARY KEY (symbol, date)
);

-- Daily candles for market context indices (NIFTY50, INDIAVIX, sectors)
CREATE TABLE IF NOT EXISTS index_prices (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL, high REAL, low REAL, close REAL,
    adj_close REAL,
    volume  INTEGER,
    source  TEXT,
    PRIMARY KEY (symbol, date)
);

-- Splits / bonuses (value = share multiplier, e.g. 2.0 for 1:1 bonus) and dividends
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    kind    TEXT NOT NULL CHECK (kind IN ('split', 'dividend')),
    value   REAL NOT NULL,
    PRIMARY KEY (symbol, date, kind)
);

-- Intraday OHLCV candles (1-min / 5-min ...)
CREATE TABLE IF NOT EXISTS intraday_prices (
    symbol    TEXT NOT NULL,
    ts        TEXT NOT NULL,                 -- ISO timestamp, IST
    interval  TEXT NOT NULL,                 -- e.g. ONE_MINUTE
    open      REAL, high REAL, low REAL, close REAL,
    volume    INTEGER,
    PRIMARY KEY (symbol, interval, ts)
);

-- Company news with sentiment
CREATE TABLE IF NOT EXISTS news (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,
    published_at  TEXT NOT NULL,
    source        TEXT,
    title         TEXT NOT NULL,
    url           TEXT UNIQUE,
    sentiment     REAL,                      -- -1 (negative) .. +1 (positive)
    fetched_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Model predictions (intraday and long-term)
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon         TEXT NOT NULL CHECK (horizon IN ('intraday', 'longterm')),
    date            TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('up', 'down')),
    confidence      REAL,
    rank            INTEGER,
    entry_price     REAL,
    stop_loss       REAL,
    target          REAL,
    reasons         TEXT,                    -- JSON list of top signals
    model_version   TEXT,
    actual_exit     REAL,
    actual_return   REAL,
    correct         INTEGER,                 -- 1 / 0, filled after evaluation
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (horizon, date, symbol)
);

-- Paper trades generated from predictions (horizon: longterm, longterm_short, intraday)
CREATE TABLE IF NOT EXISTS paper_trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon        TEXT NOT NULL,
    prediction_id  INTEGER REFERENCES predictions(id),
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL CHECK (side IN ('long', 'short')),
    qty            INTEGER NOT NULL,
    entry_time     TEXT NOT NULL,
    entry_price    REAL NOT NULL,
    exit_time      TEXT,
    exit_price     REAL,
    costs          REAL,
    pnl            REAL,
    status         TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed'))
);

-- Real trades entered manually (from Groww)
CREATE TABLE IF NOT EXISTS portfolio_trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon      TEXT NOT NULL CHECK (horizon IN ('intraday', 'longterm')),
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    qty          INTEGER NOT NULL,
    price        REAL NOT NULL,
    charges      REAL NOT NULL DEFAULT 0,
    trade_date   TEXT NOT NULL,
    notes        TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Model training runs
CREATE TABLE IF NOT EXISTS model_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon     TEXT NOT NULL,
    version     TEXT NOT NULL,
    trained_at  TEXT NOT NULL DEFAULT (datetime('now')),
    train_from  TEXT,
    train_to    TEXT,
    metrics     TEXT                         -- JSON
);

-- Paper trading accounts (one per horizon)
CREATE TABLE IF NOT EXISTS paper_accounts (
    horizon     TEXT PRIMARY KEY,
    capital     REAL NOT NULL,
    cash        REAL NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Orders decided by the model, filled at the next available price
CREATE TABLE IF NOT EXISTS paper_orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon     TEXT NOT NULL,
    created     TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    reason      TEXT,
    status      TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'cancelled')),
    fill_time   TEXT,
    fill_price  REAL
);

-- Daily paper portfolio value
CREATE TABLE IF NOT EXISTS paper_equity (
    horizon   TEXT NOT NULL,
    date      TEXT NOT NULL,
    cash      REAL NOT NULL,
    holdings  REAL NOT NULL,
    equity    REAL NOT NULL,
    PRIMARY KEY (horizon, date)
);

-- Alerts from the live monitor (stop-loss, news, ...)
CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL DEFAULT (datetime('now')),
    source    TEXT NOT NULL,             -- paper-longterm / portfolio / ...
    kind      TEXT NOT NULL,             -- stop-loss / target / negative-news / info
    symbol    TEXT,
    message   TEXT NOT NULL,
    sent      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (source, kind, symbol, message)
);

-- Shadow predictions of strategy variants, judged like real ones (live strategy race)
CREATE TABLE IF NOT EXISTS shadow_predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    variant       TEXT NOT NULL,
    date          TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    direction     TEXT NOT NULL,
    entry_price   REAL,
    nifty_entry   REAL,
    actual_exit   REAL,
    actual_return REAL,
    correct       INTEGER,
    base_rate     REAL,
    evaluated_at  TEXT,
    UNIQUE (variant, date, symbol)
);

-- Latest long-term trading score and rank per stock (for "sell now" signals)
CREATE TABLE IF NOT EXISTS lt_scores (
    date    TEXT NOT NULL,
    symbol  TEXT PRIMARY KEY,
    score   REAL,
    rank    INTEGER
);

-- Latest live prices seen by the monitor
CREATE TABLE IF NOT EXISTS live_prices (
    symbol  TEXT PRIMARY KEY,
    price   REAL NOT NULL,
    ts      TEXT NOT NULL,
    source  TEXT
);

-- Provisional intraday re-scoring of the long-term model
CREATE TABLE IF NOT EXISTS live_scores (
    symbol         TEXT PRIMARY KEY,
    ts             TEXT NOT NULL,
    score          REAL,
    rank           INTEGER,
    official_rank  INTEGER,
    price          REAL
);

-- 9:45 prices of every stock (random-pick baseline for intraday accuracy)
CREATE TABLE IF NOT EXISTS intraday_open (
    date    TEXT NOT NULL,
    symbol  TEXT NOT NULL,
    c30     REAL NOT NULL,
    PRIMARY KEY (date, symbol)
);

-- App settings editable from the UI
CREATE TABLE IF NOT EXISTS app_settings (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news(symbol, published_at);
CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(horizon, date);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # The dashboard refreshes parts of pages from other threads; each part opens its own
    # connection, and check_same_thread=False keeps a shared one from crashing the page.
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# Columns added after a table was first released: (table, column, type).
MIGRATIONS = [
    ("daily_prices", "adj_close", "REAL"),
    ("stocks", "tradable", "INTEGER NOT NULL DEFAULT 1"),
    ("predictions", "nifty_entry", "REAL"),
    ("predictions", "base_rate", "REAL"),
    ("predictions", "evaluated_at", "TEXT"),
    ("predictions", "horizon_days", "INTEGER"),
    ("shadow_predictions", "horizon_days", "INTEGER"),
    ("paper_trades", "reason", "TEXT"),
    ("paper_trades", "exit_reason", "TEXT"),
    ("paper_trades", "stop_loss", "REAL"),
    ("paper_trades", "target", "REAL"),
]


def _relax_paper_trades(conn: sqlite3.Connection) -> None:
    """Older databases only allowed two paper books; rebuild the table without that limit."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'paper_trades'").fetchone()
    if row is None or "CHECK (horizon IN" not in row[0]:
        return
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(paper_trades)")]
    conn.execute("ALTER TABLE paper_trades RENAME TO paper_trades_old")
    create = SCHEMA[SCHEMA.index("CREATE TABLE IF NOT EXISTS paper_trades"):]
    conn.execute(create[:create.index(");") + 2])
    for col, typ in [("reason", "TEXT"), ("exit_reason", "TEXT"), ("stop_loss", "REAL"),
                     ("target", "REAL")]:
        if col not in {r["name"] for r in conn.execute("PRAGMA table_info(paper_trades)")}:
            conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {typ}")
    names = ", ".join(cols)
    conn.execute(f"INSERT INTO paper_trades ({names}) SELECT {names} FROM paper_trades_old")
    conn.execute("DROP TABLE paper_trades_old")


def _migrate(conn: sqlite3.Connection) -> None:
    _relax_paper_trades(conn)
    for table, column, col_type in MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
