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
    volume  INTEGER,
    source  TEXT,
    PRIMARY KEY (symbol, date)
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

-- Paper trades generated from predictions
CREATE TABLE IF NOT EXISTS paper_trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon        TEXT NOT NULL CHECK (horizon IN ('intraday', 'longterm')),
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
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
