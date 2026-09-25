"""Git-friendly market data store: plain CSV files, one per year.

Market data lives on the `market-data` git branch, updated daily by a
GitHub Actions job. The Mac only fetches a shallow copy for training, so
nothing large or long-lived is kept locally. Files are sorted and yearly
so daily updates produce small, appended diffs.

Layout:
    universe.csv
    corporate_actions.csv
    daily/<YEAR>.csv       stock candles
    indices/<YEAR>.csv     index candles
"""

from __future__ import annotations

import csv
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

from stockpredictor.config import PROJECT_ROOT

DATA_BRANCH = "market-data"
DEFAULT_STORE_DIR = PROJECT_ROOT / "market-data"

PRICE_COLS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]
UNIVERSE_COLS = ["symbol", "name", "industry", "isin", "active", "tradable"]
ACTION_COLS = ["symbol", "date", "kind", "value"]
PRICE_TABLES = {"daily": "daily_prices", "indices": "index_prices"}


def _write_csv(path: Path, cols: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(v) for v in r])


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return "" if v is None else v


def export_store(conn: sqlite3.Connection, store_dir: Path) -> None:
    """Write the database's market data to CSV files (full rewrite, deterministic order)."""
    _write_csv(store_dir / "universe.csv", UNIVERSE_COLS, conn.execute(
        f"SELECT {', '.join(UNIVERSE_COLS)} FROM stocks ORDER BY symbol"))
    _write_csv(store_dir / "corporate_actions.csv", ACTION_COLS, conn.execute(
        f"SELECT {', '.join(ACTION_COLS)} FROM corporate_actions ORDER BY symbol, date, kind"))

    for folder, table in PRICE_TABLES.items():
        years = [r[0] for r in conn.execute(
            f"SELECT DISTINCT substr(date, 1, 4) FROM {table} ORDER BY 1")]
        for old in (store_dir / folder).glob("*.csv"):
            if old.stem not in years:
                old.unlink()
        for year in years:
            _write_csv(store_dir / folder / f"{year}.csv", PRICE_COLS, conn.execute(
                f"SELECT {', '.join(PRICE_COLS)} FROM {table} "
                f"WHERE date LIKE ? ORDER BY symbol, date", (f"{year}-%",)))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in csv.DictReader(f)]


def import_store(conn: sqlite3.Connection, store_dir: Path) -> None:
    """Load CSV files into the database (used by the GitHub Actions updater)."""
    stocks = _read_csv(store_dir / "universe.csv")
    for r in stocks:
        if r.get("tradable") is None:      # files written before the Nifty 200 change
            r["tradable"] = r["active"]
    conn.executemany(
        """INSERT OR REPLACE INTO stocks (symbol, name, industry, isin, active, tradable)
           VALUES (:symbol, :name, :industry, :isin, :active, :tradable)""", stocks)
    conn.executemany(
        "INSERT OR REPLACE INTO corporate_actions VALUES (:symbol, :date, :kind, :value)",
        _read_csv(store_dir / "corporate_actions.csv"))
    for folder, table in PRICE_TABLES.items():
        for path in sorted((store_dir / folder).glob("*.csv")):
            conn.executemany(
                f"INSERT OR REPLACE INTO {table} ({', '.join(PRICE_COLS)}) "
                f"VALUES ({', '.join(':' + c for c in PRICE_COLS)})",
                _read_csv(path))
    conn.commit()


# --- Reading for training (pandas, no database needed) -----------------------

def local_overlay_path(folder: str) -> Path:
    """Prices the Mac fetched itself when the git data was late (merged on load)."""
    from stockpredictor.config import DATA_DIR

    return DATA_DIR / "local_prices" / f"{folder}.csv"


def _load_prices(store_dir: Path, folder: str) -> pd.DataFrame:
    files = sorted((store_dir / folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No data in {store_dir / folder}. Run `python -m stockpredictor sync-data` first.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    overlay = local_overlay_path(folder)
    if overlay.exists():                      # git data wins where both have a day
        df = pd.concat([df, pd.read_csv(overlay)], ignore_index=True) \
            .drop_duplicates(["symbol", "date"], keep="first")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def save_local_overlay(folder: str, rows: pd.DataFrame, keep_after: str | None = None) -> int:
    path = local_overlay_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=PRICE_COLS)
    df = pd.concat([rows[PRICE_COLS], old], ignore_index=True).drop_duplicates(
        ["symbol", "date"], keep="first")
    if keep_after:                            # drop days the git data now covers
        df = df[df["date"] > keep_after]
    df.sort_values(["symbol", "date"]).to_csv(path, index=False, float_format="%.4f")
    return len(rows)


def load_daily(store_dir: Path = DEFAULT_STORE_DIR) -> pd.DataFrame:
    return _load_prices(store_dir, "daily")


def load_indices(store_dir: Path = DEFAULT_STORE_DIR) -> pd.DataFrame:
    return _load_prices(store_dir, "indices")


def load_universe(store_dir: Path = DEFAULT_STORE_DIR) -> pd.DataFrame:
    uni = pd.read_csv(store_dir / "universe.csv")
    if "tradable" not in uni:
        uni["tradable"] = uni["active"]
    uni["tradable"] = uni["tradable"].fillna(uni["active"]).astype(int)
    return uni


def tradable(universe: pd.DataFrame) -> list[str]:
    """Stocks the app trades and shows (Nifty 100); training uses all active stocks."""
    flag = universe["tradable"] if "tradable" in universe else universe["active"]
    return sorted(universe.loc[(universe["active"] == 1) & (flag == 1), "symbol"])


def load_actions(store_dir: Path = DEFAULT_STORE_DIR) -> pd.DataFrame:
    path = store_dir / "corporate_actions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=ACTION_COLS)


# --- Syncing the data branch to the Mac --------------------------------------

def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def sync(store_dir: Path = DEFAULT_STORE_DIR, remote: str | None = None) -> None:
    """Fetch the latest market-data branch as a shallow (history-free) checkout."""
    if not (store_dir / ".git").exists():
        remote = remote or _git("remote", "get-url", "origin", cwd=PROJECT_ROOT)
        store_dir.mkdir(parents=True, exist_ok=True)
        _git("init", "-q", cwd=store_dir)
        _git("remote", "add", "origin", remote, cwd=store_dir)
    _git("fetch", "-q", "--depth", "1", "origin", DATA_BRANCH, cwd=store_dir)
    _git("reset", "-q", "--hard", "FETCH_HEAD", cwd=store_dir)
    # Drop objects from previous syncs so only the latest snapshot is kept.
    _git("reflog", "expire", "--expire=now", "--all", cwd=store_dir)
    _git("gc", "-q", "--prune=now", cwd=store_dir)
