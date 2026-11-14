"""Paper state between the Mac and GitHub.

The Mac paper-trades during market hours (a day with the Mac off has no trades). After the
close it uploads its paper state to the `paper-state` branch; GitHub's evening run
(train-models.yml, `cloud-evening`) makes the long-term decisions, judges the picks, sends
the paper feedback into that night's training and publishes the state back; the next
morning the Mac takes that state over before trading.

The real portfolio you enter yourself (portfolio_trades) never leaves the Mac (public repo).
"""

from __future__ import annotations

import gzip
import logging
import shutil
import sqlite3
import tempfile
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

from stockpredictor import store

log = logging.getLogger(__name__)
from stockpredictor.config import MARKET as _MARKET  # noqa: E402

STATE_BRANCH = _MARKET.state_branch       # India "paper-state", US "paper-state-us"
STATE_FILE = "state.db"
PRIVATE_TABLES = ("portfolio_trades",)
SAVED_KEY = "state_saved_at"


def _saved_at(conn: sqlite3.Connection, schema: str = "main") -> str:
    try:
        row = conn.execute(f"SELECT value FROM {schema}.app_settings WHERE key = ?",
                           (SAVED_KEY,)).fetchone()
    except sqlite3.OperationalError:
        return ""
    return row[0] if row else ""


def fetch_state(dest: Path) -> bool:
    """Download the published state into `dest` (False: none published yet)."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            if not store._branch_checkout(Path(tmp) / "s", STATE_BRANCH):
                return False
        except Exception as exc:
            log.warning("no paper state on GitHub yet (%s)", exc)
            return False
        src = Path(tmp) / "s" / STATE_FILE
        parts = sorted((Path(tmp) / "s").glob(STATE_FILE + ".gz.part*"))
        if not src.exists() and parts:            # uploaded in small pieces (upload_pieces)
            src.write_bytes(gzip.decompress(b"".join(p.read_bytes() for p in parts)))
        if not src.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        return True


def snapshot(db: Path, out: Path) -> None:
    """Consistent copy of the database without the private tables, stamped with the time."""
    src = sqlite3.connect(db)
    dst = sqlite3.connect(out)
    with dst:
        src.backup(dst)
    src.close()
    for t in PRIVATE_TABLES:
        try:
            dst.execute(f"DELETE FROM {t}")
        except sqlite3.OperationalError:
            pass
    # Price history lives on the data branch; the state keeps the last 15 days (latest prices
    # for the dashboard) - on GitHub the database holds 20 years of every stock.
    for t in ("daily_prices", "index_prices"):
        try:
            dst.execute(f"DELETE FROM {t} WHERE date < date((SELECT MAX(date) FROM {t}), '-15 days')")
        except sqlite3.OperationalError:
            pass
    dst.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                (SAVED_KEY, datetime.now().astimezone().isoformat(timespec="seconds")))
    dst.commit()
    dst.execute("VACUUM")
    dst.close()


def publish_state(db: Path) -> None:
    """From GitHub (good network): the state as one file."""
    with tempfile.TemporaryDirectory() as tmp:
        snapshot(db, Path(tmp) / STATE_FILE)
        store.publish_branch(Path(tmp), STATE_BRANCH,
                             f"Paper state {datetime.now():%Y-%m-%d %H:%M}")


def upload_pieces(db: Path, piece_kb: int = 15) -> int:
    """From the Mac, whose network drops uploads above ~16 KB: the compressed state in small
    pieces, one push each."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        snapshot(db, work / "state.db")
        data = gzip.compress((work / "state.db").read_bytes(), 9)
        repo = work / "repo"
        repo.mkdir()
        store._git("init", "-q", "-b", STATE_BRANCH, cwd=repo)
        size, n = piece_kb * 1024, 0
        for i in range(0, len(data), size):
            (repo / f"{STATE_FILE}.gz.part{n:04d}").write_bytes(data[i:i + size])
            store._git("add", "-A", cwd=repo)
            store._git("-c", "user.name=stockpredictor", "-c", "user.email=s@localhost",
                       "commit", "-q", "-m", f"state piece {n}", cwd=repo)
            for _ in range(6):
                try:
                    store._git("push", "-q", "--force", store.default_remote(),
                               f"{STATE_BRANCH}:refs/heads/{STATE_BRANCH}", cwd=repo)
                    break
                except Exception:
                    _time.sleep(5)
            else:
                raise RuntimeError(f"could not upload piece {n}")
            n += 1
        return n


def pull_state(conn: sqlite3.Connection, today: str) -> bool:
    """Morning on the Mac: take over GitHub's newer state (its evening work), keeping this
    Mac's private tables. Never once today's trading has started here."""
    if conn.execute("SELECT 1 FROM predictions WHERE date = ? AND horizon LIKE 'intraday%' "
                    "LIMIT 1", (today,)).fetchone():
        return False
    with tempfile.TemporaryDirectory() as tmp:
        remote = Path(tmp) / "remote.db"
        if not fetch_state(remote):
            return False
        from stockpredictor import db

        db.init_db(remote)                       # same tables as this code version
        conn.execute("ATTACH DATABASE ? AS r", (str(remote),))
        try:
            if _saved_at(conn, "r") <= _saved_at(conn):
                return False
            tables = [t for (t,) in conn.execute(
                "SELECT name FROM r.sqlite_master WHERE type = 'table' AND name NOT LIKE "
                "'sqlite_%'") if t not in PRIVATE_TABLES]
            with conn:
                for t in tables:
                    cols = [c[1] for c in conn.execute(f"PRAGMA main.table_info({t})")]
                    rcols = {c[1] for c in conn.execute(f"PRAGMA r.table_info({t})")}
                    keep = ", ".join(c for c in cols if c in rcols)
                    if not keep:
                        continue
                    conn.execute(f"DELETE FROM main.{t}")
                    conn.execute(f"INSERT INTO main.{t} ({keep}) SELECT {keep} FROM r.{t}")
            log.info("Took over the paper state from GitHub (saved %s)", _saved_at(conn))
            return True
        finally:
            conn.execute("DETACH DATABASE r")


def _monitor(settings, store_dir: Path, clock=None):
    from stockpredictor import db
    from stockpredictor.live import monitor, prices

    db.init_db(settings.db_path)
    conn = db.connect(settings.db_path)
    return monitor.Monitor(conn, store_dir, prices.LivePrices(settings),
                           settings.paper_capital_longterm,
                           capital_intraday=settings.paper_capital_intraday,
                           sync=lambda d: store.sync(d, years=4),   # recent years only
                           **({"clock": clock} if clock else {}))


def evening(settings, store_dir: Path) -> None:
    """GitHub, after the evening data update: long-term decisions, judging and the paper
    feedback for tonight's training, then publish the state."""
    from stockpredictor.live.prices import now_ist

    fetch_state(Path(settings.db_path))
    now = now_ist()
    if now.hour < 9:          # GitHub's evening runs often start after midnight: that
        now = (now - timedelta(days=1)).replace(hour=23, minute=30)   # is still yesterday's
    mon = _monitor(settings, store_dir, clock=lambda: now)            # evening work
    try:
        if not (now.weekday() < 5 and mon.after_close_job(now)):   # (it sends the feedback)
            log.warning("after-close work not done today; sending the paper feedback anyway")
            mon.push_feedback()
    finally:
        publish_state(Path(settings.db_path))
