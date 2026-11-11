"""Paper trading without the Mac: GitHub Actions runs the live monitor through market hours
(paper-trading.yml) and the after-close work in the evening (train-models.yml). The paper
state (SQLite: trades, picks, accounts, alerts, settings) lives on the `paper-state` branch;
every run starts from it and publishes it back every few minutes. The Mac (or any viewer)
only downloads it.

The real portfolio you enter yourself (portfolio_trades) is never published (public repo).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import tempfile
import threading
import urllib.request
from datetime import datetime, time
from pathlib import Path

from stockpredictor import store

log = logging.getLogger(__name__)
STATE_BRANCH = "paper-state"
STATE_FILE = "state.db"
PRIVATE_TABLES = ("portfolio_trades",)
PUBLISH_EVERY = 180          # seconds
RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}/{file}"


def fetch_state(dest: Path) -> bool:
    """Download the published state into `dest` (False: no state published yet)."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            if not store._branch_checkout(Path(tmp) / "s", STATE_BRANCH):
                return False
        except Exception as exc:
            log.warning("no paper state on GitHub yet (%s)", exc)
            return False
        src = Path(tmp) / "s" / STATE_FILE
        parts = sorted((Path(tmp) / "s").glob(STATE_FILE + ".gz.part*"))
        if not src.exists() and parts:            # seeded in small pieces (see seed_state)
            import gzip

            src.write_bytes(gzip.decompress(b"".join(p.read_bytes() for p in parts)))
        if not src.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        return True


def snapshot(db: Path, out: Path) -> None:
    """Consistent copy of the live database without the private tables."""
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
    dst.commit()
    dst.execute("VACUUM")
    dst.close()


def publish_state(db: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        snapshot(db, Path(tmp) / STATE_FILE)
        store.publish_branch(Path(tmp), STATE_BRANCH,
                             f"Paper state {datetime.now():%Y-%m-%d %H:%M}")


class Publisher(threading.Thread):
    """Publishes the state every PUBLISH_EVERY seconds while a session runs."""

    def __init__(self, db: Path):
        super().__init__(daemon=True)
        self.db, self.stop = db, threading.Event()

    def run(self) -> None:
        while not self.stop.wait(PUBLISH_EVERY):
            try:
                publish_state(self.db)
            except Exception as exc:
                log.warning("could not publish the paper state: %s", exc)


def _monitor(settings, store_dir: Path):
    from stockpredictor import db
    from stockpredictor.live import monitor, prices

    db.init_db(settings.db_path)
    conn = db.connect(settings.db_path)
    return monitor.Monitor(conn, store_dir, prices.LivePrices(settings),
                           settings.paper_capital_longterm,
                           capital_intraday=settings.paper_capital_intraday,
                           sync=lambda d: store.sync(d, years=4))   # recent years only


def session(settings, store_dir: Path, until: time) -> None:
    """Market hours on GitHub: start from the published state, run the live monitor until
    `until` (IST), publishing the state as it goes and at the end."""
    from stockpredictor.live import monitor as MON

    from stockpredictor.live.prices import now_ist

    fetch_state(Path(settings.db_path))
    mon = _monitor(settings, store_dir)
    now = now_ist()
    if now.time() >= until or not mon.trading_today(now):
        log.info("nothing to do now (market closed today, or past %s)", until.strftime("%H:%M"))
        return
    pub = Publisher(Path(settings.db_path))
    pub.start()
    try:
        MON.run_forever(mon, until=until, hard_stop=True)
    finally:
        pub.stop.set()
        publish_state(Path(settings.db_path))


def evening(settings, store_dir: Path) -> None:
    """After the evening data update: long-term decisions, judging and the paper feedback
    for tonight's training, then publish the state."""
    from stockpredictor.live.prices import now_ist

    fetch_state(Path(settings.db_path))
    mon = _monitor(settings, store_dir)
    now = now_ist()
    try:
        if not (now.weekday() < 5 and mon.after_close_job(now)):   # (it sends the feedback)
            log.warning("after-close work not done today; sending the paper feedback anyway")
            mon.push_feedback()
    finally:
        publish_state(Path(settings.db_path))


def download_view(dest: Path, repo: str) -> bool:
    """Viewer (Mac / any dashboard): the newest published state, over plain HTTPS."""
    url = RAW_URL.format(repo=repo, branch=STATE_BRANCH, file=STATE_FILE)
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
        return True
    except Exception as exc:
        log.warning("could not download the paper state: %s", exc)
        return False


def seed_state(db: Path, piece_kb: int = 15) -> int:
    """First upload from a Mac whose network drops larger uploads: the compressed state in
    small pieces, one push each. GitHub's runs then publish it whole."""
    import gzip

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        snapshot(db, work / "state.db")
        data = gzip.compress((work / "state.db").read_bytes(), 9)
        (work / "state.db").unlink()
        repo = work / "repo"
        repo.mkdir()
        store._git("init", "-q", "-b", STATE_BRANCH, cwd=repo)
        size, n = piece_kb * 1024, 0
        for i in range(0, len(data), size):
            (repo / f"{STATE_FILE}.gz.part{n:04d}").write_bytes(data[i:i + size])
            store._git("add", "-A", cwd=repo)
            store._git("-c", "user.name=stockpredictor", "-c", "user.email=s@localhost",
                       "commit", "-q", "-m", f"state piece {n}", cwd=repo)
            for _ in range(5):
                try:
                    store._git("push", "-q", "--force", store.default_remote(),
                               f"{STATE_BRANCH}:refs/heads/{STATE_BRANCH}", cwd=repo)
                    break
                except Exception:
                    import time as _t
                    _t.sleep(3)
            else:
                raise RuntimeError(f"could not upload piece {n}")
            n += 1
        return n
