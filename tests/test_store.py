from stockpredictor import db, store


def seed(conn):
    conn.execute("INSERT INTO stocks (symbol, name, industry, isin, active) "
                 "VALUES ('ABC', 'Abc Ltd', 'IT', 'X1', 1)")
    conn.executemany(
        "INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [("ABC", "2023-12-29", 10, 11, 9, 10.5, 10.4, 100, "yahoo"),
         ("ABC", "2024-01-01", 10.5, 12, 10, 11.25, 11.2, 200, "yahoo")])
    conn.execute("INSERT INTO index_prices VALUES "
                 "('NIFTY50', '2024-01-01', 1, 2, 0.5, 1.5, 1.5, 0, 'yahoo')")
    conn.execute("INSERT INTO corporate_actions VALUES ('ABC', '2024-01-01', 'split', 2.0)")
    conn.commit()


def test_export_import_roundtrip(tmp_path):
    src, dst, out = tmp_path / "a.db", tmp_path / "b.db", tmp_path / "store"
    db.init_db(src)
    db.init_db(dst)
    with db.connect(src) as conn:
        seed(conn)
        store.export_store(conn, out)
    assert sorted(p.name for p in (out / "daily").glob("*.csv")) == ["2023.csv", "2024.csv"]

    with db.connect(dst) as conn:
        store.import_store(conn, out)
        rows = conn.execute("SELECT symbol, date, close, volume FROM daily_prices "
                            "ORDER BY date").fetchall()
        assert [tuple(r) for r in rows] == [("ABC", "2023-12-29", 10.5, 100),
                                           ("ABC", "2024-01-01", 11.25, 200)]
        assert conn.execute("SELECT value FROM corporate_actions").fetchone()[0] == 2.0

    daily = store.load_daily(out)
    assert list(daily["close"]) == [10.5, 11.25]
    assert store.load_indices(out)["symbol"].tolist() == ["NIFTY50"]


def test_export_is_deterministic(tmp_path):
    path = tmp_path / "a.db"
    db.init_db(path)
    with db.connect(path) as conn:
        seed(conn)
        store.export_store(conn, tmp_path / "s1")
        store.export_store(conn, tmp_path / "s2")
    assert (tmp_path / "s1/daily/2024.csv").read_text() == (tmp_path / "s2/daily/2024.csv").read_text()


def test_sync_keeps_only_latest_snapshot(tmp_path):
    import subprocess

    def git(*a, cwd):
        subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)

    remote = tmp_path / "remote"
    remote.mkdir()
    git("init", "-q", "-b", "market-data", cwd=remote)
    git("config", "user.email", "t@t", cwd=remote)
    git("config", "user.name", "t", cwd=remote)
    (remote / "daily").mkdir()
    for day in ("2024-01-01", "2024-01-02"):
        (remote / "daily" / "2024.csv").write_text(
            "symbol,date,open,high,low,close,adj_close,volume,source\n"
            f"ABC,{day},1,1,1,1,1,1,yahoo\n")
        git("add", "-A", cwd=remote)
        git("commit", "-q", "-m", day, cwd=remote)
        store.sync(tmp_path / "local", remote=f"file://{remote}")

    local = tmp_path / "local"
    assert store.load_daily(local)["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]
    log = subprocess.run(["git", "log", "--oneline"], cwd=local, capture_output=True, text=True)
    assert len(log.stdout.strip().splitlines()) == 1
