"""Stock Predictor dashboard.  Start with:  python -m stockpredictor app"""

from __future__ import annotations

import json
from datetime import date, datetime
from datetime import time as dtime
from pathlib import Path

import pandas as pd
import streamlit as st

from stockpredictor import config, db, store
from stockpredictor.app import charts as C
from stockpredictor.config import load_settings
from stockpredictor.data import fundamentals as FUND
from stockpredictor.data import news as N
from stockpredictor.features.labels import reason_text
from stockpredictor.live.prices import in_market_hours, now_ist
from stockpredictor.models import longterm as M
from stockpredictor.models import trainer as T
from stockpredictor.paper import daily as D
from stockpredictor.backtest import intraday as BI
from stockpredictor.models import intraday as MI
from stockpredictor.paper import engine as E
from stockpredictor.paper import intraday as PI
from stockpredictor.paper import scoreboard as SB
from stockpredictor.portfolio import real as R

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

SETTINGS = load_settings()
STORE = store.DEFAULT_STORE_DIR
MARKET_CLOSE = dtime(15, 30)
MODEL_NAMES = {"longterm": "Long-term", "intraday": "Intraday 12:30",
               "intraday_close": "Intraday close"}
REFRESH = "60s"   # always: a page opened before 9:15 must still go live


# --- Shared data ------------------------------------------------------------------

def conn():
    db.init_db(SETTINGS.db_path)
    return db.connect(SETTINGS.db_path)


def _store_version() -> float:
    files = list(STORE.glob("*/*.csv"))
    return max((f.stat().st_mtime for f in files), default=0)


@st.cache_resource(show_spinner="Loading market data…")
def market(version: float) -> E.MarketContext | None:
    try:
        return E.MarketContext.load(STORE)
    except FileNotFoundError:
        return None


def ctx() -> E.MarketContext | None:
    return market(_store_version())


def live_prices(c) -> dict[str, float]:
    return {r["symbol"]: r["price"] for r in c.execute("SELECT symbol, price FROM live_prices")}


def current_prices(c) -> dict[str, float]:
    """Live prices where the monitor has them, otherwise latest close."""
    m = ctx()
    base = m.closes_on(m.daily["date"].max()) if m else {}
    return {**base, **live_prices(c)}


@st.cache_data(show_spinner=False)
def _prev_closes(version: float, day: str) -> dict[str, float]:
    m = market(version)
    if m is None:
        return {}
    d = m.daily[m.daily["date"] < pd.Timestamp(day)].sort_values("date")
    return d.groupby("symbol")["close"].last().to_dict()


def prev_closes() -> dict[str, float]:
    """Yesterday's close per stock (the base for today's % up/down)."""
    return _prev_closes(_store_version(), f"{now_ist():%Y-%m-%d}")


def closing_prices(day: str) -> dict[str, float]:
    """Market close (15:30) on `day`: the official close once the evening data is in; before
    that, on the day itself after 15:30, the last live price."""
    m, now = ctx(), now_ist()
    out = {}
    if m is not None:
        d = m.daily[m.daily["date"] == pd.Timestamp(day)]
        out = dict(zip(d["symbol"], d["close"]))
    if not out and day == f"{now:%Y-%m-%d}" and now.time() >= MARKET_CLOSE:
        out = live_prices(conn())
    return out


def pct(new: pd.Series, base: pd.Series) -> pd.Series:
    return pd.to_numeric(new, errors="coerce") / pd.to_numeric(base, errors="coerce") - 1


PCT = st.column_config.NumberColumn(format="%+.2f%%")


def as_pct(x: pd.Series) -> pd.Series:
    return (x * 100).round(2)


def setting(c, key, default):
    row = c.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def put(c, key, value):
    c.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, str(value)))
    c.commit()


def dash(values, fmt: str) -> list[str]:
    """Format numbers for display, with '—' where there is no value yet."""
    return ["—" if v is None or pd.isna(v) else fmt.format(v) for v in values]


def rupees(v: float) -> str:
    return f"₹{v:,.0f}"


def need_data() -> bool:
    if ctx() is None:
        st.warning("No market data yet. In the terminal run: `python -m stockpredictor sync-data`")
        return True
    return False


# --- Sidebar -----------------------------------------------------------------------

def sidebar():
    c = conn()
    with st.sidebar:
        m = ctx()
        if m is not None:
            st.caption(f"Data up to **{m.daily['date'].max():%d %b %Y}**")
        hb = setting(c, "monitor_heartbeat", None)
        if hb:
            age = (now_ist() - datetime.fromisoformat(hb)).total_seconds() / 60
            state = "🟢 running" if age < 3 else f"⚪ last seen {age:.0f} min ago"
            st.caption(f"Live monitor: {state} · prices from {setting(c, 'live_source', '—')}")
        else:
            st.caption("Live monitor: ⚪ not started (`python -m stockpredictor run`)")
        st.divider()
        accuracy_now_sidebar()
        st.divider()
        st.subheader("🔔 Alerts")
        alerts = c.execute("SELECT ts, kind, message FROM alerts ORDER BY id DESC LIMIT 8").fetchall()
        if not alerts:
            st.caption("No alerts yet.")
        icons = {"stop-loss": "🛑", "negative-news": "📰", "decision": "🧭", "fill": "✅",
                 "info": "ℹ️"}
        for a in alerts:
            st.caption(f"{icons.get(a['kind'], '•')} {a['message']}")


@st.fragment(run_every=REFRESH)
def accuracy_now_sidebar():
    sb = SB.compute(conn(), now_ist().replace(tzinfo=None))
    st.subheader("📊 Accuracy now")
    it, ij, lo, lj = (sb["intraday_today"], sb["intraday_judged"], sb["longterm_open"],
                      sb["longterm_judged"])
    if it["buy_n"] + it["sell_n"]:
        st.metric("Intraday today", f"{it['right']:.0%}",
                  f"{(it['right'] - (it['random'] or 0)) * 100:+.0f} pts vs random"
                  if it["random"] is not None else None)
        st.caption(f"Buys {it['buy_right']}/{it['buy_n']} up · sells {it['sell_right']}/{it['sell_n']} down")
    else:
        st.caption("Intraday today: picks at 9:46")
    if ij["n"]:
        st.caption(f"Intraday judged: **{ij['accuracy']:.0%}** of {ij['n']} (random {ij['random']:.0%})")
    if lo["n"]:
        st.caption(f"Long-term this week: **{lo['on_track']}/{lo['n']}** on track")
    st.caption(f"Long-term judged: **{lj['accuracy']:.0%}** of {lj['n']} (random {lj['random']:.0%})"
               if lj["n"] else "Long-term judged: first results after 1 week")


# --- Pages -------------------------------------------------------------------------

def page_picks():
    st.title("Long-term picks")
    if need_data():
        return
    c, m = conn(), ctx()
    last = ensure_longterm_picks(c, m)
    accuracy_now_panel("longterm")
    preds = pd.read_sql("SELECT * FROM predictions WHERE horizon = 'longterm' AND date = ? "
                        "AND horizon_days = ?", c, params=(last, M.HORIZON))
    summary = N.news_summary(m.news, pd.Timestamp(now_ist())).set_index("symbol")
    fund = FUND.load_latest(STORE).set_index("symbol")
    st.caption(f"Predictions for the **next week** (5 trading days), made after the close of "
               f"**{pd.Timestamp(last):%d %b %Y}**. Each is judged after one week against Nifty 50. "
               f"Live price, Today % and Since pick % update every minute.")
    picks_tables(preds, summary, fund)
    sell_now(c, preds)
    live_block()

    st.subheader("📰 Latest headlines for picks")
    picks = preds.loc[preds["direction"] == "up", "symbol"].tolist()
    heads = m.news[m.news["symbol"].isin(picks)].sort_values("published", ascending=False).head(15)
    if heads.empty:
        st.caption("No news collected yet — the news job runs 4 times per trading day.")
    for h in heads.itertuples():
        mood = "🟢" if h.sentiment > 0.2 else "🔴" if h.sentiment < -0.2 else "⚪"
        st.markdown(f"{mood} **{h.symbol}** — [{h.title}]({h.url}) "
                    f"<span style='opacity:.6'>· {h.source} · {h.published:%d %b %H:%M}</span>",
                    unsafe_allow_html=True)


@st.fragment(run_every=REFRESH)
def picks_tables(preds: pd.DataFrame, summary: pd.DataFrame, fund: pd.DataFrame):
    c = conn()
    live, prev = current_prices(c), prev_closes()

    def table(direction):
        p = preds[preds["direction"] == direction].sort_values("confidence", ascending=False)
        rows = []
        for r in p.itertuples():
            sent = summary["sent_mean_7d"].get(r.symbol)
            now = live.get(r.symbol)
            rows.append({
                "Stock": r.symbol,
                "Confidence": r.confidence, "Pick price": r.entry_price,
                "Live price": now,
                "Today %": as_pct(pct(pd.Series([now]), pd.Series([prev.get(r.symbol)])))[0],
                "Since pick %": as_pct(pct(pd.Series([now]), pd.Series([r.entry_price])))[0],
                "Why": reason_text(json.loads(r.reasons or "[]")),
                "News (7d)": "—" if pd.isna(sent) else
                ("🟢 positive" if sent > 0.2 else "🔴 negative" if sent < -0.2 else "⚪ neutral"),
                "P/E": fund["pe"].get(r.symbol), "ROE": fund["roe"].get(r.symbol),
            })
        df = pd.DataFrame(rows).rename(columns={"Why": "Signals (↑ raised score, ↓ lowered it)"})
        df["P/E"] = dash(pd.to_numeric(df["P/E"], errors="coerce"), "{:.1f}")
        df["ROE"] = dash(pd.to_numeric(df["ROE"], errors="coerce"), "{:.1%}")
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Pick price": st.column_config.NumberColumn(format="₹%.2f"),
            "Live price": st.column_config.NumberColumn(format="₹%.2f"),
            "Today %": PCT, "Since pick %": PCT,
            "Signals (↑ raised score, ↓ lowered it)": st.column_config.TextColumn(width="large")})

    st.subheader("▲ 10 Buy candidates — expected to rise more than Nifty next week")
    table("up")


def ensure_longterm_picks(c, m) -> str:
    """Make sure 1-week predictions exist for the latest data day (runs the model if not)."""
    latest = f"{m.feats['date'].max():%Y-%m-%d}"
    have = c.execute("SELECT COUNT(*) FROM predictions WHERE horizon = 'longterm' AND date = ? "
                     "AND horizon_days = ?", (latest, M.HORIZON)).fetchone()[0]
    if not have:
        with st.spinner("Making this week's predictions (first time after an update can take "
                        "~30 s while the model trains)…"):
            E.ensure_account(c, SETTINGS.paper_capital_longterm)
            model = D.load_or_train(m)
            rules, mode = D.get_rules(c)
            E.run_decision(c, m, model, pd.Timestamp(latest), rules, mode)
    return latest


def sell_now(c, preds: pd.DataFrame):
    """Holdings (paper and yours) that the model now expects to fall."""
    st.subheader("🔔 Sell now — holdings the model expects to fall")
    ranks = dict(c.execute("SELECT symbol, rank FROM lt_scores").fetchall())
    n = len(ranks) or 100
    rules, _ = D.get_rules(c)
    rows = []
    held = [("Paper", s) for s in E.holdings(c)]
    real = R.holdings(c, "longterm")
    if not real.empty:
        held += [("My portfolio", s) for s in real.loc[real["qty"] > 0, "symbol"]]
    for who, sym in held:
        r = ranks.get(sym)
        reasons = []
        if r is not None and r > rules.exit_rank:
            reasons.append(f"trading rank {r} of {n} (below {rules.exit_rank})")
        if reasons:
            rows.append({"Where": who, "Stock": sym, "Why": "; ".join(reasons)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("None of your paper or real long-term holdings is flagged — nothing to sell.")


@st.fragment(run_every=REFRESH)
def live_block():
    # Fragments re-run in another thread: always open a fresh connection here.
    c = conn()
    live = pd.read_sql("SELECT * FROM live_scores ORDER BY rank", c)
    if live.empty:
        return
    st.subheader("⚡ Live ranking (provisional)")
    st.caption(f"Re-scored at {live['ts'].iloc[0]} with live prices. Official decisions are made "
               "after the close; ↑/↓ shows movement vs the last official ranking.")
    live["Move"] = [("—" if pd.isna(o) else f"↑ {int(o - r)}" if o > r else
                     f"↓ {int(r - o)}" if o < r else "=") for r, o in
                    zip(live["rank"], live["official_rank"])]
    st.dataframe(live.head(15)[["rank", "symbol", "Move", "price", "official_rank"]].rename(
        columns={"rank": "Live rank", "symbol": "Stock", "price": "Live price",
                 "official_rank": "Official rank"}), hide_index=True, width="stretch",
        column_config={"Live price": st.column_config.NumberColumn(format="₹%.2f")})


def page_paper_longterm():
    st.title("Paper trading — Long-term")
    rules, _ = D.get_rules(conn())
    st.caption(f"Buy-only virtual portfolio (₹1 lakh). It holds the top {rules.n_hold} stocks "
               "with the strongest sustained buy signal (the Long-term picks page lists the top "
               "10) and sells a holding when the model expects it to fall (rank drops below the "
               "exit rank), on a stop-loss, or on severe bad news. Decisions after the close, "
               "orders filled at the next market price.")
    if need_data():
        return
    accuracy_now_panel("longterm")
    paper_longterm()


def page_paper_intraday():
    st.title("Paper trading — Intraday")
    st.caption("Two books, each starting every day with its own ₹1 lakh. At 9:46 each model buys "
               "the stocks it expects to rise and short-sells the ones it expects to fall, with "
               "a stop-loss and target. Book 1 closes everything at 12:30, book 2 at 15:15 "
               "(the close). After the close they are compared and both learn.")
    tab1, tab2, tab3, tab4 = st.tabs(["Until 12:30", "Until close (15:15)", "🏆 Competition",
                                      "🔁 Historical replays"])
    with tab1:
        accuracy_now_panel(PI.HORIZON)
        paper_intraday(PI.HORIZON)
    with tab2:
        accuracy_now_panel(PI.CLOSE_HORIZON)
        paper_intraday(PI.CLOSE_HORIZON)
    with tab3:
        competition()
    with tab4:
        replay_history()


def replay_history():
    """After each close: 3 rounds of paper trading on past days, each learning from the last."""
    c = conn()
    st.subheader("🔁 Historical paper-trading replays")
    st.caption(f"After every close each intraday model paper-trades the last {T.REPLAY_DAYS} days "
               f"again in {T.REPLAY_ROUNDS} rounds (₹1 lakh a day, same rules and costs). Every "
               "day is traded by a model trained only on earlier days; each round learns from "
               "the previous round's judged buy and sell picks (wrong 2×, right 1.5×). If the "
               "last round beats the first, that learning is kept for the live model.")
    tuning_checks(c)
    runs = pd.read_sql("SELECT * FROM replay_runs ORDER BY id", c)
    if runs.empty:
        st.caption("The first replay runs after today's close (or now, at the weekend).")
        return
    last = runs[runs["run_at"] == runs["run_at"].max()]
    st.caption(f"Latest run {last['run_at'].iloc[0][:16].replace('T', ' ')} · "
               f"{last['period'].iloc[0]}")
    for h, g in last.groupby("horizon", sort=False):
        kept = bool(g["adopted"].max())
        st.markdown(f"**{MODEL_NAMES.get(h, h)}** — "
                    + ("✅ learning kept (last round beat the first)" if kept
                       else "➖ no improvement over the rounds, not used"))
        st.dataframe(pd.DataFrame({
            "Round": g["round"], "Days": g["days"], "Avg P&L per day": g["avg_day_pnl"],
            "Profitable days": g["win_days"], "Picks right": g["accuracy"],
            "Random picks": g["random"], "IC": g["ic"]}), hide_index=True, width="stretch",
            column_config={"Avg P&L per day": st.column_config.NumberColumn(format="₹%.0f"),
                           "Profitable days": st.column_config.NumberColumn(format="percent"),
                           "Picks right": st.column_config.NumberColumn(format="percent"),
                           "Random picks": st.column_config.NumberColumn(format="percent"),
                           "IC": st.column_config.NumberColumn(format="%.3f")})
    daily = runs[runs["round"] == runs.groupby("run_at")["round"].transform("max")]
    if daily["run_at"].nunique() >= 2:
        st.caption("Final-round accuracy of each daily replay")
        d = daily.assign(date=pd.to_datetime(daily["run_at"]),
                         series=daily["horizon"].map(MODEL_NAMES), value=daily["accuracy"])
        st.altair_chart(C.lines(d[["date", "series", "value"]], "date", "value", "series", ".0%"),
                        width="stretch")


def tuning_checks(c):
    """The paper-trading check run after every background tuning round."""
    st.subheader("🧪 Paper-trading check after each tuning round")
    st.caption(f"After every tuning round the settings in use are paper-traded on the last "
               f"{T.REPLAY_DAYS} days (live rules: top buys + sells, ₹1 lakh a day, costs; each "
               "day predicted by a model trained only on earlier days). New settings are kept "
               "only if they also paper-trade at least as well (P&L per day and picks right).")
    t = pd.read_sql("SELECT * FROM tune_checks ORDER BY id DESC LIMIT 40", c)
    if t.empty:
        st.caption("Appears after the next tuning round (every few minutes).")
        return
    st.dataframe(pd.DataFrame({
        "Time": t["run_at"].str[5:16].str.replace("T", " "),
        "Model": t["horizon"].map(MODEL_NAMES), "Days": t["days"],
        "Avg P&L per day": t["avg_day_pnl"], "Profitable days": t["win_days"],
        "Picks right": t["accuracy"], "Random picks": t["random"], "IC": t["ic"],
        "Settings": ["✅ new adopted" if a else "kept" for a in t["adopted"]]}),
        hide_index=True, width="stretch", column_config={
            "Avg P&L per day": st.column_config.NumberColumn(format="₹%.0f"),
            "Profitable days": st.column_config.NumberColumn(format="percent"),
            "Picks right": st.column_config.NumberColumn(format="percent"),
            "Random picks": st.column_config.NumberColumn(format="percent"),
            "IC": st.column_config.NumberColumn(format="%.3f")})


def competition():
    """Day by day: which book did better, and the running score."""
    c = conn()
    cap = SETTINGS.paper_capital_intraday
    a = PI.daily_results(c, cap, PI.HORIZON)
    b = PI.daily_results(c, cap, PI.CLOSE_HORIZON)
    st.subheader("🏆 12:30 vs close — who wins each day?")
    if a.empty or b.empty:
        st.caption("The competition starts on the first trading day both books trade "
                   "(results after 15:15).")
        return
    m = a.merge(b, on="date", suffixes=("_1230", "_close"))
    m = m[(m["open_1230"] == 0) & (m["open_close"] == 0)]
    if m.empty:
        st.caption("Today's result appears after 15:15.")
        return
    m["Winner"] = ["12:30" if x > y else "Close" if y > x else "Tie"
                   for x, y in zip(m["pnl_1230"], m["pnl_close"])]
    wins = m["Winner"].value_counts()
    k = st.columns(4)
    k[0].metric("Days compared", len(m))
    k[1].metric("12:30 book won", int(wins.get("12:30", 0)))
    k[2].metric("Close book won", int(wins.get("Close", 0)))
    lead = m["pnl_1230"].sum() - m["pnl_close"].sum()
    k[3].metric("P&L difference (12:30 − close)", C.money(lead))
    t = m.iloc[::-1]
    st.dataframe(pd.DataFrame({
        "Day": t["date"], "Winner": t["Winner"],
        "12:30 P&L": t["pnl_1230"], "Close P&L": t["pnl_close"],
        "12:30 picks right": (t["accuracy_1230"] * 100).round(0),
        "Close picks right": (t["accuracy_close"] * 100).round(0)}),
        hide_index=True, width="stretch", column_config={
            "12:30 P&L": st.column_config.NumberColumn(format="₹%.0f"),
            "Close P&L": st.column_config.NumberColumn(format="₹%.0f"),
            "12:30 picks right": st.column_config.NumberColumn(format="%.0f%%"),
            "Close picks right": st.column_config.NumberColumn(format="%.0f%%")})
    st.caption("How they learn from each other: each model's self-tuning may blend in the other "
               "model's ranking when that makes it more accurate on days it hasn't seen — the "
               "blend is kept only if it wins. Both retrain on each day's full session after the "
               "close.")
    exit_comparison()


@st.fragment(run_every=REFRESH)
def paper_longterm():
    c = conn()
    E.ensure_account(c, SETTINGS.paper_capital_longterm)
    prices = current_prices(c)
    v = E.value(c, prices)
    k = st.columns(4)
    k[0].metric("Capital", rupees(v["capital"]))
    k[1].metric("Current value", rupees(v["equity"]), C.money(v["pnl"]))
    k[2].metric("Cash", rupees(v["cash"]))
    k[3].metric("Holdings", f"{v['positions']}")
    with st.expander("Why can these differ from this week's 10 buy candidates?"):
        st.write("The buy candidates are the model's best guesses for **next week**. Trading on "
                 "them directly would change most holdings every week, and in the backtest the "
                 "charges ate all the profit. The paper portfolio therefore buys the stocks whose "
                 "weekly prediction has been strong **over the last 20 days** (blended with "
                 "12-month momentum) and holds them until they fall out of the top half. In the "
                 "2015-2026 backtest (10 holdings) this earned ~20% a year after costs vs ~9% "
                 "for Nifty 50.")
    paper_book(c, E.HORIZON, "Holdings", prices)
    eq = pd.read_sql("SELECT date, equity FROM paper_equity WHERE horizon = ? ORDER BY date",
                     c, params=(E.HORIZON,))
    if len(eq) >= 2:
        st.subheader("Value over time vs Nifty 50")
        m = ctx()
        eq["date"] = pd.to_datetime(eq["date"])
        nifty = m.indices[m.indices["symbol"] == "NIFTY50"].set_index("date")["close"]
        n = nifty.reindex(eq["date"]).ffill().bfill().values
        df = pd.concat([eq.assign(series="Buy book", value=eq["equity"]),
                        eq.assign(series="Nifty 50", value=n / n[0] * eq["equity"].iloc[0])])
        st.altair_chart(C.lines(df[["date", "series", "value"]], "date", "value", "series",
                                ",.0f", "Value (₹)"), width="stretch")


def paper_book(c, h: str, title: str, prices: dict[str, float]):
    """Two tables: what the paper account is buying (holdings + queued buys) and what it is
    selling (queued sells + stocks already sold)."""
    rules, _ = D.get_rules(c)
    sign = E.book_sign(h)
    buy_word, sell_word = ("Buying", "Selling") if sign > 0 else ("Shorting", "Covering")
    pending = pd.read_sql("SELECT created, symbol, side, reason FROM paper_orders "
                          "WHERE horizon = ? AND status = 'pending'", c, params=(h,))
    pending["Live price"] = pending["symbol"].map(prices)
    pending["Today %"] = as_pct(pct(pending["Live price"], pending["symbol"].map(prev_closes())))
    order_cols = {"created": "Decided", "symbol": "Stock", "reason": "Why"}
    order_cfg = {"Live price": st.column_config.NumberColumn(format="₹%.2f"), "Today %": PCT}

    st.subheader(f"▲ {buy_word} — {title.lower()}")
    pos = pd.read_sql("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'open'", c, params=(h,))
    if pos.empty:
        st.caption("No open positions yet — orders fill at the next market price after a decision.")
    else:
        pos["Live price"] = pos["symbol"].map(prices).fillna(pos["entry_price"])
        pos["P&L"] = sign * (pos["Live price"] - pos["entry_price"]) * pos["qty"] - pos["costs"]
        pos["P&L %"] = sign * (pos["Live price"] / pos["entry_price"] - 1)
        pos["Today %"] = as_pct(pct(pos["Live price"], pos["symbol"].map(prev_closes())))
        pos["Since entry %"] = as_pct(pct(pos["Live price"], pos["entry_price"]))
        pos["Stop-loss"] = pos["entry_price"] * (1 - sign * rules.stop_loss)
        pos["Since"] = pd.to_datetime(pos["entry_time"]).dt.strftime("%d %b %Y")
        entry = "Buy price" if sign > 0 else "Short price"
        st.dataframe(pos[["symbol", "qty", "entry_price", "Live price", "Today %", "Since entry %",
                          "P&L", "P&L %", "Stop-loss", "Since"]].rename(
            columns={"symbol": "Stock", "qty": "Qty", "entry_price": entry,
                     "P&L": "P&L (after costs)"}),
            hide_index=True, width="stretch", column_config={
                c_: st.column_config.NumberColumn(format="₹%.2f")
                for c_ in [entry, "Live price", "P&L (after costs)", "Stop-loss"]} | {
                "P&L %": st.column_config.NumberColumn(format="percent"),
                "Today %": PCT, "Since entry %": PCT})
    buys = pending[pending["side"] == "buy"]
    if not buys.empty:
        st.caption(f"Queued {buy_word.lower()} orders (fill at the next market price):")
        st.dataframe(buys[["created", "symbol", "Live price", "Today %", "reason"]].rename(
            columns=order_cols), hide_index=True, width="stretch", column_config=order_cfg)

    st.subheader(f"▼ {sell_word}")
    sells = pending[pending["side"] == "sell"]
    if sells.empty:
        st.caption(f"No {sell_word.lower()} orders queued.")
    else:
        st.caption(f"Queued {sell_word.lower()} orders (fill at the next market price):")
        st.dataframe(sells[["created", "symbol", "Live price", "Today %", "reason"]].rename(
            columns=order_cols), hide_index=True, width="stretch", column_config=order_cfg)
    closed = pd.read_sql("SELECT * FROM paper_trades WHERE horizon = ? AND status = 'closed' "
                         "ORDER BY exit_time DESC", c, params=(h,))
    st.caption(f"{'Sold' if sign > 0 else 'Covered'} ({len(closed)})"
               + (f" · won {(closed['pnl'] > 0).mean():.0%} · total P&L "
                  f"{C.money(closed['pnl'].sum())}" if len(closed) else " — none yet"))
    if not closed.empty:
        closed["Return"] = sign * (closed["exit_price"] / closed["entry_price"] - 1)
        closed["Result"] = ["✅ profit" if p > 0 else "❌ loss" for p in closed["pnl"]]
        st.dataframe(closed[["symbol", "entry_time", "entry_price", "exit_time", "exit_price",
                             "Return", "pnl", "exit_reason", "Result"]].rename(columns={
            "symbol": "Stock", "entry_time": "Bought", "entry_price": "Buy price",
            "exit_time": "Sold", "exit_price": "Sell price", "pnl": "P&L (after costs)",
            "exit_reason": "Why sold"}), hide_index=True, width="stretch", column_config={
                "Return": st.column_config.NumberColumn(format="percent"),
                "P&L (after costs)": st.column_config.NumberColumn(format="₹%.0f")})


def page_intraday_1230():
    page_intraday(MI.TRADE)


def page_intraday_close():
    page_intraday(MI.CLOSE)
    st.divider()
    exit_comparison()


def page_intraday(target: MI.Target):
    first = target == MI.TRADE
    st.title("Intraday — until 12:30" if first else "Intraday — until close")
    accuracy_now_panel(target.horizon)
    c = conn()
    rules, enabled = PI.get_rules(c)
    exit_at = "12:30" if first else "15:15 (close)"
    st.caption(f"At 9:45 this model ranks all Nifty 250 stocks on the first 30 minutes and "
               f"predicts the move until {exit_at}. Stop-loss {rules.stop_loss}%, target "
               f"{rules.target or 'none'}%, squared off at {exit_at}. It lists 10 buy and 10 sell "
               f"candidates; 🧪 marks the {rules.n_long} + {rules.n_short} strongest that are "
               "paper-traded on this model's own ₹1 lakh. Both intraday models keep learning "
               "until the close, compete, and can learn from each other.")
    if not enabled:
        st.warning("Intraday is switched off in Settings.")
    last = c.execute("SELECT MAX(date) FROM predictions WHERE horizon = ?",
                     (target.horizon,)).fetchone()[0]
    if last:
        intraday_live(last, target)
    elif first:
        intraday_preview()
    else:
        st.caption("Its first picks appear at 9:46 on the next trading day.")


def exit_comparison():
    """Which is the better time to exit: 12:30 or the close?"""
    st.header("⚖️ Exit at 12:30 or at the close?")
    c, m = conn(), ctx()
    path = T.COMPARE_PATH
    if path.exists():
        r = json.loads(path.read_text())
        a, b = r[MI.TRADE.horizon], r[MI.CLOSE.horizon]
        st.subheader("History (walk-forward backtest, same days, costs included)")
        rows = [{"Model": "Until 12:30", **a}, {"Model": "Until the close", **b}]
        t = pd.DataFrame(rows)
        st.dataframe(pd.DataFrame({
            "Model": t["Model"], "Days": t["days"], "Avg P&L per day (₹1 lakh)": t["avg_day_pnl"],
            "Profitable days": t["win_days"], "Picks right": t["direction_accuracy"],
            "Random picks": t["random_baseline"], "Prediction quality (IC)": t["ic"]}),
            hide_index=True, width="stretch", column_config={
                "Avg P&L per day (₹1 lakh)": st.column_config.NumberColumn(format="₹%.0f"),
                "Profitable days": st.column_config.NumberColumn(format="percent"),
                "Picks right": st.column_config.NumberColumn(format="percent"),
                "Random picks": st.column_config.NumberColumn(format="percent"),
                "Prediction quality (IC)": st.column_config.NumberColumn(format="%.3f")})
        best, other = (a, b) if a["avg_day_pnl"] >= b["avg_day_pnl"] else (b, a)
        name = "12:30" if best is a else "the close"
        st.caption(f"{r['period']} ({a['days']} days, each predicted by a model trained only on "
                   f"earlier days; same stop-loss/target rules). So far exiting at **{name}** "
                   f"earned more: {C.money(best['avg_day_pnl'])} vs "
                   f"{C.money(other['avg_day_pnl'])} per day. Updated "
                   f"{r['updated'][:16].replace('T', ' ')}, refreshed after every close.")
    else:
        st.caption("The historical comparison is computed in the background (needs the 12:30 "
                   "prices in the intraday history; a few minutes after start).")
    if m is not None:
        live = PI.exit_comparison(c, m.daily[["symbol", "date", "close"]])
        st.subheader("Live (judged days)")
        if live.empty:
            st.caption("Appears after the first trading day with both models.")
        else:
            st.dataframe(live, hide_index=True, width="stretch", column_config={
                "Accuracy": st.column_config.NumberColumn(format="percent"),
                "Random": st.column_config.NumberColumn(format="percent"),
                "Avg move in our favour": st.column_config.NumberColumn(format="percent")})
            st.caption("'Same picks held to the close' uses the official close once the evening "
                       "data is in. A few days prove little — trust the comparison after "
                       "several weeks.")


def intraday_preview():
    m = ctx()
    if m is None:
        return
    with st.spinner("Preparing a preview from the latest trading day…"):
        res = PI.preview(m, STORE)
    if res is None:
        st.info("Not enough intraday history yet (needs 40 trading days). Run the task "
                "'Angel One: download intraday history (one-time)'.")
        return
    picks, day = res
    st.info(f"Live picks appear at **9:46 AM** on trading days while the program runs. Meanwhile, "
            f"this is what the model would have picked on **{day:%d %b %Y}** (trained only on earlier "
            f"days) and how it turned out.")
    for side, title in (("long", "▲ 10 Buy — expected to rise 9:45 → 12:30"),
                        ("short", "▼ 10 Sell — expected to fall 9:45 → 12:30")):
        p = picks[picks["side"] == side].copy()
        right = (p["move"] > 0).mean()
        st.markdown(f"**{title}** · {right:.0%} moved the predicted way "
                    f"(random picks that day: {p['baseline'].iloc[0]:.0%})")
        p["Result"] = ["✅" if x > 0 else "❌" for x in p["move"]]
        p["Why"] = [reason_text(r) for r in p["reasons"]]
        st.dataframe(p[["symbol", "confidence", "c30", "px_1230", "move", "Result", "Why"]].rename(
            columns={"symbol": "Stock", "confidence": "Confidence", "c30": "Price 9:45",
                     "px_1230": "Price 12:30", "move": "Move (in our favour)",
                     "Why": "Signals (↑ raised score, ↓ lowered it)"}),
            hide_index=True, width="stretch", column_config={
                "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Price 9:45": st.column_config.NumberColumn(format="₹%.2f"),
                "Price 12:30": st.column_config.NumberColumn(format="₹%.2f"),
                "Move (in our favour)": st.column_config.NumberColumn(format="percent")})


@st.fragment(run_every=REFRESH)
def intraday_live(day: str, target: MI.Target = MI.TRADE):
    c = conn()
    traded_model = target == MI.TRADE
    until = "12:30" if traded_model else "close"
    exit_col = "Exit 12:30" if traded_model else "Exit 15:15"
    preds = pd.read_sql("SELECT * FROM predictions WHERE horizon = ? AND date = ?",
                        c, params=(target.horizon, day))
    live = live_prices(c)
    closes = closing_prices(day)
    st.subheader(f"Picks for {pd.Timestamp(day):%d %b %Y}")
    st.caption(f"{exit_col} = price at this model's square-off (judged here) · Live = now · "
               "Close 15:30 = market close (appears after 15:30).")
    traded = {(r[0], r[1]) for r in c.execute(
        "SELECT symbol, side FROM paper_trades WHERE horizon = ? AND entry_time LIKE ?",
        (target.horizon, f"{day}%"))}
    for direction, title in (("up", f"▲ 10 Buy candidates (expected to rise 9:45 → {until})"),
                             ("down", f"▼ 10 Sell candidates (expected to fall 9:45 → {until})")):
        p = preds[preds["direction"] == direction].sort_values("rank").copy()
        if p.empty:
            continue
        sign = 1 if direction == "up" else -1
        live_px = p["symbol"].map(live).astype(float)
        # Exit price at 12:30 once squared off; the live price until then.
        now_px = p["actual_exit"].astype(float).fillna(live_px)
        p[exit_col] = p["actual_exit"].astype(float)
        p["Live"] = live_px
        p["Close 15:30"] = p["symbol"].map(closes).astype(float)
        p["Move"] = dash(sign * (now_px / p["entry_price"] - 1), "{:+.2%}")
        p["Today %"] = as_pct(pct(live_px, p["symbol"].map(prev_closes())))
        p["Share since 9:45"] = as_pct(pct(now_px, p["entry_price"]))
        p["target"] = dash(p["target"], "₹{:,.2f}")
        p["Result"] = ["⏳" if pd.isna(x) else "✅" if x else "❌" for x in p["correct"]]
        p["Why"] = [reason_text(json.loads(r or "[]")) for r in p["reasons"]]
        side = "long" if direction == "up" else "short"
        p["Paper"] = ["🧪" if (s_, side) in traded else "" for s_ in p["symbol"]]
        st.markdown(f"**{title}**")
        cols = ["symbol", "Paper", "confidence", "entry_price", "stop_loss", "target",
                exit_col, "Live", "Close 15:30", "Today %", "Share since 9:45", "Move",
                "Result", "Why"]
        st.dataframe(p[cols].rename(columns={
            "symbol": "Stock", "confidence": "Confidence", "entry_price": "Entry 9:45",
            "stop_loss": "Stop-loss", "target": "Target", "Move": "Move (in our favour)",
            "Why": "Signals (↑ raised score, ↓ lowered it)"}), hide_index=True, width="stretch",
            column_config={"Confidence": st.column_config.ProgressColumn(
                format="percent", min_value=0, max_value=1),
                "Today %": PCT, "Share since 9:45": PCT,
                **{k_: st.column_config.NumberColumn(format="₹%.2f")
                   for k_ in ["Entry 9:45", "Stop-loss", exit_col, "Live", "Close 15:30"]}})


@st.fragment(run_every=REFRESH)
def paper_intraday(horizon: str = PI.HORIZON):
    c = conn()
    E.ensure_account(c, SETTINGS.paper_capital_intraday, horizon)
    live = live_prices(c)
    v = PI.value(c, live, horizon)
    k = st.columns(4)
    k[0].metric("Started today with", rupees(v["capital"]),
                help="Every trading day starts fresh with ₹1 lakh")
    k[1].metric("Value now", rupees(v["equity"]), C.money(v["pnl"]))
    k[2].metric("Cash", rupees(v["cash"]))
    k[3].metric("Open positions", f"{v['positions']}")
    exit_name = "Exit (12:30 or stop/target)" if horizon == PI.HORIZON \
        else "Exit (15:15 or stop/target)"
    trades = pd.read_sql("SELECT * FROM paper_trades WHERE horizon = ? "
                         "ORDER BY entry_time DESC, id", c, params=(horizon,))
    if trades.empty:
        st.caption("No intraday paper trades yet — they open at 9:46 on trading days.")
        return
    trades["Day"] = trades["entry_time"].str[:10]
    trades["sign"] = trades["side"].map({"long": 1, "short": -1})
    rules, _ = PI.get_rules(c)
    for side, title in (("long", f"▲ Buy trades — the top {rules.n_long} of the 10 buy candidates"),
                        ("short", f"▼ Sell trades (short) — the top {rules.n_short} of the 10 "
                                  "sell candidates")):
        t = trades[trades["side"] == side].copy()
        st.subheader(title)
        if t.empty:
            st.caption("None yet.")
            continue
        closed = t[t["status"] == "closed"]
        if not closed.empty:
            by_day = closed.groupby("Day")["pnl"].sum()
            st.caption(f"{len(closed)} closed trades · won {(closed['pnl'] > 0).mean():.0%} · "
                       f"profitable days {(by_day > 0).mean():.0%} · total P&L {C.money(closed['pnl'].sum())}")
        t["Live/Exit"] = [live.get(s_, e) if st_ == "open" else x
                          for s_, e, x, st_ in zip(t["symbol"], t["entry_price"], t["exit_price"], t["status"])]
        today = f"{now_ist():%Y-%m-%d}"
        t["Exit"] = t["exit_price"].where(t["status"] == "closed")
        t["Live"] = [live.get(s_) if d_ == today else None for s_, d_ in zip(t["symbol"], t["Day"])]
        close_by_day = {d_: closing_prices(d_) for d_ in t["Day"].unique()}
        t["Close 15:30"] = [close_by_day[d_].get(s_) for s_, d_ in zip(t["symbol"], t["Day"])]
        t["P&L"] = [p if st_ == "closed" else sg * (lv - e) * q - cst
                    for p, st_, sg, lv, e, q, cst in zip(t["pnl"], t["status"], t["sign"], t["Live/Exit"],
                                                        t["entry_price"], t["qty"], t["costs"])]
        t["Share move %"] = as_pct(pct(t["Live/Exit"], t["entry_price"]))
        t["Today %"] = [as_pct(pct(pd.Series([lv]), pd.Series([prev_closes().get(s_)])))[0]
                        if st_ == "open" else None
                        for s_, lv, st_ in zip(t["symbol"], t["Live/Exit"], t["status"])]
        t["Status"] = ["⏳ open" if x == "open" else ("✅ " if p > 0 else "❌ ") + (r or "")
                       for x, p, r in zip(t["status"], t["P&L"], t["exit_reason"])]
        st.dataframe(t[["Day", "symbol", "qty", "entry_price", "Exit", "Live", "Close 15:30",
                        "Share move %", "Today %", "stop_loss", "target", "P&L", "Status"]].rename(columns={
            "symbol": "Stock", "qty": "Qty", "entry_price": "Entry 9:45",
            "Exit": exit_name, "stop_loss": "Stop-loss",
            "target": "Target", "P&L": "P&L (after costs)"}), hide_index=True, width="stretch",
            column_config={k_: st.column_config.NumberColumn(format="₹%.2f")
                           for k_ in ["Entry 9:45", exit_name, "Live",
                                      "Close 15:30", "Stop-loss", "Target", "P&L (after costs)"]}
            | {"Share move %": PCT, "Today %": PCT})
    intraday_days(c, horizon)


def intraday_days(c, horizon: str = PI.HORIZON):
    """Every day's result on a fresh ₹1 lakh, compared day by day."""
    cap = SETTINGS.paper_capital_intraday
    d = PI.daily_results(c, cap, horizon)
    st.subheader("📅 Day by day (each day starts with ₹1 lakh)")
    if d.empty:
        st.caption("The first day's result appears after "
                   f"{'12:30' if horizon == PI.HORIZON else '15:15'}.")
        return
    done = d[d["open"] == 0]
    if len(done) >= 2:
        k = st.columns(4)
        k[0].metric("Days traded", len(done))
        k[1].metric("Profitable days", f"{(done['pnl'] > 0).mean():.0%}")
        k[2].metric("Average day", C.money(done["pnl"].mean()),
                    f"{done['pnl_pct'].mean():+.2%}")
        acc = done.dropna(subset=["accuracy"])
        if len(acc):
            k[3].metric("Pick accuracy", f"{acc['accuracy'].mean():.0%}",
                        f"{(acc['accuracy'] - acc['random']).mean() * 100:+.0f} pts vs random"
                        if acc["random"].notna().any() else None)
    if len(done) >= 10:        # is training making it better? recent days vs the ones before
        n = min(10, len(done) // 2)
        recent, before = done.tail(n), done.iloc[-2 * n:-n]
        st.caption(f"Last {n} days vs the {n} before: average P&L "
                   f"{C.money(recent['pnl'].mean())} vs {C.money(before['pnl'].mean())}, "
                   f"pick accuracy {recent['accuracy'].mean():.0%} vs {before['accuracy'].mean():.0%}.")
    t = d.iloc[::-1].copy()
    t["Status"] = ["⏳ trading" if o else ("✅ profit" if p > 0 else "❌ loss")
                   for o, p in zip(t["open"], t["pnl"])]
    t["Buys right"] = [f"{r}/{n}" if n else "—" for r, n in zip(t["buy_right"], t["buy_n"])]
    t["Sells right"] = [f"{r}/{n}" if n else "—" for r, n in zip(t["sell_right"], t["sell_n"])]
    t["Won"] = [f"{w}/{n}" for w, n in zip(t["won"], t["trades"])]
    t["P&L %"] = as_pct(t["pnl_pct"])
    for k_ in ("accuracy", "random"):
        t[k_] = (pd.to_numeric(t[k_]) * 100).round(0)
    st.dataframe(t[["date", "Status", "end_value", "pnl", "P&L %", "Won", "Buys right",
                    "Sells right", "accuracy", "random"]].rename(columns={
        "date": "Day", "end_value": "Ended with", "pnl": "P&L (after costs)",
        "accuracy": "Pick accuracy", "random": "Random picks"}), hide_index=True,
        width="stretch", column_config={
            "Ended with": st.column_config.NumberColumn(format="₹%.0f"),
            "P&L (after costs)": st.column_config.NumberColumn(format="₹%.0f"),
            "P&L %": PCT, "Pick accuracy": st.column_config.NumberColumn(format="%.0f%%"),
            "Random picks": st.column_config.NumberColumn(format="%.0f%%")})
    if len(done) >= 2:
        st.altair_chart(C.bars(done.assign(Day=done["date"].str[5:]), "Day", "pnl", ",.0f",
                               "P&L per day (₹)", ref=0), width="stretch")
        acc = done.dropna(subset=["accuracy"])
        if len(acc) >= 2:
            a = pd.concat([acc.assign(series="Model picks", value=acc["accuracy"]),
                           acc.assign(series="Random picks", value=acc["random"])])
            a["date"] = pd.to_datetime(a["date"])
            st.altair_chart(C.lines(a[["date", "series", "value"]].dropna(), "date", "value",
                                    "series", ".0%", "Picks right"), width="stretch")
    st.caption("Each day's picks and trades are kept. After the square-off they are judged and fed "
               "back into training — buy and sell picks, wrong calls weighted more — so later "
               "days can be compared with earlier ones.")


def page_portfolio():
    st.title("My portfolio")
    st.caption("Your real Groww trades, entered by hand. Prices update live while the monitor runs.")
    for tab, h in zip(st.tabs(["Long-term", "Intraday"]), R.HORIZONS):
        with tab:
            portfolio_tab(h)


def portfolio_tab(h: str):
    c = conn()
    with st.expander("➕ Add a trade", expanded=R.trades(c, h).empty):
        with st.form(f"add_{h}", clear_on_submit=True):
            cols = st.columns(6)
            m = ctx()
            options = sorted(m.universe["symbol"]) if m is not None else []
            sym = cols[0].selectbox("Stock", options, index=None, accept_new_options=True,
                                    placeholder="e.g. RELIANCE")
            side = cols[1].selectbox("Side", ["buy", "sell"])
            qty = cols[2].number_input("Quantity", min_value=1, step=1)
            price = cols[3].number_input("Price ₹", min_value=0.01, step=0.05, format="%.2f")
            charges = cols[4].number_input("Charges ₹", min_value=0.0, step=1.0)
            day = cols[5].date_input("Date", value=date.today())
            notes = st.text_input("Notes (optional)")
            if st.form_submit_button("Save trade", type="primary"):
                try:
                    R.add_trade(c, h, sym or "", side, int(qty), float(price), f"{day:%Y-%m-%d}",
                                float(charges), notes)
                    st.success("Saved.")
                except ValueError as exc:
                    st.error(str(exc))
    portfolio_live(h)
    t = R.trades(c, h)
    if not t.empty:
        with st.expander("All trades"):
            st.dataframe(t[["id", "trade_date", "symbol", "side", "qty", "price", "charges", "notes"]],
                         hide_index=True, width="stretch")
            tid = st.number_input("Trade id to delete", min_value=0, step=1, key=f"del_{h}")
            if st.button("Delete trade", key=f"delb_{h}") and tid:
                R.delete_trade(c, int(tid))
                st.rerun()


@st.fragment(run_every=REFRESH)
def portfolio_live(h: str):
    c = conn()
    s = R.summary(c, h, current_prices(c))
    k = st.columns(4)
    k[0].metric("Invested", rupees(s.invested))
    k[1].metric("Current value", rupees(s.value),
                C.money(s.unrealized) if s.invested else None)
    k[2].metric("Unrealized P&L", C.money(s.unrealized))
    k[3].metric("Realized P&L", C.money(s.realized))
    if s.table.empty:
        st.caption("No holdings.")
        return
    t = s.table
    t["Alert"] = ["🛑 below stop-loss" if p <= sl else "" for p, sl in zip(t["price"], t["stop_loss"])]
    t["Today %"] = as_pct(pct(t["price"], t["symbol"].map(prev_closes())))
    st.dataframe(t[["symbol", "qty", "avg_cost", "price", "Today %", "value", "unrealized", "unrealized_pct",
                    "stop_loss", "Alert"]].rename(columns={
        "symbol": "Stock", "qty": "Qty", "avg_cost": "Avg cost", "price": "Price",
        "value": "Value", "unrealized": "P&L", "unrealized_pct": "P&L %", "stop_loss": "Stop-loss"}),
        hide_index=True, width="stretch", column_config={
            k_: st.column_config.NumberColumn(format="₹%.2f")
            for k_ in ["Avg cost", "Price", "Value", "P&L", "Stop-loss"]} | {
            "P&L %": st.column_config.NumberColumn(format="percent"), "Today %": PCT})
    st.caption("Allocation")
    st.altair_chart(C.hbars(t.assign(w=t["weight"]), "symbol", "w", ".0%"),
                    width="stretch")


def page_accuracy():
    st.title("Accuracy")
    tab_lt, tab_id, tab_cl = st.tabs(["Long-term", "Intraday until 12:30", "Intraday until close"])
    with tab_lt:
        accuracy_now_panel("longterm")
        accuracy_tab("longterm")
        strategy_race()
    with tab_id:
        accuracy_now_panel("intraday")
        accuracy_tab("intraday")
    with tab_cl:
        accuracy_now_panel("intraday_close")
        accuracy_tab("intraday_close")


@st.fragment(run_every=REFRESH)
def accuracy_now_panel(horizon: str):
    """This model's accuracy, predicted-UP and predicted-DOWN picks in separate tables."""
    now = now_ist().replace(tzinfo=None)
    name = {"longterm": "Long-term", "intraday": "Intraday until 12:30",
            "intraday_close": "Intraday until close (15:15)"}[horizon]
    st.subheader(f"📊 {name} — accuracy · {now:%H:%M}")
    c = conn()
    acc = SB.by_direction(c, horizon, now, live_prices(c), prev_closes())
    sides = (("up", "▲ Predicted UP (buy picks)"),) if horizon == "longterm" else \
        (("up", "▲ Predicted UP (buy picks)"), ("down", "▼ Predicted DOWN (sell picks)"))
    cols = st.columns(2)
    for col, (d, title) in zip(cols, sides):
        df = pd.DataFrame(acc[d])
        df["vs random"] = ((pd.to_numeric(df["Accuracy"]) - pd.to_numeric(df["Random"])) * 100).round()
        for k_ in ("Accuracy", "Random"):
            df[k_] = (pd.to_numeric(df[k_]) * 100).round(1)
        with col:
            st.markdown(f"**{title}**")
            st.dataframe(df, hide_index=True, width="stretch", column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.0f%%"),
                "Random": st.column_config.NumberColumn(format="%.0f%%"),
                "vs random": st.column_config.NumberColumn(format="%+d pts")})
    st.caption(("Judged after 1 week: a buy pick is right if it beat Nifty 50 (long-term predicts "
                "only UP). 'Today' uses live prices vs yesterday's close." if horizon == "longterm"
                else "Judged at 12:30: a buy pick is right if it rose from 9:45, a sell pick if it "
                "fell." if horizon == "intraday"
                else "Judged at 15:15 (the close square-off): a buy pick is right if it rose from "
                "9:45, a sell pick if it fell.") + " 'Random' = picking stocks at random on the same days. "
               "Updates every minute.")


ACCURACY_TEXT = {
    "longterm": ("A long-term prediction is judged after 1 week: a buy pick is ✅ if it beat "
                 "Nifty 50 (long-term makes buy predictions only).",
                 "First results appear one week after the first prediction.", "vs Nifty"),
    "intraday": ("An intraday pick is judged at 12:30 the same day: a long is ✅ if the price rose "
                 "from 9:45, a short is ✅ if it fell. Random baseline = share of all Nifty 250 "
                 "stocks that moved that way.", "Results appear after the first trading day.",
                 "9:45 → 12:30"),
    "intraday_close": ("An until-close pick is judged at the 15:15 square-off: a buy is ✅ if the "
                       "price rose from 9:45, a sell is ✅ if it fell.",
                       "Results appear after the first trading day's close.", "9:45 → 15:15"),
}


def accuracy_tab(h: str):
    c = conn()
    a = E.accuracy(c, h)
    intro, waiting, ret_label = ACCURACY_TEXT[h]
    st.caption(intro)
    k = st.columns(4)
    k[0].metric("Predictions made", a["total"])
    k[1].metric("Judged so far", a["matured"])
    if a["matured"]:
        k[2].metric("Accuracy", f"{a['accuracy']:.0%}",
                    f"{(a['accuracy'] - a['random_baseline']) * 100:+.0f} pts vs random")
        k[3].metric("Random-pick baseline", f"{a['random_baseline']:.0%}")
        cols = st.columns(2)
        with cols[0]:
            st.caption("Accuracy by pick strength (dashed line = random picks)")
            bc = pd.DataFrame({"bucket": list(a["by_confidence"]),
                               "accuracy": list(a["by_confidence"].values())})
            st.altair_chart(C.bars(bc, "bucket", "accuracy", ref=a["random_baseline"],
                                   sort=list(a["by_confidence"])), width="stretch")
        with cols[1]:
            st.caption("Rolling accuracy (30 prediction days)")
            roll = a["rolling"].reset_index().rename(columns={"correct": "value"})
            roll["date"] = pd.to_datetime(roll["date"])
            st.altair_chart(C.lines(roll.assign(series="Model"), "date", "value", "series",
                                    ".0%"), width="stretch")
        up_name, down_name = ("Longs", "Shorts") if h.startswith("intraday") \
            else ("Up picks", "Weakest picks")
        down = f" · {down_name}: {a['accuracy_down']:.0%}" if pd.notna(a["accuracy_down"]) else ""
        st.caption(f"{up_name}: {a['accuracy_up']:.0%}{down}"
                   + (f" · Paper trades profitable: {a['profitable_trades']:.0%} of "
                      f"{a['closed_trades']}" if a.get("closed_trades") else ""))
    else:
        k[2].metric("Accuracy", "—")
        st.info(waiting)
    predictions_table(h, ret_label)


@st.fragment(run_every=REFRESH)
def predictions_table(h: str, ret_label: str):
    """Prediction vs reality; open predictions show the live price (latest close after hours)."""
    c = conn()
    preds = pd.read_sql("SELECT * FROM predictions WHERE horizon = ? "
                        "ORDER BY date DESC, direction DESC, rank", c, params=(h,))
    if preds.empty:
        return
    now = preds["symbol"].map(current_prices(c)).astype(float)
    judged = preds["correct"].notna()
    preds["Price now/at end"] = preds["actual_exit"].astype(float).where(judged, now)
    preds["Share %"] = as_pct(pct(preds["Price now/at end"], preds["entry_price"]))
    preds["Result"] = ["⏳ open" if pd.isna(x) else "✅" if x else "❌" for x in preds["correct"]]
    if h.startswith("intraday"):
        preds["direction"] = preds["direction"].map({"up": "long ▲", "down": "short ▼"})
    st.subheader("Prediction vs reality")
    st.caption(f"Open predictions: live price, updated every minute (after 15:30 the last "
               f"price of the day) · {now_ist():%H:%M}")
    st.dataframe(preds[["date", "symbol", "direction", "confidence", "entry_price",
                        "Price now/at end", "Share %", "actual_return", "Result"]].rename(columns={
        "date": "Date", "symbol": "Stock", "direction": "Predicted", "confidence": "Confidence",
        "entry_price": "Price then", "actual_return": ret_label}), hide_index=True,
        width="stretch",
        column_config={"Confidence": st.column_config.NumberColumn(format="percent"),
                       ret_label: st.column_config.NumberColumn(format="percent"),
                       "Share %": PCT,
                       "Price then": st.column_config.NumberColumn(format="₹%.2f"),
                       "Price now/at end": st.column_config.NumberColumn(format="₹%.2f")})


def page_model():
    st.title("Model")
    path = M.MODEL_DIR / "meta.json"
    if path.exists():
        meta = json.loads(path.read_text())
        st.caption(f"Trained {meta['trained_at'][:16].replace('T', ' ')} on every week since "
                   f"2005 whose 1-week outcome is known (up to {meta['train_to']}); today's picks "
                   "use today's data.")
        status = M.MODEL_DIR.parent / "status.json"
        if config.CLOUD_TRAINING and status.exists():
            s_ = json.loads(status.read_text())
            st.info(f"☁️ Trained on GitHub every hour on all history (data to {s_['data_to']}). "
                    f"Last run {s_['updated'][:16].replace('T', ' ')} UTC: {s_['tuning_rounds']} "
                    f"self-tuning rounds, {s_['adopted']} improvement(s) adopted. "
                    "The Mac downloads new models automatically.")
        model = M.LongTermModel.load(M.MODEL_DIR)
        from stockpredictor.features.labels import label
        imp = model.importance().head(15)
        imp = (imp / imp.sum()).rename("share").reset_index().rename(columns={"index": "feature"})
        imp["feature"] = imp["feature"].map(label)
        st.subheader("What the model relies on most")
        st.altair_chart(C.hbars(imp, "feature", "share", ".0%"), width="stretch")
    else:
        st.info("No model trained yet.")
    if not config.CLOUD_TRAINING and st.button("Retrain now") and not need_data():
        from stockpredictor.features import longterm as F
        with st.spinner("Training…"):
            m = ctx()
            M.LongTermModel.train(M.add_labels(F.weekly_snapshots(m.feats), m.daily, m.indices)) \
                .save(M.MODEL_DIR)
        st.rerun()

    bt = M.MODEL_DIR / "backtest.json"
    st.subheader("Backtest (walk-forward, costs included)")
    if not bt.exists():
        st.caption("Run `python -m stockpredictor backtest` in the terminal (takes a few minutes).")
        model_intraday()
        training_history()
        return
    r = json.loads(bt.read_text())
    st.caption(f"{r['period']} · out-of-sample: each year predicted by a model trained only on "
               "earlier data. ⚠️ Uses today's Nifty 250 members for the whole period "
               "(survivorship bias), so absolute returns are overstated — compare against the "
               "equal-weight line, which has the same bias.")
    rows = {**{("Model (" + k + ")" if not k.startswith("momentum") else "Momentum only"): v
               for k, v in r["strategies"].items()},
            "Nifty 50": r["benchmarks"]["nifty50"],
            "Equal-weight Nifty 250": r["benchmarks"]["equal_weight_nifty100"]}
    st.dataframe(pd.DataFrame([{"Strategy": k, "Yearly return": v["cagr"], "Volatility": v["volatility"],
                                "Sharpe": v["sharpe"], "Worst fall": v["max_drawdown"]}
                               for k, v in rows.items()]), hide_index=True, width="stretch",
                 column_config={c_: st.column_config.NumberColumn(format="percent")
                                for c_ in ["Yearly return", "Volatility", "Worst fall"]} |
                 {"Sharpe": st.column_config.NumberColumn(format="%.2f")})
    k = st.columns(3)
    k[0].metric("Top-10 picks beat Nifty (3 mo)", f"{r['hit_rate_top10']:.0%}",
                f"{(r['hit_rate_top10'] - r['base_rate_all']) * 100:+.0f} pts vs all stocks")
    k[1].metric("Prediction quality (IC)", f"{r['ic_mean']:.3f}")
    k[2].metric("Weeks with positive IC", f"{r['ic_positive_share']:.0%}")
    curves = bt.with_suffix(".curves.csv")
    if curves.exists():
        cv = pd.read_csv(curves, index_col=0, parse_dates=True)
        names = {"model_weekly": "Model", "nifty50": "Nifty 50",
                 "equal_weight_nifty100": "Equal-weight Nifty 250", "momentum_only": "Momentum only"}
        cv = cv[[c_ for c_ in names if c_ in cv]].rename(columns=names)
        long = cv.reset_index(names="date").melt("date", var_name="series", value_name="value").dropna()
        st.altair_chart(C.lines(long, "date", "value", "series", ",.0f", "₹ (from ₹1 lakh)"),
                        width="stretch")
        with st.expander("Table view"):
            st.dataframe(cv.resample("YE").last().style.format("₹{:,.0f}"), width="stretch")
    model_intraday()
    training_history()


def training_history():
    """Continuous-training log: every retrain and weekly self-tuning result."""
    c = conn()
    runs = T.load_runs(c)
    st.divider()
    st.header("Continuous training")
    st.caption("Models retrain after every close on all history plus judged paper-trading "
               "predictions (wrong ones weigh more). Each weekend they self-tune: "
               "several settings are scored out-of-sample (each period predicted by a model "
               "trained only on earlier data) and new settings are adopted only if they beat the "
               "current ones. Prediction quality (IC) above 0 means better than random; "
               "0.03–0.08 is typical for real market prediction.")
    if runs.empty:
        st.caption("No training runs logged yet — they start with the live monitor, or run "
                   "`python -m stockpredictor improve --tune`.")
        return
    runs["m"] = runs["metrics"].map(json.loads)
    tunes = runs[runs["kind"] == "tune"].copy()
    if not tunes.empty:
        tunes["date"] = pd.to_datetime(tunes["version"])
        tunes["value"] = tunes["m"].map(lambda m: m.get("ic"))
        tunes["series"] = tunes["horizon"].map(MODEL_NAMES)
        if len(tunes) >= 2:
            st.caption("Prediction quality (IC) at each weekly tuning")
            st.altair_chart(C.lines(tunes[["date", "series", "value"]], "date", "value", "series",
                                    ".3f", "IC"), width="stretch")
        table = pd.DataFrame([{
            "When": r.version[:16].replace("T", " "), "Model": r.series,
            "IC": r.m.get("ic"), "Before": r.m.get("previous_ic"),
            "Result": "✅ new settings adopted" if r.m.get("adopted") else "kept current settings",
            "Momentum blend": r.m.get("params", {}).get("mom_weight")} for r in tunes.itertuples()])
        table["Momentum blend"] = pd.to_numeric(table["Momentum blend"], errors="coerce")
        st.dataframe(table.iloc[::-1], hide_index=True, width="stretch", column_config={
            "IC": st.column_config.NumberColumn(format="%.3f"),
            "Before": st.column_config.NumberColumn(format="%.3f"),
            "Momentum blend": st.column_config.NumberColumn(format="percent")})
    retrains = runs[runs["kind"] == "retrain"]
    if not retrains.empty:
        last = retrains.groupby("horizon").tail(1)
        st.caption("Last retrain: " + " · ".join(
            f"{MODEL_NAMES.get(r.horizon, r.horizon)} "
            f"{r.version[:16].replace('T', ' ')} (data to {r.train_to})" for r in last.itertuples()))


def strategy_race():
    st.subheader("🏁 Live strategy race")
    st.caption("Every day three variants also make paper predictions: the AI model, a 50/50 "
               "blend with plain momentum, and momentum only. Each is judged after 1 week like "
               "the real picks. Once every variant has 20+ judged days, the system switches to "
               "the one with the best live accuracy if it leads by 5+ points.")
    c = conn()
    race = E.strategy_race(c)
    current_w = M.current_params().get("mom_weight", 0.0)
    now_using = min(E.SHADOW_VARIANTS, key=lambda k: abs(E.SHADOW_VARIANTS[k] - current_w))
    started = c.execute("SELECT MIN(date), COUNT(DISTINCT date) FROM shadow_predictions").fetchone()
    if race.empty:
        msg = (f"Race started {started[0]}; {started[1]} prediction days so far. "
               if started[0] else "The race starts with the first daily decision. ")
        st.info(msg + f"First results one week later. Currently using: **{now_using}**.")
        return
    race["Using now"] = ["✅" if v == now_using else "" for v in race["variant"]]
    st.dataframe(race.rename(columns={
        "variant": "Strategy", "days": "Judged days", "predictions": "Predictions",
        "accuracy": "Live accuracy", "random": "Random picks",
        "avg_excess_up": "Top picks vs Nifty"}), hide_index=True, width="stretch",
        column_config={k_: st.column_config.NumberColumn(format="percent")
                       for k_ in ["Live accuracy", "Random picks", "Top picks vs Nifty"]})


def model_intraday():
    st.divider()
    st.header("Intraday model")
    path = MI.MODEL_DIR / "meta.json"
    if not path.exists():
        st.info("Not trained yet: `python -m stockpredictor intraday-backfill` (Angel One, one-time), "
                "then `python -m stockpredictor train-intraday`.")
        return
    meta = json.loads(path.read_text())
    st.caption(f"Trained {meta['trained_at'][:16].replace('T', ' ')} on {meta.get('train_days', '?')} "
               f"days up to {meta['train_to']}. Retrains daily, self-tunes weekly.")
    from stockpredictor.features.labels import label
    imp = MI.IntradayModel.load().importance().head(12)
    imp = (imp / imp.sum()).rename("share").reset_index().rename(columns={"index": "feature"})
    imp["feature"] = imp["feature"].map(label)
    st.altair_chart(C.hbars(imp, "feature", "share", ".0%"), width="stretch")
    bt = MI.MODEL_DIR / "backtest.json"
    if not bt.exists():
        st.caption("Run `python -m stockpredictor backtest-intraday` for backtest results.")
        return
    r = json.loads(bt.read_text())
    st.subheader("Intraday backtest (walk-forward, costs included)")
    st.caption(f"{r['period']} · IC {r['ic_mean']:.3f}, positive on {r['ic_positive_share']:.0%} of days")
    rows = [{"Strategy": k, "Days": m.get("trading_days", 0), "Return": m.get("return_pct"),
             "Sharpe": m.get("sharpe"), "Profitable days": m.get("win_days"),
             "Direction accuracy": m.get("direction_accuracy"), "Random": m.get("random_baseline")}
            for k, m in r["strategies"].items() if m.get("trading_days")]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        k_: st.column_config.NumberColumn(format="percent")
        for k_ in ["Return", "Profitable days", "Direction accuracy", "Random"]} | {
        "Sharpe": st.column_config.NumberColumn(format="%.2f")})


def page_settings():
    st.title("Settings")
    c = conn()
    rules, mode = D.get_rules(c)
    st.subheader("Long-term paper trading rules")
    with st.form("rules"):
        cols = st.columns(3)
        n_hold = cols[0].number_input("Stocks to hold", 3, 30, rules.n_hold)
        exit_rank = cols[1].number_input("Sell when rank falls below", 5, 100, rules.exit_rank)
        sl = cols[2].number_input("Stop-loss %", 3.0, 50.0, rules.stop_loss * 100, step=1.0)
        cols = st.columns(2)
        reb = cols[0].selectbox("Rank rebalancing", ["weekly", "daily"],
                                index=0 if mode == "weekly" else 1,
                                help="Backtest: weekly had better risk-adjusted returns and "
                                     "~1/3 lower costs. Stop-loss and news exits are always daily.")
        news_exit = cols[1].checkbox(
            "Sell on severe negative news", rules.news_exit,
            help="Bad news (2+ negative headlines in 3 days) always blocks buying. Severe news "
                 "(3+ very negative headlines) also sells a paper holding. Headlines that only "
                 "report price moves are ignored - the model already sees prices.")
        if st.form_submit_button("Save rules", type="primary"):
            for k_, v_ in {"lt_n_hold": n_hold, "lt_exit_rank": exit_rank, "lt_stop_loss": sl / 100,
                           "lt_rebalance": reb, "lt_news_exit": int(news_exit)}.items():
                put(c, k_, v_)
            st.success("Saved.")

    st.subheader("Intraday paper trading rules")
    irules, ienabled = PI.get_rules(c)
    with st.form("irules"):
        cols = st.columns(3)
        on = cols[0].checkbox("Intraday picks enabled", ienabled)
        nl = cols[1].number_input("Buy trades per day", 0, 10, irules.n_long)
        ns = cols[2].number_input("Sell (short) trades per day", 0, 10, irules.n_short)
        cols = st.columns(3)
        levels = list(BI.I.LEVELS)
        isl = cols[0].selectbox("Stop-loss %", levels, index=levels.index(BI.snap(irules.stop_loss)))
        itp = cols[1].selectbox("Target %", [0.0] + levels,
                                index=([0.0] + levels).index(BI.snap(irules.target)),
                                help="0 = no target, hold until 12:30")
        skip = cols[2].selectbox("Skip weak days", [0.0, 0.3, 0.5], index=[0.0, 0.3, 0.5].index(
            irules.skip_quantile) if irules.skip_quantile in (0.0, 0.3, 0.5) else 0,
            help="Skip days whose signal is weaker than this share of the last 60 days")
        st.caption("10 + 10 trades of ~₹5,000 each: charges are ~0.35% per round trip. Fewer, larger "
                   "positions cost less as a share (brokerage is capped at ₹20 an order).")
        if st.form_submit_button("Save intraday rules", type="primary"):
            for k_, v_ in {"id_enabled": int(on), "id_n_long": nl, "id_n_short": ns,
                           "id_stop_loss": isl, "id_target": itp, "id_skip_q": skip}.items():
                put(c, k_, v_)
            st.success("Saved.")

    st.subheader("My portfolio stop-loss")
    with st.form("pf_sl"):
        cols = st.columns(2)
        lt = cols[0].number_input("Long-term default stop-loss %", 1.0, 50.0,
                                  R.stop_loss_pct(c, "longterm", "") * 100, step=1.0)
        it = cols[1].number_input("Intraday default stop-loss %", 0.2, 20.0,
                                  R.stop_loss_pct(c, "intraday", "") * 100, step=0.1)
        if st.form_submit_button("Save"):
            R.set_stop_loss(c, "longterm", None, lt / 100)
            R.set_stop_loss(c, "intraday", None, it / 100)
            st.success("Saved.")

    st.subheader("Paper account")
    cap = st.number_input("Starting capital ₹", 10_000, 10_000_000,
                          int(SETTINGS.paper_capital_longterm), step=10_000)
    which = st.selectbox("Account", ["longterm", "intraday"],
                         format_func={"longterm": "Long-term (buy-only)", "intraday": "Intraday"}.get)
    if st.button("Reset paper account", help="Deletes this account's paper trades and starts again"):
        for t in ("paper_trades", "paper_orders", "paper_equity", "paper_accounts"):
            c.execute(f"DELETE FROM {t} WHERE horizon = ?", (which,))
        c.execute("DELETE FROM app_settings WHERE key IN ('lt_last_rebalance')")
        c.commit()
        E.ensure_account(c, cap, which)
        st.success(f"Paper account reset to {rupees(cap)}.")

    st.subheader("Connections")
    st.write(f"Angel One keys: {'✅ set' if SETTINGS.angel.is_complete else '— not set (Yahoo used for live prices)'}")


pages = [st.Page(page_picks, title="Long-term picks", icon="📈", default=True),
         st.Page(page_intraday_1230, title="Intraday — until 12:30", icon="⚡"),
         st.Page(page_intraday_close, title="Intraday — until close", icon="🔔"),
         st.Page(page_paper_longterm, title="Paper trading — Long-term", icon="🧪"),
         st.Page(page_paper_intraday, title="Paper trading — Intraday", icon="🧪"),
         st.Page(page_portfolio, title="My portfolio", icon="💼"),
         st.Page(page_accuracy, title="Accuracy", icon="🎯"),
         st.Page(page_model, title="Model", icon="🧠"),
         st.Page(page_settings, title="Settings", icon="⚙️")]
sidebar()
st.navigation(pages).run()
