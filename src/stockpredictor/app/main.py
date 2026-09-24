"""Stock Predictor dashboard.  Start with:  python -m stockpredictor app"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from stockpredictor import db, store
from stockpredictor.app import charts as C
from stockpredictor.config import load_settings
from stockpredictor.data import fundamentals as FUND
from stockpredictor.data import news as N
from stockpredictor.features.labels import reason_text
from stockpredictor.live.prices import in_market_hours, now_ist
from stockpredictor.models import longterm as M
from stockpredictor.paper import daily as D
from stockpredictor.paper import engine as E
from stockpredictor.portfolio import real as R

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

SETTINGS = load_settings()
STORE = store.DEFAULT_STORE_DIR
REFRESH = "60s" if in_market_hours() else None


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


def setting(c, key, default):
    row = c.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def put(c, key, value):
    c.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, str(value)))
    c.commit()


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
        st.subheader("🔔 Alerts")
        alerts = c.execute("SELECT ts, kind, message FROM alerts ORDER BY id DESC LIMIT 8").fetchall()
        if not alerts:
            st.caption("No alerts yet.")
        icons = {"stop-loss": "🛑", "negative-news": "📰", "decision": "🧭", "fill": "✅",
                 "info": "ℹ️"}
        for a in alerts:
            st.caption(f"{icons.get(a['kind'], '•')} {a['message']}")


# --- Pages -------------------------------------------------------------------------

def page_picks():
    st.title("Long-term picks")
    if need_data():
        return
    c, m = conn(), ctx()
    last = c.execute("SELECT MAX(date) FROM predictions WHERE horizon = 'longterm'").fetchone()[0]
    if not last:
        st.info("No predictions yet. Click below (or let the live monitor run after 5:15 PM).")
        if st.button("Run today's decision now", type="primary"):
            with st.spinner("Scoring stocks…"):
                D.run_daily(c, m, SETTINGS.paper_capital_longterm)
            st.rerun()
        return
    preds = pd.read_sql("SELECT * FROM predictions WHERE horizon = 'longterm' AND date = ?",
                        c, params=(last,))
    summary = N.news_summary(m.news, pd.Timestamp(now_ist())).set_index("symbol")
    fund = FUND.load_latest(STORE).set_index("symbol")
    st.caption(f"Official decision of **{pd.Timestamp(last):%d %b %Y}** (after close). "
               "Top = most likely to beat Nifty over 3 months; bottom = most likely to lag.")

    def table(direction):
        p = preds[preds["direction"] == direction].sort_values("rank")
        rows = []
        for r in p.itertuples():
            sent = summary["sent_mean_7d"].get(r.symbol)
            rows.append({
                "Stock": r.symbol,
                "Confidence": r.confidence, "Price": r.entry_price,
                "Why": reason_text(json.loads(r.reasons or "[]")),
                "News (7d)": "—" if pd.isna(sent) else
                ("🟢 positive" if sent > 0.2 else "🔴 negative" if sent < -0.2 else "⚪ neutral"),
                "P/E": fund["pe"].get(r.symbol), "ROE": fund["roe"].get(r.symbol),
            })
        df = pd.DataFrame(rows).rename(columns={"Why": "Signals (↑ raised score, ↓ lowered it)"})
        df[["P/E", "ROE"]] = df[["P/E", "ROE"]].apply(pd.to_numeric, errors="coerce")
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Price": st.column_config.NumberColumn(format="₹%.2f"),
            "P/E": st.column_config.NumberColumn(format="%.1f"),
            "ROE": st.column_config.NumberColumn(format="percent"),
            "Signals (↑ raised score, ↓ lowered it)": st.column_config.TextColumn(width="large")})

    st.subheader("▲ Top picks (buy candidates)")
    table("up")
    live_block(c)
    with st.expander("▼ Weakest stocks (avoid / may lag Nifty)"):
        table("down")

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
def live_block(c=None):
    c = c or conn()
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


def page_paper():
    st.title("Paper trading")
    tab_lt, tab_id = st.tabs(["Long-term", "Intraday"])
    with tab_id:
        st.info("Intraday paper trading arrives in Track B (after Angel One intraday data).")
    with tab_lt:
        if need_data():
            return
        paper_longterm()


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
    k[3].metric("Positions", f"{v['positions']}")

    pos = pd.read_sql("SELECT * FROM paper_trades WHERE horizon = 'longterm' AND status = 'open'", c)
    rules, mode = D.get_rules(c)
    st.subheader("Holdings")
    if pos.empty:
        st.caption("No open positions yet.")
    else:
        pos["Live price"] = pos["symbol"].map(prices)
        pos["Value"] = pos["qty"] * pos["Live price"]
        pos["P&L"] = (pos["Live price"] - pos["entry_price"]) * pos["qty"] - pos["costs"]
        pos["P&L %"] = pos["Live price"] / pos["entry_price"] - 1
        pos["Stop-loss"] = pos["entry_price"] * (1 - rules.stop_loss)
        pos["Held since"] = pd.to_datetime(pos["entry_time"]).dt.strftime("%d %b %Y")
        st.dataframe(pos[["symbol", "qty", "entry_price", "Live price", "Value", "P&L", "P&L %",
                          "Stop-loss", "Held since"]].rename(
            columns={"symbol": "Stock", "qty": "Qty", "entry_price": "Buy price"}),
            hide_index=True, width="stretch", column_config={
                c_: st.column_config.NumberColumn(format="₹%.2f")
                for c_ in ["Buy price", "Live price", "Value", "P&L", "Stop-loss"]} | {
                "P&L %": st.column_config.NumberColumn(format="percent")})

    pending = pd.read_sql("SELECT created, symbol, side, reason FROM paper_orders "
                          "WHERE horizon = 'longterm' AND status = 'pending'", c)
    if not pending.empty:
        st.caption("Queued orders (fill at the next market price):")
        st.dataframe(pending, hide_index=True, width="stretch")

    eq = pd.read_sql("SELECT date, equity FROM paper_equity WHERE horizon = 'longterm' ORDER BY date", c)
    if len(eq) >= 2:
        st.subheader("Value over time vs Nifty 50")
        m = ctx()
        eq["date"] = pd.to_datetime(eq["date"])
        nifty = m.indices[m.indices["symbol"] == "NIFTY50"].set_index("date")["close"]
        n = nifty.reindex(eq["date"]).ffill().bfill().values
        df = pd.concat([eq.assign(series="Model", value=eq["equity"]),
                        eq.assign(series="Nifty 50", value=n / n[0] * eq["equity"].iloc[0])])
        st.altair_chart(C.lines(df, "date", "value", "series", ",.0f", "₹"),
                        width="stretch")

    st.subheader("Closed trades — prediction vs reality")
    closed = pd.read_sql("SELECT * FROM paper_trades WHERE horizon = 'longterm' "
                         "AND status = 'closed' ORDER BY exit_time DESC", c)
    if closed.empty:
        st.caption("No closed trades yet.")
    else:
        closed["Return"] = closed["exit_price"] / closed["entry_price"] - 1
        closed["Result"] = ["✅ profit" if p > 0 else "❌ loss" for p in closed["pnl"]]
        st.dataframe(closed[["symbol", "entry_time", "entry_price", "exit_time", "exit_price",
                             "Return", "pnl", "exit_reason", "Result"]].rename(columns={
            "symbol": "Stock", "entry_time": "Bought", "entry_price": "Buy", "exit_time": "Sold",
            "exit_price": "Sell", "pnl": "P&L (after costs)", "exit_reason": "Why sold"}),
            hide_index=True, width="stretch", column_config={
                "Return": st.column_config.NumberColumn(format="percent"),
                "P&L (after costs)": st.column_config.NumberColumn(format="₹%.0f")})


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
    st.dataframe(t[["symbol", "qty", "avg_cost", "price", "value", "unrealized", "unrealized_pct",
                    "stop_loss", "Alert"]].rename(columns={
        "symbol": "Stock", "qty": "Qty", "avg_cost": "Avg cost", "price": "Price",
        "value": "Value", "unrealized": "P&L", "unrealized_pct": "P&L %", "stop_loss": "Stop-loss"}),
        hide_index=True, width="stretch", column_config={
            k_: st.column_config.NumberColumn(format="₹%.2f")
            for k_ in ["Avg cost", "Price", "Value", "P&L", "Stop-loss"]} | {
            "P&L %": st.column_config.NumberColumn(format="percent")})
    st.caption("Allocation")
    st.altair_chart(C.hbars(t.assign(w=t["weight"]), "symbol", "w", ".0%"),
                    width="stretch")


def page_accuracy():
    st.title("Accuracy")
    c = conn()
    tab_lt, tab_id = st.tabs(["Long-term", "Intraday"])
    with tab_id:
        st.info("Intraday accuracy arrives in Track B.")
    with tab_lt:
        a = E.accuracy(c)
        st.caption("A long-term prediction is judged after 3 months: a top pick is ✅ if it beat "
                   "Nifty 50, a weakest-stock pick is ✅ if it lagged Nifty 50.")
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
            st.caption(f"Up picks: {a['accuracy_up']:.0%} · Weakest picks: {a['accuracy_down']:.0%}"
                       + (f" · Paper trades profitable: {a['profitable_trades']:.0%} of "
                          f"{a['closed_trades']}" if a.get("closed_trades") else ""))
        else:
            k[2].metric("Accuracy", "—")
            st.info("First results appear ~3 months after the first prediction. Meanwhile the "
                    "table below shows how each open prediction is doing so far.")
        preds = pd.read_sql("SELECT * FROM predictions WHERE horizon = 'longterm' "
                            "ORDER BY date DESC, direction DESC, rank", c)
        if not preds.empty:
            preds["Result"] = ["⏳ open" if pd.isna(x) else "✅" if x else "❌"
                               for x in preds["correct"]]
            st.subheader("Prediction vs reality")
            st.dataframe(preds[["date", "symbol", "direction", "confidence", "entry_price",
                                "actual_exit", "actual_return", "Result"]].rename(columns={
                "date": "Date", "symbol": "Stock", "direction": "Predicted", "confidence": "Confidence",
                "entry_price": "Price then", "actual_exit": "Price now/at end",
                "actual_return": "vs Nifty"}), hide_index=True, width="stretch",
                column_config={"Confidence": st.column_config.NumberColumn(format="percent"),
                               "vs Nifty": st.column_config.NumberColumn(format="percent"),
                               "Price then": st.column_config.NumberColumn(format="₹%.2f"),
                               "Price now/at end": st.column_config.NumberColumn(format="₹%.2f")})


def page_model():
    st.title("Model")
    path = M.MODEL_DIR / "meta.json"
    if path.exists():
        meta = json.loads(path.read_text())
        st.caption(f"Trained {meta['trained_at'][:16].replace('T', ' ')} on data up to "
                   f"{meta['train_to']}. Retrains automatically every weekend.")
        model = M.LongTermModel.load(M.MODEL_DIR)
        from stockpredictor.features.labels import label
        imp = model.importance().head(15)
        imp = (imp / imp.sum()).rename("share").reset_index().rename(columns={"index": "feature"})
        imp["feature"] = imp["feature"].map(label)
        st.subheader("What the model relies on most")
        st.altair_chart(C.hbars(imp, "feature", "share", ".0%"), width="stretch")
    else:
        st.info("No model trained yet.")
    if st.button("Retrain now") and not need_data():
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
        return
    r = json.loads(bt.read_text())
    st.caption(f"{r['period']} · out-of-sample: each year predicted by a model trained only on "
               "earlier data. ⚠️ Uses today's Nifty 100 members for the whole period "
               "(survivorship bias), so absolute returns are overstated — compare against the "
               "equal-weight line, which has the same bias.")
    rows = {**{("Model (" + k + ")" if not k.startswith("momentum") else "Momentum only"): v
               for k, v in r["strategies"].items()},
            "Nifty 50": r["benchmarks"]["nifty50"],
            "Equal-weight Nifty 100": r["benchmarks"]["equal_weight_nifty100"]}
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
                 "equal_weight_nifty100": "Equal-weight Nifty 100", "momentum_only": "Momentum only"}
        cv = cv[[c_ for c_ in names if c_ in cv]].rename(columns=names)
        long = cv.reset_index(names="date").melt("date", var_name="series", value_name="value").dropna()
        st.altair_chart(C.lines(long, "date", "value", "series", ",.0f", "₹ (from ₹1 lakh)"),
                        width="stretch")
        with st.expander("Table view"):
            st.dataframe(cv.resample("YE").last().style.format("₹{:,.0f}"), width="stretch")


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
    if st.button("Reset long-term paper account", help="Deletes paper trades and starts again"):
        for t in ("paper_trades", "paper_orders", "paper_equity", "paper_accounts"):
            c.execute(f"DELETE FROM {t} WHERE horizon = 'longterm'")
        c.execute("DELETE FROM app_settings WHERE key IN ('lt_last_rebalance')")
        c.commit()
        E.ensure_account(c, cap)
        st.success(f"Paper account reset to {rupees(cap)}.")

    st.subheader("Connections")
    st.write(f"Angel One keys: {'✅ set' if SETTINGS.angel.is_complete else '— not set (Yahoo used for live prices)'}")
    st.write(f"Telegram: {'✅ set' if SETTINGS.telegram_bot_token else '— final phase'}")


pages = [st.Page(page_picks, title="Long-term picks", icon="📈", default=True),
         st.Page(page_paper, title="Paper trading", icon="🧪"),
         st.Page(page_portfolio, title="My portfolio", icon="💼"),
         st.Page(page_accuracy, title="Accuracy", icon="🎯"),
         st.Page(page_model, title="Model", icon="🧠"),
         st.Page(page_settings, title="Settings", icon="⚙️")]
sidebar()
st.navigation(pages).run()
