"""IPO analysis page: every open or coming NSE IPO with its terms, live subscription by investor
type with what each number means, a plain reading of the demand, and a final call."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockpredictor.app import ui as UI

MEANING = {
    "Institutions (QIB)": "Mutual funds, foreign funds, banks, insurers. They study the company "
                          "and bid mostly on the last day: the most informative number.",
    "Foreign institutions": "Share of the institutions' bids from foreign funds.",
    "Banks and insurers": "Share of the institutions' bids from banks and insurers.",
    "Mutual funds": "Share of the institutions' bids from Indian mutual funds.",
    "Other institutions": "Share of the institutions' bids from others (AIFs, pension funds).",
    "HNI / NII (above ₹2 lakh)": "Wealthy individuals and companies. Often borrow to apply and "
                                 "sell on listing day: high demand means they expect a listing pop.",
    "Big HNI (above ₹10 lakh)": "Bids above ₹10 lakh; the most leveraged, most listing-focused money.",
    "Small HNI (₹2–10 lakh)": "Bids of ₹2–10 lakh; allotted by lottery of minimum lots.",
    "Retail (up to ₹2 lakh)": "Individuals like you. Above 1× the allotment is a lottery of one lot "
                              "per applicant.",
    "Employees": "The company's staff quota (often at a discount).",
    "Shareholders of the parent": "Quota for shareholders of the listed parent company: owning one "
                                  "of its shares before the prospectus date lets you apply here too.",
    "Policyholders": "Quota for policyholders (insurance companies' IPOs).",
    "Total": "All categories together.",
}


def _f(x) -> float:
    """A number, or 0 for None / NaN / text."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(x) else x


def _band(x: float | None, cuts: list[tuple[float, str]]) -> str:
    if x is None or pd.isna(x):
        return "no bids yet"
    for lo, word in cuts:
        if x >= lo:
            return word
    return cuts[-1][1]


QIB_WORDS = [(50, "very strong"), (10, "strong"), (3, "moderate"), (1, "weak"),
             (0, "under-subscribed")]


def reading(row: dict, bids: pd.DataFrame, nifty_20d: float | None) -> list[tuple[str, str]]:
    """Plain-language reading of the demand: [(tone 'good'|'bad'|'info', text)]."""
    t = dict(zip(bids["key"], bids["times"])) if not bids.empty else {}
    q, nii, big, small, ret = (t.get(k) for k in ("1", "2", "2.1", "2.2", "3"))
    out = []
    if q is not None and pd.notna(q):
        w = _band(q, QIB_WORDS)
        out.append(("good" if q >= 10 else "bad" if q < 3 else "info",
                    f"Institutions: {q:.1f}× ({w}{', so far' if row.get('_open') else ''}). "
                    "In this system's data, issues that closed with QIB demand "
                    + ("above 50× listed up about 9 times in 10." if q >= 50 else
                       "of 10–50× listed up about 6 times in 10." if q >= 10 else
                       "of 3–10× listed up only about 4 times in 10." if q >= 3 else
                       "under 3× listed up only about 1 time in 4.")))
    qbid = bids[bids["key"].isin(["1(a)", "1(b)", "1(c)", "1(d)"])] if not bids.empty else bids
    if not qbid.empty and qbid["bid"].sum() > 0:
        share = qbid.set_index("category")["bid"] / qbid["bid"].sum()
        top = share.idxmax().strip()
        out.append(("info", f"Most institution bids come from {top.lower()} ({share.max():.0%})."
                    + (" Indian mutual funds research their bids closely; their backing is a "
                       "good sign." if top == "Mutual funds" and share.max() >= 0.4 else "")))
    if big is not None and pd.notna(big):
        if big >= 50:
            out.append(("info", f"Big HNIs (above ₹10 lakh): {big:.0f}×. Very heavy, often "
                        "borrowed, money expecting a listing pop. It supports the listing day but "
                        "leaves quickly, so the price can fall soon after."))
        elif big >= 10:
            out.append(("good", f"Big HNIs: {big:.0f}×. Strong demand from wealthy investors."))
        elif big < 1 and q is not None and pd.notna(q) and q >= 1:
            out.append(("bad", f"Big HNIs: {big:.1f}×. Wealthy investors are not chasing it: "
                        "they don't expect much of a listing gain."))
    elif nii is not None and pd.notna(nii):
        out.append(("info", f"HNIs: {nii:.1f}× ({_band(nii, QIB_WORDS)})."))
    if ret is not None and pd.notna(ret):
        if ret > 1:
            out.append(("info", f"Retail: {ret:.1f}×. Your chance of getting one lot is roughly "
                        f"1 in {max(1, round(ret)):.0f} (allotment is a lottery of one lot per "
                        "applicant, so apply for one lot only)."))
        else:
            out.append(("info", f"Retail: {ret:.2f}×. Every retail applicant should get shares."))
        if ret >= 10 and q is not None and pd.notna(q) and q < 3:
            out.append(("bad", "Retail demand is high but institutions are not convinced: "
                        "retail-led issues have a poor listing record."))
    fresh, ofs = _f(row.get("fresh_shares")), _f(row.get("ofs_shares"))
    if fresh + ofs > 0:
        o = ofs / (fresh + ofs)
        if o >= 0.75:
            out.append(("bad" if o == 1 else "info",
                        f"{o:.0%} of the issue is an offer for sale: existing owners are selling "
                        "and the company gets little or no new money. Ask why they are selling now."))
        elif o <= 0.25:
            out.append(("good", f"{1 - o:.0%} of the issue is fresh shares: the money goes into the "
                        "company (growth or debt repayment). Check the 'objects of the issue'."))
        else:
            out.append(("info", f"{1 - o:.0%} fresh shares, {o:.0%} existing owners selling."))
    if nifty_20d is not None and pd.notna(nifty_20d):
        out.append(("bad" if nifty_20d < -0.03 else "good" if nifty_20d > 0.03 else "info",
                    f"Market mood: Nifty {nifty_20d:+.1%} over 20 days. "
                    + ("Falling markets hurt listings." if nifty_20d < -0.03 else
                       "Rising markets help listings." if nifty_20d > 0.03 else "Neutral.")))
    return out


FINAL = {
    "Apply": ("Apply", "Apply for one lot at the cut-off price. Most of an IPO's gain usually "
              "comes on listing day and IPOs on average lag the market in the years after, so "
              "plan to sell on listing day unless you would buy the business at that price."),
    "Neutral": ("Optional", "The listing is close to a coin toss. Apply only if you like the "
                "business at this price and would hold it; not for a quick gain."),
    "Avoid": ("Skip", "Demand and history point to a weak listing. Skip it, or wait to buy after "
              "listing if the price falls and the business still appeals to you."),
    "Wait": ("Wait for the last day", "Institutions bid mostly on the final day; the call is made "
             "from the closing-day numbers (after about 5 pm)."),
}


def render(issues: pd.DataFrame, err: str, model, indices: pd.DataFrame, today: pd.Timestamp) -> None:
    from stockpredictor.models import ipo as MIPO

    st.title("IPO analysis")
    st.caption("Every mainboard IPO open or coming on NSE: its terms, live demand by investor type "
               "and what each number means, a reading of the demand, and a final call from a "
               "model trained on how past NSE IPOs listed. Refreshes every 15 minutes.")
    with st.expander("How to read this page"):
        st.markdown("""
- **Subscribed (×)**: bids ÷ shares on offer in that category. 5× means five times more shares
  were asked for than are available.
- **Institutions (QIB)** are the best-informed bidders and bid mostly on the last day. Their
  final demand is the strongest single signal for listing day.
- **HNI / NII** are wealthy individuals and companies, often using borrowed money. Very high
  HNI demand means they expect a listing pop. It helps the listing day but this money sells
  quickly.
- **Retail** is you. Above 1× the allotment is a lottery: one lot per applicant, so apply for
  one lot only. Applying from family members' accounts raises your chances legally.
- **Fresh issue vs offer for sale**: a fresh issue raises money for the company; an offer for
  sale lets existing owners cash out.
- **Chance of listing up**: the model's probability that the first day closes at least 1% above
  the issue price, learned from past NSE mainboard IPOs (subscription by category, issue size,
  market mood).
- **Final call**: *Apply*, *Optional*, *Skip*, or *Wait for the last day* while institutions
  are still to bid.
- The grey-market premium is not used: it is unofficial and easy to manipulate.
""")
    if err:
        st.warning(f"NSE did not answer ({err[:120]}). Try again in a few minutes.")
    if issues.empty and not err:
        st.info("No IPO is open or announced on NSE right now.")
    n20 = None
    if indices is not None and not indices.empty:
        n = indices[indices["symbol"] == "NIFTY50"].sort_values("date")["close"]
        n20 = float(n.iloc[-1] / n.iloc[-21] - 1) if len(n) > 21 else None
    for _, row in issues.iterrows():
        card(row, model, indices, today, n20, MIPO)
    if model is not None:
        reliability(model.info)


def card(row: pd.Series, model, indices, today, n20, MIPO) -> None:
    bids = row.get("bids")
    bids = bids if isinstance(bids, pd.DataFrame) else pd.DataFrame(columns=["key", "category", "times", "offered", "bid"])
    with st.container(border=True):
        st.markdown(f"### {row['company'] or row['symbol']}")
        sme = row["segment"] != "mainboard"
        bits = [row["symbol"], "SME" if sme else "Mainboard"]
        if pd.notna(row.get("price_high")):
            bits.append(f"price band ₹{row['price_low']:g}–{row['price_high']:g}")
        if pd.notna(row.get("start")):
            bits.append(f"bidding {row['start']:%d %b} – {row['end']:%d %b}")
        st.caption(" · ".join(bits))
        if pd.notna(row.get("start")) and pd.notna(row.get("end")):
            if row["start"] > today:
                status = f"Opens {row['start']:%A %d %b}"
            elif row["end"] == today:
                status = "Closes today (bids until 5 pm)"
            elif row["end"] > today:
                days = int(np.busday_count(row["start"].date(), today.date())) + 1
                total = int(np.busday_count(row["start"].date(), row["end"].date())) + 1
                status = f"Day {days} of {total}"
            else:
                status = "Bidding closed; allotment and listing next"
        else:
            status = ""
        lot, hi = _f(row.get("lot")), _f(row.get("price_high"))
        size_sh = _f(row.get("fresh_shares")) + _f(row.get("ofs_shares"))
        tiles = [UI.card("Status", status or "—", "", 0.0)]
        if lot and hi:
            tiles.append(UI.card("Minimum investment", f"₹{lot * hi:,.0f}", f"1 lot = {lot:,.0f} shares "
                                 f"at ₹{hi:g}", 0.0))
        if size_sh and hi:
            ofs = _f(row.get("ofs_shares")) / size_sh
            tiles.append(UI.card("Issue size", f"₹{size_sh * hi / 1e7:,.0f} cr",
                                 f"{1 - ofs:.0%} fresh, {ofs:.0%} offer for sale", 1 - ofs))
        t = dict(zip(bids["key"], bids["times"])) if not bids.empty else {}
        if _f(t.get("total")) > 0:
            tiles.append(UI.card("Subscribed so far", f"{t['total']:.2f}×",
                                 f"institutions {_f(t.get('1')):.1f}×", min(1.0, t["total"] / 10)))
        UI.cards(tiles)
        if sme:
            st.caption("SME issue: not scored. SME shares trade thinly after listing, in large "
                       "lots, with little history to learn from.")
        if not bids.empty and bids["offered"].fillna(0).sum() > 0:
            qbids = bids[bids["key"].isin(["1(a)", "1(b)", "1(c)", "1(d)"])]
            qsum = qbids["bid"].sum()
            show = []
            for r in bids.itertuples():
                name = r.category.strip()
                if r.key in ("1(a)", "1(b)", "1(c)", "1(d)"):
                    val = f"{r.bid / qsum:.0%} of QIB bids" if qsum else ""
                else:
                    val = f"{r.times:.2f}×" if pd.notna(r.times) else "no bids yet"
                show.append({"Investor type": ("   " if r.category.startswith(" ") else "") + name,
                             "Subscribed": val, "What it means": MEANING.get(name, "")})
            st.dataframe(pd.DataFrame(show), hide_index=True, width="stretch",
                         column_config={"What it means": st.column_config.TextColumn(width="large")})
        elif pd.notna(row.get("start")) and row["start"] > today:
            st.caption("No bids yet: subscription appears once bidding opens.")
        still = pd.notna(row.get("end")) and row["end"] >= today
        notes = reading({**row.to_dict(), "_open": still}, bids, n20)
        if notes:
            st.markdown("**Reading the demand**")
            icon = {"good": ":material/thumb_up:", "bad": ":material/warning:", "info": ":material/info:"}
            for tone, text in notes:
                st.markdown(f"{icon[tone]} {text}")
        if row.get("managers"):
            st.caption(f"Lead managers: {row['managers']}")
        if sme or model is None:
            if model is None and not sme:
                st.caption("The IPO model trains in tonight's GitHub run.")
            return
        if pd.notna(row.get("start")) and row["start"] > today:
            st.info("**Final call after bidding opens.** Check the prospectus meanwhile (see "
                    "*How to read this page*).")
            return
        x = pd.DataFrame([{**row.to_dict(), "qib_x": t.get("1"), "nii_x": t.get("2"),
                           "rii_x": t.get("3"), "total_x": t.get("total"),
                           "shares_offered": bids.loc[bids["key"] == "total", "offered"].max()
                           if not bids.empty else row.get("shares_offered")}]).assign(issue_price=None)
        x = MIPO.add_inputs(x, indices if indices is not None else
                            pd.DataFrame(columns=["symbol", "date", "close"]))
        x["recent_listings"] = model.info.get("recent_listings_now")
        label, p, why = MIPO.verdict(model, x.iloc[0], closes_today=row["end"] == today,
                                     still_open=row["end"] >= today)
        lean = ("Apply" if p >= MIPO.APPLY_AT else "Avoid" if p < MIPO.AVOID_AT else "Neutral")
        title, text = FINAL[label]
        show = {"Apply": st.success, "Avoid": st.error, "Neutral": st.warning, "Wait": st.info}[label]
        lean_txt = (f" On today's numbers it leans to **{FINAL[lean][0].lower()}**." if label == "Wait"
                    else "")
        show(f"**Final call: {title}.** Chance of listing above the issue price: **{p:.0%}**. "
             f"{text}{lean_txt}")
        with st.expander("Why"):
            for w in why:
                st.markdown(f"- {w[0].upper() + w[1:]}")


def reliability(info: dict) -> None:
    st.subheader("How reliable is the final call?")
    wf = info.get("walk_forward") or {}
    if wf:
        tiles = [UI.card("All IPOs", f"{wf['all_listed_up']:.0%} listed up",
                         f"median listing day {wf['all_median_gain']:+.0%}", wf["all_listed_up"])]
        if wf.get("apply_n"):
            tiles.append(UI.card(f"'Apply' calls ({wf['apply_n']})", f"{wf['apply_listed_up']:.0%} listed up",
                                 f"median {wf['apply_median_gain']:+.0%}", wf["apply_listed_up"], "up"))
        if wf.get("avoid_n"):
            tiles.append(UI.card(f"'Skip' calls ({wf['avoid_n']})", f"{wf['avoid_listed_up']:.0%} listed up",
                                 f"median {wf['avoid_median_gain']:+.0%}", wf["avoid_listed_up"], "down"))
        UI.cards(tiles)
        st.caption(f"Each year's IPOs were judged by a model trained only on earlier years "
                   f"({info['ipos']} mainboard IPOs since {info['since'][:4]}).")
    qb = pd.DataFrame(info.get("qib_buckets", []))
    if not qb.empty:
        qb["QIB demand"] = [f"{lo:g}–{'∞' if hi > 1e9 else f'{hi:g}'}×" for lo, hi in zip(qb["lo"], qb["hi"])]
        st.dataframe(qb[["QIB demand", "n", "listed_up", "median_gain"]].rename(columns={
            "n": "IPOs", "listed_up": "Listed up", "median_gain": "Median listing-day gain"}),
            hide_index=True, width="stretch", column_config={
                "Listed up": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Median listing-day gain": st.column_config.NumberColumn(format="percent")})
    rec = pd.DataFrame(info.get("recent", []))
    if not rec.empty:
        st.subheader("Recent listings")
        rec = rec.iloc[::-1].copy()
        if "listing_date" in rec:
            rec["listing_date"] = pd.to_datetime(rec["listing_date"]).dt.strftime("%d %b %Y")
        keep = [c for c in ("company", "listing_date", "issue_price", "qib_x", "total_x",
                            "gain_open", "gain_close") if c in rec]
        st.dataframe(rec[keep], hide_index=True, width="stretch", column_config={
            "company": "Company", "listing_date": "Listed",
            "issue_price": st.column_config.NumberColumn("Issue price", format="₹%.0f"),
            "qib_x": st.column_config.NumberColumn("QIB demand", format="%.1f×"),
            "total_x": st.column_config.NumberColumn("Total demand", format="%.1f×"),
            "gain_open": st.column_config.NumberColumn("Listing open", format="percent"),
            "gain_close": st.column_config.NumberColumn("Listing-day close", format="percent")})
