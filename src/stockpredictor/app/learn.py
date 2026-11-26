"""Learn: a complete guide to the Indian market, from how an order is filled to how this
system decides, in six parts. Rules and figures are as of late 2026; tax and SEBI rules
change, so check the current circulars before acting on them."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from stockpredictor.app import ui as UI

PARTS = ["Market basics", "Reading a company", "Reading the market", "Trading and risk",
         "Futures and options", "Algos and this system"]


def render() -> None:
    st.title("Learn")
    st.caption("A complete guide in six parts: how the market works, how to judge a company and "
               "an IPO, how news and market moves affect prices, how to manage risk, how futures "
               "and options work, and how algorithmic trading (including this system) is done "
               "properly. Read it in order the first time; each part stands on its own later.")
    part = st.segmented_control("Part", PARTS, default=PARTS[0], label_visibility="collapsed")
    {PARTS[0]: basics, PARTS[1]: company, PARTS[2]: market, PARTS[3]: risk,
     PARTS[4]: derivatives, PARTS[5]: algos}[part or PARTS[0]]()


# --- 1. Market basics ---------------------------------------------------------------------

def basics() -> None:
    a, b, c, d = st.tabs(["How the market works", "Orders and sessions", "Indices and sectors",
                          "Costs and tax"])
    with a:
        st.markdown("""
### How the market works

A share is a small piece of ownership in a company. Shares trade on two exchanges, the **NSE**
and the **BSE**. Almost all trading in India happens on the NSE.

- **SEBI** is the regulator. It writes the rules for companies, brokers and traders.
- **Brokers** (Angel One, Zerodha, Groww and others) send your orders to the exchange.
- **Depositories** (NSDL and CDSL) hold your shares in your demat account. Your broker never
  holds them.
- **Clearing corporations** guarantee that both sides of a trade are honoured.
- **Settlement is T+1:** shares and money change hands the next working day.

**Who moves prices.** Prices move when buyers or sellers are more eager than the other side:
- Foreign institutions (FPIs/FIIs)
- Domestic institutions (DIIs: mutual funds, insurers, pension funds)
- Promoters (the founders and owning groups)
- Retail traders like you

FPI and DII flows are published every evening (the *fiidii* data this system downloads).
When FPIs sell heavily for weeks, large-caps usually fall even if the companies are doing well.

**Why prices move at all.** Over years, a share follows the company's profits and cash flow.
Over days, it follows news, flows and mood. Over minutes, it is mostly noise. This is why
long-term investing rests on business quality, while intraday trading needs a real, measured
edge and very low costs.
""")
    with b:
        st.markdown("""
### Orders and the trading day

| Time (IST) | What happens |
|---|---|
| 9:00 – 9:08 | **Pre-open**: orders are collected, no trades |
| 9:08 – 9:15 | The opening price is found by auction |
| 9:15 – 15:30 | **Normal market**: continuous trading |
| 15:30 – 15:40 | Closing price = average of the last 30 minutes |
| 15:40 – 16:00 | Post-close session at the closing price |

**Order types**
- **Market**: buy or sell now at the best available price. Fast, but you can pay a poor
  price in thin stocks.
- **Limit**: only at your price or better. It may not fill.
- **Stop-loss (SL / SL-M)**: becomes an order once the price touches your trigger. Use it to
  cap a loss.
- **GTT (good till triggered)**: waits for up to a year for its trigger. Useful for
  long-term targets and stops.

**Product types**
- **CNC (delivery)**: you own the shares and can hold them for years.
- **MIS (intraday)**: closed the same day, by you or by the broker before the close.
- **NRML**: futures and options carried overnight.

**Circuit limits.** Most stocks have a daily price band of 2%, 5%, 10% or 20%. At the band,
trading continues only at that price, and you may not be able to exit. The whole market halts
for a while if the Nifty or Sensex moves 10%, 15% or 20% in a day.
""")
    with c:
        st.markdown("""
### Indices and sectors

An index is a basket that tells you how a part of the market is doing.
- **Nifty 50**: the 50 largest NSE companies, weighted by free-float market value. The
  market's headline number.
- **Sensex**: the 30 largest BSE companies.
- **Nifty Bank, Nifty IT, Nifty Pharma, Nifty Auto, Nifty Metal**: sectors move together
  because they share costs, customers and regulation.
- **Nifty Midcap 150 / Smallcap 250**: smaller companies. They rise faster in good years
  and fall harder in bad ones.
- **India VIX**: the volatility the options market expects over the next 30 days, in
  yearly %. Around 11–14 is calm, above 20 is fearful.

**Why sectors matter for single stocks.** Roughly half of a typical stock's daily move is the
market and its sector, not the company. If steel prices fall, a well-run steel company
usually falls too. That is why this system's news reading includes sector and policy news
(steel duties, RBI rates, USFDA, telecom tariffs), not only company headlines.
""")
    with d:
        st.markdown("""
### Costs and tax

**Costs per trade (typical discount broker)**
- **Brokerage**: ₹0 for delivery at many brokers; ₹20 or 0.03% (whichever is lower) for
  intraday and F&O.
- **STT (securities transaction tax)**:
  - Delivery: 0.1% on both buy and sell
  - Intraday: 0.025% on the sell side
  - Futures: 0.02% on the sell side
  - Options: 0.1% of the premium on the sell side
- **Smaller charges**: exchange charges, SEBI fee, 18% GST on brokerage and charges, and
  stamp duty on buys.
- **Slippage**: the gap between the price you saw and the price you got. It is often larger
  than all the charges above.

A round trip in intraday cash costs about 0.06–0.1% before slippage. With a ₹5 minimum
brokerage, small orders cost much more. This system found that ₹1,000 trades cost 1.32% per
round trip, which is why each paper book trades at most 3–5 stocks a side.

**Tax (resident individuals, rules from the July 2024 budget)**
- **Short-term capital gains** (listed shares held 12 months or less): 20%.
- **Long-term capital gains** (held over 12 months): 12.5% on gains above ₹1.25 lakh a year.
- **Intraday profits**: speculative business income, taxed at your slab rate.
- **F&O profits**: non-speculative business income, taxed at your slab rate. Losses can be
  carried forward; turnover may require a tax audit.
- **Dividends**: taxed at your slab rate.

Rules change in budgets: check the current year's rules or ask a tax adviser.
""")


# --- 2. Reading a company -----------------------------------------------------------------

def company() -> None:
    a, b, c, d, e = st.tabs(["Financial statements", "Valuation", "Quality and red flags",
                             "Quarterly results", "Analysing an IPO"])
    with a:
        st.markdown("""
### The three financial statements

**Profit and loss (P&L).** What the company earned over a period:
- Revenue (sales)
- Minus costs, giving EBITDA (operating profit before depreciation)
- Minus depreciation, interest and tax, giving net profit

Look at the trend over 5 years, not one quarter.

**Balance sheet.** What the company owns and owes on one date:
- **Assets**: plants, cash, receivables, inventory
- **Liabilities**: debt, payables
- **Equity**: what belongs to shareholders

**Cash flow statement.** Where the cash actually went:
- **Operating cash flow** should be close to net profit over time. Profit without cash is
  the oldest warning sign in accounting.
- **Free cash flow** = operating cash flow minus capital spending. This is what can pay
  dividends, buy back shares or repay debt.

**Ratios worth knowing**

| Ratio | What it tells you | Rough guide |
|---|---|---|
| Revenue growth | Is the business getting bigger? | Compare with nominal GDP (~10–11%) |
| Operating margin | How much of each ₹100 of sales is operating profit | Compare within the sector |
| ROE | Profit per ₹ of shareholders' money | Above 15% is good |
| ROCE | Profit per ₹ of all capital, debt included | Above the cost of debt; above 15% good |
| Debt / equity | How much the company leans on borrowing | Below 1 for most non-financials |
| Interest coverage | Operating profit ÷ interest | Below 3 is fragile |
| Receivable days | How long customers take to pay | Rising every year is a warning |

Banks and NBFCs are read differently. Look at net interest margin, gross and net NPAs,
provisions, capital adequacy and deposit growth.
""")
    with b:
        st.markdown("""
### Valuation: what the price already assumes

- **P/E (price ÷ earnings per share)**: how many years of today's profit you pay for.
  A P/E of 60 needs years of fast growth just to stand still.
- **EV/EBITDA**: like P/E, but counts debt. Better for comparing companies with different
  borrowing.
- **P/B (price ÷ book value)**: used for banks, NBFCs and asset-heavy businesses.
- **PEG (P/E ÷ growth %)**: a P/E of 30 with 30% growth (PEG 1) can be cheaper than a P/E of
  15 with 5% growth.
- **Dividend yield and free-cash-flow yield**: the cash you are paid now.

**How to use them**
- Compare a company with its own 5–10 year history and with its closest peers, never in
  isolation.
- A stock that is cheap on every measure is often cheap for a reason: falling profits,
  debt, governance or a dying product. Check why before buying.
- Expensive stocks can stay expensive for years if growth continues. They fall hardest when
  growth disappoints.

**What the evidence says.** Across decades and countries, three things have beaten the
market on average over long periods, with long dry spells:
- **Momentum**: stocks that rose most over the last 12 months, excluding the last month
  (Jegadeesh and Titman, 1993).
- **Quality**: profitable, low-debt companies.
- **Value**: low price relative to fundamentals.

India's Nifty 200 Momentum 30 and Nifty 100 Low Volatility 30 indices are built on these
ideas. This system's long-term model uses momentum, quality and value features together.
""")
    with c:
        st.markdown("""
### Quality and red flags

**Good signs**
- Steady ROCE above 15% through a full business cycle
- Operating cash flow close to profit
- Low or falling debt
- Promoters holding steady or buying
- Clean, unchanged auditors
- Few related-party transactions
- Dividends or buybacks paid from free cash flow

**Red flags**
- **Pledged promoter shares**: if the price falls, lenders sell, and the price falls further.
- **The auditor resigns** or qualifies the accounts.
- **Receivables or inventory grow much faster than sales.**
- **Frequent equity raises**, warrants to promoters or preferential allotments.
- **Big related-party deals**: money moving between the company and its promoters' other
  businesses.
- **Large contingent liabilities** (court cases, guarantees) in the notes to the accounts.
- **Promoters selling steadily** while management talks up the future.
- **A sudden change of business** into whatever is fashionable.

Where to look: the annual report (especially the notes and auditor's report), shareholding
patterns (quarterly on NSE), credit-rating reports, and the exchange announcements page.
""")
    with d:
        st.markdown("""
### Quarterly results

Companies report within 45 days of each quarter's end (60 days for the March quarter).
**What moves the price is the surprise, not the number**: profit up 20% can send a stock down
if the market expected 30%.

Read, in order:
1. **Revenue growth** and its mix (price vs volume, segments)
2. **Margins**: rising costs usually show here first
3. **One-offs**: exceptional items, tax write-backs, other income
4. **Management commentary and guidance** in the investor presentation or conference call
5. **Order book** for capital-goods and defence companies; **asset quality** for banks

**After the results.** Stocks that surprise positively tend to keep drifting in the same
direction for weeks (post-earnings announcement drift, Bernard and Thomas 1989). Results-day
moves are large and hard to predict, so the intraday model's rules can skip stocks reporting
that day. This system's long-term model includes the price reaction to the latest results.
""")
    with e:
        st.markdown("""
### Analysing an IPO, step by step

1. **Read the red herring prospectus (RHP).** It covers the business, the risk factors, three
   years of financials and the "objects of the issue". It is on SEBI's and NSE's sites.
2. **Fresh issue or offer for sale?** A fresh issue raises money for the company. An offer for
   sale (OFS) is existing owners selling. A large OFS from private-equity holders is a reason
   to look harder at the price.
3. **Price it against listed peers.** At the top of the price band, compare P/E, EV/EBITDA,
   price-to-book, return on equity and growth. Paying more than peers needs faster growth.
4. **Check ownership after listing.** Look at promoter holding, any pledged shares, and
   lock-ins.
5. **Read the anchor book.** Large institutions buy the day before the issue opens. Their
   names and the 30/90-day lock-in tell you who is committed.
6. **Watch subscription by category, especially on the last day.** Institutions (QIBs) bid
   mostly on the final day, and their demand is the strongest single signal. In this system's
   data (NSE mainboard IPOs since late 2023), the share that listed above the issue price was:
   - QIB demand under 3×: 26% (median listing −5%)
   - QIB 10–50×: 57% (median +5%)
   - QIB 50–100×: 86% (median +19%)
   - QIB over 100×: 95% (median +29%)
7. **Treat the grey-market premium as gossip.** It is unofficial and easy to move.
8. **Know your allotment odds.** Heavy retail oversubscription makes allotment a lottery.
9. **Decide your listing-day plan in advance.** IPOs list worse in falling markets.

The **IPO analysis** page scores each IPO with a model trained on past listings (subscription
by category, issue size, market mood). Tested year by year on IPOs it had not seen, its
"Apply" calls listed up 88% of the time (median +23%), against 65% for all IPOs.

**Red flags**
- QIB demand under 3×
- A mostly-OFS issue priced above its peers
- Pledged promoter shares
- A sector in decline
""")


# --- 3. Reading the market ----------------------------------------------------------------

def market() -> None:
    a, b, c = st.tabs(["News that moves stocks", "Market moves", "Charts, honestly"])
    with a:
        st.markdown("""
### News that moves stocks, and news that doesn't

**Company events that move prices**

| Event | Usual effect |
|---|---|
| Results surprise, guidance change | Large; often keeps drifting for weeks |
| Big orders and contracts (vs annual revenue) | Up, more for smaller companies |
| Regulatory approvals or warnings (USFDA, SEBI, RBI, courts) | Sharp, either way |
| Mergers, acquisitions, stake sales | The buyer often falls, the target rises |
| Promoter buying, or pledging / selling | Buying is confident; pledging is a warning |
| Block and bulk deals | Often a dip on the day if at a discount |
| Management changes, resignations, frauds | Sharp falls on bad surprises |
| Buybacks, bonus issues, splits, dividends | Small and mostly known in advance |
| Sector policy (duties, tariffs, rates, prices) | Moves the whole sector together |

**Noise this system drops**
- Tips and targets ("stocks to buy", brokerage upgrades and downgrades)
- Price reports ("shares rise 3%"): the price is already known
- Generated ratings and valuation pages
- Lists of stocks, chart and open-interest commentary
- Job adverts and admin filings

**How the system reads news**
- **Fetching**: GitHub fetches Google News for every stock and for 15 sector and policy
  themes at 8:30, 10:00, 12:00, 14:00 and 17:00 on weekdays.
- **Tone**: FinBERT, a finance language model, gives each headline a tone from −1 to +1.
- **Weight**: a second model, trained on more than 40,000 past headlines, learned which kinds
  of headlines came with unusual moves in their stock. Its out-of-sample correlation is 0.20,
  against 0.15 for tone alone.
- **Use in the picks**:
  - Stocks with strong negative news are not bought.
  - Among the day's 25 buy and 25 sell candidates, news against a trade removes it, and news
    for it adds a point.
  - The *News drift* book trades fresh news directly.

**The evidence.** Price moves that come with news tend to continue; moves without news tend
to reverse (Chan 2003; Savor 2012). Negative words in news predict lower returns, mostly
within a day or two (Tetlock 2007). The *Gap fade without news* book tests the reversal half.
""")
    with b:
        st.markdown("""
### Reading the market's mood

- **Breadth**: how many stocks rose versus fell. An index up on narrow breadth (a few large
  stocks) is weaker than one up on broad breadth.
- **Sector rotation**: money moves between sectors (IT to banks, defensives to cyclicals).
  The sector heatmap shows this better than the index does.
- **FPI and DII flows**: heavy, persistent FPI selling weighs on large caps; DIIs (fed by
  monthly SIPs) often buy those dips.
- **India VIX**: rising VIX means fear and larger moves. Option premiums rise with it.
- **Global cues before 9:15**: the US close, Asian markets, crude oil, USD/INR and US bond
  yields. This system's intraday model uses the overnight US, Japan and Hong Kong moves,
  the rupee and crude.
- **Scheduled events**: RBI policy, the Budget, US Fed decisions, inflation data, results
  season, and expiry days (Tuesday for Nifty weeklies).

**Gaps.** A stock that opens far from yesterday's close has either news (the gap often holds
or extends) or no news (it often fills back). The first 30 minutes show which, and that is
why every intraday pick here waits until 9:46.

**Intraday patterns with evidence**
- **Market intraday momentum.** The Nifty's move from the previous close to the first half
  hour predicts the direction of the last half hour (Gao, Han, Li and Zhou 2018). Tested in
  the lab's Nifty futures book.
- **Overnight and intraday returns differ.** Much of the long-run return comes overnight
  (Lou, Polk and Skouras 2019).
- **Volume confirms.** Stocks with unusual morning volume ("stocks in play") trend more often
  than quiet ones (Zarattini, Barbon and Aziz 2024).
""")
    with c:
        st.markdown("""
### Charts, honestly

- **Candles**: each shows the open, high, low and close for its period. Green closes above
  its open, red below.
- **Support and resistance**: prices where buyers or sellers appeared before. They are
  useful for placing stops, but not reliable predictions.
- **Moving averages (50 and 200 days)**: a simple trend filter. A market below its 200-day
  average has historically had worse returns and more crashes. The long-term book halves
  new positions in that case.
- **VWAP**: the day's volume-weighted average price, the benchmark big buyers and sellers
  are measured against. Above VWAP means buyers are in control that day.
- **Opening range**: the high and low of the first 15–30 minutes. Breakouts from it are the
  basis of the *Stocks in play* book.
- **RSI, MACD and Bollinger Bands**: different views of momentum and stretch. They are
  features in this system's models, not rules on their own.

**What the research says.** Most named chart patterns (head and shoulders, flags, triangles)
have weak or no edge once tested properly on out-of-sample data and costs. Trend (momentum)
and volume are the parts of technical analysis with consistent evidence. Treat a chart as a
way to see where you are wrong (your stop), not as a forecast.
""")


# --- 4. Trading and risk --------------------------------------------------------------------

def risk() -> None:
    a, b, c, d = st.tabs(["The odds", "Position sizing", "Drawdowns", "Psychology"])
    with a:
        st.markdown("""
### The odds, from SEBI's own studies

- **Intraday cash trading:** about 70% of individual intraday traders lost money in FY23.
  Among those trading more than 500 times a year, about 80% lost (SEBI, July 2024).
- **Futures and options:** 93% of more than 1 crore individual F&O traders lost money over
  FY22–FY24, about ₹2 lakh each on average including costs. Only about 1% made more than
  ₹1 lakh (SEBI, September 2024).

**Why most lose**
- Costs paid on every trade
- Trading too often and too large
- No measured edge
- Losses run while winners are cut

**What wins over time**
- Low costs and few trades
- A small, honestly tested edge
- Strict loss limits
- Long-term investing in good businesses

This is why everything here is paper first, and why a strategy is trusted only after it beats
chance by a clear margin (a t-statistic above 2) over enough days.

**Expectancy** is what a strategy makes per trade on average:

> expectancy = win rate × average win − loss rate × average loss − costs

A 40% win rate is fine if winners are twice the size of losers. A 70% win rate loses money if
the losses are four times the wins, which is the trap in option selling.
""")
    with b:
        st.markdown("### Position sizing")
        st.caption("Risk a fixed small share of capital per trade; let the stop distance decide the size.")
        c1, c2, c3 = st.columns(3)
        cap = c1.number_input("Capital (₹)", 1_000, 10_000_000, 100_000, step=10_000)
        risk_pct = c2.number_input("Risk per trade (%)", 0.1, 5.0, 1.0, step=0.1)
        price = c3.number_input("Share price (₹)", 1.0, 100_000.0, 500.0, step=10.0)
        stop_pct = st.slider("Stop-loss distance (%)", 0.3, 10.0, 1.5, 0.1)
        qty = int(cap * risk_pct / 100 / (price * stop_pct / 100))
        qty = min(qty, int(cap // price))
        UI.cards([UI.card("Shares to buy", f"{qty:,}", f"₹{qty * price:,.0f} position", 0.0),
                  UI.card("Loss if stopped", UI.money(-qty * price * stop_pct / 100),
                          f"{risk_pct:.1f}% of capital", 0.0, "down"),
                  UI.card("Position size", f"{qty * price / cap:.0%}", "of your capital",
                          min(1.0, qty * price / cap))])
        st.markdown("""
- **Risk 0.5–1% of capital per trade.** Ten losses in a row, which happens, then cost about
  10%, not everything.
- **Size from the stop, not from conviction.** A wider stop means fewer shares.
- **Cap the total.** Limit open risk (all stops together) and set a daily loss limit. Every
  paper book here stops for the day at −1.5%.
- **Watch correlation.** Five bank stocks are close to one big bet on banks.
""")
    with c:
        st.markdown("### Drawdowns: why losses matter more than gains")
        loss = st.slider("A fall of", 5, 90, 30, 5, format="%d%%")
        need = 1 / (1 - loss / 100) - 1
        st.markdown(f"needs a gain of **{need:.0%}** to get back to where you started.")
        x = np.arange(5, 91, 5)
        df = pd.DataFrame({"Fall": x / 100, "Gain needed": 1 / (1 - x / 100) - 1})
        st.altair_chart(alt.Chart(df).mark_line(point=True, color=UI.NEUTRAL).encode(
            x=alt.X("Fall:Q", axis=alt.Axis(format="%")),
            y=alt.Y("Gain needed:Q", axis=alt.Axis(format="%"), scale=alt.Scale(type="log"))
        ).properties(height=240), width="stretch")
        st.markdown("""
A strategy's **deepest fall** (maximum drawdown) matters as much as its return. A 50%
drawdown needs a 100% gain to recover, and few people keep following a strategy through one.
Every leaderboard here shows the worst day and the deepest fall next to the total.
""")
    with d:
        st.markdown("""
### Psychology: the mistakes everyone makes

- **Loss aversion**: a loss hurts about twice as much as an equal gain pleases (Kahneman and
  Tversky). So people hold losers too long, hoping, and sell winners too early to lock in
  the pleasure. This is the disposition effect, and it is costly.
- **Overconfidence**: after a few wins people trade bigger and more often. Trading volume
  and returns are negatively related for individual investors (Barber and Odean 2000).
- **Revenge trading**: trying to win back a loss the same day, usually with larger size.
- **Confirmation bias**: reading only the news that agrees with your position.
- **Recency**: expecting the last few weeks to continue forever.

**What helps**
- Write the plan before the trade (entry, stop, target, size) and follow it.
- Keep a journal of why you entered and how you felt.
- Use hard limits (daily loss, number of trades).
- Let rules, or a tested algorithm, do the trading: that is the main real benefit of algos.
""")


# --- 5. Futures and options -------------------------------------------------------------------

def derivatives() -> None:
    a, b, c, d = st.tabs(["Futures", "Options", "Option selling", "Open interest and PCR"])
    with a:
        st.markdown("""
### Futures

A future is an agreement to buy or sell at a fixed price on an expiry date.
- Stock and index futures expire on the last Tuesday of the month.
- They trade in **lots**: Nifty is 65 units, so one lot at 26,000 is about ₹17 lakh of
  exposure.
- You put up only a **margin** (roughly 12–20% for stocks, less for the index). That is
  **leverage**: a 5% move can be a 25–40% gain or loss on your margin.
- Profit and loss are settled every day (**mark-to-market**). A loss must be covered in
  cash the next morning.
- **Basis** = future price − spot price. It is usually positive (interest cost) and shrinks
  to zero at expiry.
- **Rollover**: closing this month's contract and opening next month's. High rollover means
  traders are keeping their positions.

Futures are a cheap way to trade direction (low costs, easy shorting) and a fast way to lose
money (leverage). The F&O book here trades the model's top picks as stock futures to measure
exactly that.
""")
    with b:
        st.markdown("""
### Options

- A **call** gives the right to buy at the strike price. It gains when the price rises.
- A **put** gives the right to sell at the strike price. It gains when the price falls.
- The buyer pays a **premium** and can lose at most that premium. The seller receives it and
  can lose much more.

**Premium = intrinsic value + time value.** Intrinsic value is what the option would be worth
if exercised now. Time value is the price of what might still happen before expiry. It decays
to zero at expiry, slowly at first and fast in the last days.

**The Greeks**
- **Delta**: how much the premium moves for a ₹1 move in the underlying (0 to 1 for calls).
  At the money is about 0.5.
- **Gamma**: how fast delta changes. Highest at the money near expiry, which makes expiry
  days violent.
- **Theta**: time decay per day. The buyer pays it and the seller earns it.
- **Vega**: sensitivity to implied volatility. Options get pricier when fear rises.
- **Implied volatility (IV)**: the volatility the option's price assumes. India VIX is the
  Nifty's 30-day IV.

**Buying options.** The loss is limited, but most out-of-the-money options expire worthless.
You need a large, fast move. **Selling options.** You win often and lose rarely but
heavily, and you need margin and hedges.
""")
    with c:
        st.markdown("""
### Option selling

**Why it can work.** An option's price includes the volatility the market expects. Index
options are usually priced for more movement than actually happens; this is the volatility
risk premium (Coval and Shumway 2001; Bakshi and Kapadia 2003). Nifty's VIX typically sits
2–4 points above the volatility that follows. A seller collects that difference as time value
decays.

**Why it is dangerous.** Gains are small and frequent; losses are rare and large. A few
big-move days decide a year. One public backtest of a time-based Nifty straddle made ₹19,405
in a year, but −₹94,377 without its 25 best days. Costs and spreads eat thin edges.

**The structures** (all sell time value; the bought wings cap the worst loss):
- **Straddle**: sell the at-the-money call and put. Most premium, unlimited risk.
- **Strangle**: sell a call and a put away from the money. Less premium, more room.
- **Iron fly**: a straddle plus bought wings. Capped loss, lower margin.
- **Iron condor**: a strangle plus bought wings. Capped loss, wider profit zone.

**Rules in India (2025–26)**
- Nifty weekly options expire on Tuesdays.
- Each exchange has one weekly index contract.
- The Nifty lot size is 65.
- Short options need extra margin on expiry day.

The **Option selling** page runs five such books side by side to find out which ideas earn.
""")
        x = np.arange(25200, 26801, 10)
        credit, lot = 252.0, 65
        pnl = (credit - np.abs(x - 26000) + np.maximum(np.abs(x - 26000) - 500, 0)) * lot
        pay = pd.DataFrame({"Nifty at the close": x, "Profit or loss (₹)": pnl})
        st.altair_chart(alt.Chart(pay).mark_area(opacity=0.25, color=UI.NEUTRAL,
                                                 line={"color": UI.NEUTRAL}).encode(
            x=alt.X("Nifty at the close:Q", scale=alt.Scale(zero=False)),
            y="Profit or loss (₹):Q").properties(
                height=240, title="Iron fly at 26,000: wings 500 points away, ₹252 credit, 1 lot"),
            width="stretch")
    with d:
        st.markdown("""
### Open interest and the put-call ratio

**Open interest (OI)** is the number of futures or options contracts still open. Read it
together with price:

| Price | OI | Usual reading |
|---|---|---|
| Up | Up | **Long build-up**: new buyers, strength |
| Down | Up | **Short build-up**: new sellers, weakness |
| Up | Down | **Short covering**: sellers exiting; the rise may fade |
| Down | Down | **Long unwinding**: buyers exiting; the fall may fade |

**Put-call ratio (PCR)** = put OI ÷ call OI. Very high PCR is often read as too much fear,
which is contrarian bullish. It is noisy on its own.

**Evidence from this system.** OI changes, build-up, futures basis and PCR, added to the
intraday model as features, improved its daily ranking (IC +0.0072, t = 9.0 over 2,433 days;
experiment E-011b). It was the first new information that clearly helped after many
price-only ideas failed. "Max pain" and single-day OI levels have little evidence behind them.
""")


# --- 6. Algos and this system -------------------------------------------------------------------

def algos() -> None:
    a, b, c, d = st.tabs(["Algo trading, properly", "How this system works", "The strategy lab",
                          "Reading list"])
    with a:
        st.markdown("""
### Algo trading, done properly

An algorithm is a rule a computer follows exactly: what to buy or sell, how much, and when to
get out. Writing the rule is easy. Knowing whether it has an edge is the work.

**From idea to money**
1. An idea with a reason behind it.
2. A backtest on history.
3. Walk-forward testing: each year judged by a model trained only on earlier years.
4. Paper trading on live prices.
5. Small real money.

Each step kills most ideas, and that is the point.

**Traps this project has hit, with numbers**
- **Survivorship bias.** Testing only today's index members hides the companies that failed.
  The long-term backtest fell from 21.6% to 12.4% a year once delisted stocks were included.
- **Overfitting and multiple testing.** Try enough settings and one will look brilliant by
  luck. New settings must beat the current ones with a paired t-statistic of at least 2 on
  a held-out period, and the bar rises with the number of settings tried.
- **Learning from mistakes can overfit.** Replays that weighted the model's wrong picks more
  heavily made later rounds *worse* (₹−386 → ₹−413 a day), so they are not adopted.
- **Costs.** A ₹5 minimum brokerage turned ₹1,000 trades into 1.32% round-trip costs. A
  model with a 0.05% edge per trade cannot beat that.
- **New information beats cleverer models.** Ten price-only ideas failed. F&O open interest
  was the first to clearly improve the intraday model.

**In India, SEBI's 2025 retail-algo rules** require algorithms run through brokers' APIs to be
registered with the exchange through the broker, with a unique ID. This system only paper
trades, so they don't apply here yet.
""")
    with b:
        st.markdown("""
### How this system works, in one day

| Time | Where | What |
|---|---|---|
| 8:30 | GitHub | Overnight company and sector news fetched and scored |
| 9:15 – 9:45 | Mac | Live prices every minute; the first 30 minutes are recorded |
| 9:46 | Mac | Both intraday models rank every stock |
| 9:46 – 15:15 | Mac | Stops, targets and the daily loss limit are checked every minute |
| 9:20 – 15:15 | Mac | Option-selling books enter, manage and exit their trades |
| 12:30 / 15:15 | Mac | Books square off; every pick is judged against what happened |
| 15:40 | Mac | The day's paper results go to GitHub |
| 18:37 | GitHub | Market data, F&O and delivery data, results calendar, IPOs updated |
| 21:07 | GitHub | Nightly training (below) |

At 9:46, after the ranking:
- The best 25 buys and 25 sells are saved and judged later.
- From those, 3–5 a side are traded on paper. The count depends on how many pass the checks
  (news, sector and market direction) and are not moving sideways.

The long-term book picks 5 stocks once a week, with a negative-news filter.

**How it learns every night (GitHub)**
1. **Retrains** each model on all history up to that day (about 2,500 days for intraday),
   plus every judged paper pick.
2. **Tunes the trading rules**: how many trades, the stop, the target, and which days to
   skip, chosen by paper profit on recent history.
3. **Replays** history as paper trading in rounds. It keeps a round's lessons only if they
   improve both profit and accuracy.
4. **Tries new model settings** and adopts them only if they win on data they never saw.

New information (news features, sector news, positioning data) is tested the same way before
it joins the model.
""")
    with c:
        st.markdown("""
### The strategy lab

Paper trading costs nothing, so many ideas run side by side on the same live prices with the
same costs. The **Strategy lab** page ranks them all.

| Book | Idea | Evidence |
|---|---|---|
| Model, until 12:30 / until the close | Machine-learned ranking of all stocks | This project's walk-forward tests |
| Model + morning candles | The model's shortlist re-ranked by the first 30 minutes | Experiment |
| News drift | Fresh company or sector news, traded in its direction | Chan 2003; Savor 2012; Tetlock 2008 |
| Gap fade without news | Big gaps with no news, traded back towards the close | Savor 2012 |
| Stocks in play | Highest morning volume, traded with the opening move | Zarattini, Barbon and Aziz 2024 |
| Weekly reversal | Last week's losers bought, winners sold | Lehmann 1990; Jegadeesh 1990 |
| Nifty last-half-hour momentum | Nifty futures 14:55–15:25 with the morning move | Gao, Han, Li and Zhou 2018 |
| Model picks as futures / options | The model's top picks with leverage | Experiment |
| Five option-selling books | Volatility risk premium, hedged | Coval and Shumway 2001; Bakshi and Kapadia 2003 |

Judge a book after at least 20 trading days, and trust it only with a t-statistic above 2.
Before that, the differences are mostly luck.
""")
    with d:
        st.markdown("""
### Reading list

**Papers**
- Coval and Shumway (2001), [Expected Option Returns](https://business.baylor.edu/don_cunningham/Option%20Returns.pdf), Journal of Finance.
- Bakshi and Kapadia (2003), *Delta-Hedged Gains and the Negative Market Volatility Risk Premium*, Review of Financial Studies.
- Jegadeesh and Titman (1993), *Returns to Buying Winners and Selling Losers*, Journal of Finance.
- Tetlock (2007), *Giving Content to Investor Sentiment: The Role of Media in the Stock Market*, Journal of Finance.
- Savor (2012), *Stock Returns after Major Price Changes: The Role of News*, Journal of Financial Economics.
- Gao, Han, Li and Zhou (2018), [Market Intraday Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866), Journal of Financial Economics.
- Lou, Polk and Skouras (2019), [A Tug of War: Overnight Versus Intraday Expected Returns](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650), Journal of Financial Economics.
- Gu, Kelly and Xiu (2020), [Empirical Asset Pricing via Machine Learning](https://dachxiu.chicagobooth.edu/download/ML.pdf), Review of Financial Studies.
- Barber and Odean (2000), *Trading Is Hazardous to Your Wealth*, Journal of Finance.
- Ritter and Welch (2002), *A Review of IPO Activity, Pricing, and Allocations*, Journal of Finance.

**Books**
- Benjamin Graham, *The Intelligent Investor*: margin of safety and investor temperament.
- Aswath Damodaran, *The Little Book of Valuation*: valuation without jargon.
- Marcos López de Prado, *Advances in Financial Machine Learning*: backtest overfitting and purged cross-validation.
- Ernest Chan, *Algorithmic Trading: Winning Strategies and Their Rationale*.
- Sheldon Natenberg, *Option Volatility and Pricing*.
- Euan Sinclair, *Volatility Trading*.
- Daniel Kahneman, *Thinking, Fast and Slow*: the biases behind most trading mistakes.

**Official sources**
- SEBI's studies of individual traders' results in F&O (September 2024) and intraday cash trading (July 2024), under Reports and Statistics → Research at [sebi.gov.in](https://www.sebi.gov.in).
- NSE corporate announcements and shareholding patterns, for each company's filings.

**Practitioner material**
- AlgoTest, [The 9:20 straddle strategy](https://algotest.in/blog/the-920-straddle-strategy-defined/).
- Option Alpha, [0DTE strategies: insights from 25k trades](https://optionalpha.com/blog/0dte).
""")
