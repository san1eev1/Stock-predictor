"""Learn: a complete guide to the Indian market, from how an order is filled to how a
professional trading desk works, in six parts plus a glossary. Each part is a tab; each topic
opens on click. Rules and figures are as of late 2026: tax and SEBI rules change, so check
the current circulars before acting on them."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from stockpredictor.app import ui as UI

PARTS = ["1 · Market basics", "2 · Reading a company", "3 · Reading the market",
         "4 · Trading and risk", "5 · Futures and options", "6 · Algos and this system",
         "Glossary"]


def render() -> None:
    st.title("Learn")
    st.caption("A complete guide in six parts and a glossary. Read the parts in order the first "
               "time; later, open any topic on its own. Where this project has its own evidence, "
               "the numbers are given.")
    tabs = st.tabs(PARTS)
    for tab, fn in zip(tabs, (basics, company, market, risk, derivatives, algos, glossary)):
        with tab:
            fn()


def sec(title: str, body: str | None = None, expanded: bool = False):
    """One click-to-open topic. With `body`, writes it; otherwise returns the expander."""
    box = st.expander(title, expanded=expanded)
    if body is not None:
        with box:
            st.markdown(body)
    return box


# --- 1. Market basics ---------------------------------------------------------------------

def basics() -> None:
    st.markdown("How the Indian stock market is organised, how a price is made, and what every "
                "trade costs.")
    sec("What a share is, and why prices move", """
A share is a small piece of ownership in a company: a claim on its future profits. When the
company earns more over the years, each share is worth more, and the price follows.

- **Over years**, a share follows the company's profits and cash flow.
- **Over weeks and days**, it follows news, money flows and mood.
- **Over minutes**, it is mostly noise.

Long-term investing therefore rests on business quality. Short-term trading needs a measured
edge and very low costs, because most short-term moves are random.
""")
    sec("Who's who: exchanges, SEBI, brokers, depositories", """
| Who | What they do |
|---|---|
| **NSE, BSE** | The exchanges where shares are bought and sold. Most volume is on the NSE |
| **SEBI** | The regulator: rules for companies, brokers, traders and mutual funds |
| **Brokers** (Angel One, Zerodha, Groww…) | Send your orders to the exchange; must keep your money separate |
| **Depositories** (NSDL, CDSL) | Hold your shares in your demat account, in your name, not the broker's |
| **Clearing corporations** (NSE Clearing, ICCL) | Guarantee both sides of every trade |
| **Registrars (RTAs)** | Keep the company's shareholder records; handle IPO allotment |

**Settlement is T+1.** Shares and money change hands the next working day. Since 2024 a
same-day (T+0) option exists for some stocks.
""")
    sec("How a price is made: the order book", """
The exchange keeps an **order book** for each stock: all buy orders (bids) and sell orders
(asks) waiting at each price.

- The **best bid** is the highest price someone will pay now; the **best ask** is the lowest
  price someone will sell at.
- The gap between them is the **spread**. Large stocks have spreads of 0.01–0.05%; small
  stocks can have 0.5% or more.
- A **market order** takes the best price on the other side. A big order "walks the book",
  filling at worse and worse prices. That is **market impact**.
- **Liquidity** is how much you can trade without moving the price. Daily traded value is a
  rough guide: under ₹10 crore a day, trade small.

This is why the paper books include slippage in costs (0.05% per order) and avoid thin stocks.
""")
    sec("The trading day and order types", """
| Time (IST) | What happens |
|---|---|
| 9:00 – 9:08 | **Pre-open**: orders collected, no trades |
| 9:08 – 9:15 | Opening price found by auction |
| 9:15 – 15:30 | **Normal market**: continuous trading |
| 15:30 – 15:40 | Closing price = average of the last 30 minutes |
| 15:40 – 16:00 | Post-close session at the closing price |

**Order types**
- **Market**: now, at the best available price. Fast; can be expensive in thin stocks.
- **Limit**: only at your price or better. It may not fill.
- **Stop-loss (SL)**: a limit order that activates when the trigger price is touched.
- **Stop-loss market (SL-M)**: a market order that activates at the trigger. It always
  exits, possibly at a worse price.
- **GTT (good till triggered)**: waits up to a year for its trigger. Use it for long-term
  targets and stops.
- **AMO (after-market order)**: placed after hours, sent at the next open.
""")
    sec("Product types, margin and leverage", """
- **CNC (delivery)**: you pay in full and own the shares; hold as long as you like.
- **MIS (intraday)**: must be closed the same day. The broker gives leverage (up to 5× on
  liquid stocks) and squares you off before the close if you don't.
- **NRML**: futures and options carried overnight.
- **MTF (margin trading facility)**: the broker lends part of the money for delivery
  purchases, with interest (often 12–18% a year). It is leverage with a running cost.
- **Pledging**: your holdings can be pledged as margin for F&O, with a haircut.

**Leverage cuts both ways.** At 5× leverage, a 2% move against you is a 10% loss of your
money. Most retail blow-ups come from leverage, not from bad stock picks.
""")
    sec("Circuit limits and surveillance", """
- **Price bands**: most stocks may move at most 2%, 5%, 10% or 20% in a day. At the band
  ("upper/lower circuit") trading continues only at that price; often you cannot exit.
- **Market-wide breakers**: a 10%, 15% or 20% move in the Nifty or Sensex halts all trading
  for a while.
- **ASM / GSM**: additional and graded surveillance on unusually volatile or suspicious
  stocks: higher margins, trade-for-trade settlement, sometimes once-a-week trading.
- **T2T (trade-to-trade, BE series)**: every trade must be delivered; no intraday.

A stock in ASM/GSM or stuck in circuits is a warning, whatever the story behind it.
""")
    sec("Indices and market-cap categories", """
- **Nifty 50**: the 50 largest NSE companies, weighted by free-float market value.
- **Sensex**: the 30 largest BSE companies.
- **Sector indices**: Nifty Bank, IT, Pharma, Auto, Metal, FMCG, Realty, PSU Bank…
- **Nifty Midcap 150, Smallcap 250**: smaller companies. Faster in good years, harder falls
  in bad ones.
- **India VIX**: the volatility the options market expects for the Nifty over 30 days, in
  yearly %. About 11–14 is calm; above 20 is fear.

**SEBI's size categories**, updated every six months by AMFI:
- **Large caps**: the top 100 companies by market value
- **Mid caps**: companies 101–250
- **Small caps**: from 251 down

This system covers the 250 large and mid caps (the Nifty LargeMidcap 250).
""")
    sec("Corporate actions: dividends, splits, bonus, rights, buybacks", """
- **Dividend**: cash paid per share. On the **ex-date** the price drops by about the
  dividend. With T+1 settlement the ex-date and record date are the same day.
- **Split**: one share becomes several (a 1:5 split turns ₹1,000 × 1 into ₹200 × 5). Your
  value doesn't change.
- **Bonus**: free extra shares (1:1 doubles your shares, halving the price). Value doesn't
  change either.
- **Rights issue**: the right to buy new shares, usually at a discount. Not taking part
  dilutes you; the rights entitlement can also be sold.
- **Buyback**: the company buys its own shares. Through the tender route, small
  shareholders get a reserved quota, often at a premium.

Charts and this system's data are adjusted for splits and bonuses. The paper books adjust
held positions on the ex-date, so a split never looks like a crash.
""")
    sec("What a trade costs", """
**Costs per trade at a typical discount broker**
- **Brokerage**: ₹0 for delivery at many brokers. For intraday and F&O, ₹20 per order or a
  % (0.03%–0.1% depending on broker), whichever is lower.
- **STT (securities transaction tax)**:
  - Delivery: 0.1% on both buy and sell
  - Intraday: 0.025% on the sell side
  - Futures: 0.02% on the sell side
  - Options: 0.1% of the premium on the sell side
- **Smaller charges**: exchange transaction charges, SEBI fee, 18% GST on brokerage and
  charges, and stamp duty on buys.
- **Delivery only**: DP charge (about ₹15–20 per stock sold).
- **Slippage and spread**: the hidden cost, often larger than all of the above.

**This project's measurement:** the intraday paper books pay about **0.37% per round trip**.
That is 0.1% brokerage each side at a ₹5 minimum, plus STT, charges and 0.05% slippage per
order, on about ₹9,000 per trade. Over 90 days, costs were larger than the books' whole
result. Costs are the first thing a trading business attacks.
""")
    sec("Tax on gains", """
Resident individuals, rules from the July 2024 budget:

| What | Tax |
|---|---|
| Short-term capital gains (listed shares held 12 months or less) | 20% |
| Long-term capital gains (held over 12 months) | 12.5% on gains above ₹1.25 lakh a year |
| Intraday profits | Speculative business income, taxed at your slab |
| F&O profits | Non-speculative business income, taxed at your slab |
| Dividends | Taxed at your slab (TDS above ₹5,000 a year per company) |

- Losses can be set off and carried forward under conditions: speculative losses only
  against speculative gains, for 4 years; F&O losses for 8 years.
- Large F&O turnover may require a tax audit.
- **Tax-loss harvesting**: selling a loser before 31 March to offset gains. It is legal;
  you can buy back after.

Rules change in budgets; check the current year's rules or ask a tax adviser.
""")
    sec("Staying safe", """
- Your shares sit with the depository in your name. Check your CDSL/NSDL statement monthly;
  it is the source of truth, not the broker's app.
- Never share your OTP, TPIN or broker password. No legitimate adviser needs them.
- Be wary of tips on Telegram or WhatsApp, "guaranteed returns" and "account handling". Only
  SEBI-registered investment advisers and research analysts may charge for advice; check
  their registration number on SEBI's site.
- Pump-and-dump schemes run in small, illiquid stocks. Surveillance flags (ASM/GSM) and
  sudden circuits are warnings.
- Add a nominee to your demat and bank accounts.
""")


# --- 2. Reading a company -----------------------------------------------------------------

def company() -> None:
    st.markdown("How to judge a business and its price: the statements, the ratios, the red "
                "flags, results season and IPOs.")
    sec("Start with the business", """
Before any number, answer in a sentence each:
1. **What does it sell, to whom, and why do they buy from it?**
2. **What stops a competitor from taking its customers?** Brand, cost advantage, network,
   licences, switching costs. This is the "moat".
3. **How does the industry behave?** Commodity businesses (steel, cement, chemicals) swing
   with prices and cycles; consumer brands and software are steadier.
4. **Who runs it, and are they honest with shareholders?** Read three years of the
   chairman's letters and compare what was promised with what happened.
5. **What could kill it?** Regulation, technology, one big customer, debt.

If you cannot explain the business simply, the numbers will not save you.
""")
    sec("The profit and loss statement", """
What the company earned over a period:

> Revenue − operating costs = **EBITDA** − depreciation = **EBIT** − interest = **profit before tax** − tax = **net profit**

- **Revenue growth**: is the business getting bigger? Separate price from volume where
  disclosed.
- **Margins** (EBITDA ÷ revenue, net profit ÷ revenue): rising costs show here first.
- **Other income**: interest and one-offs. Profit growth driven by it is low quality.
- **Exceptional items**: one-time gains or losses. Strip them out when comparing years.

Look at five years, not one quarter.
""")
    sec("The balance sheet", """
What the company owns and owes on one date:

- **Assets**: plants and equipment, cash, investments, receivables (money customers owe),
  inventory.
- **Liabilities**: borrowings, payables (money owed to suppliers), provisions.
- **Equity**: what belongs to shareholders (assets − liabilities).

What to check:
- **Debt vs equity and vs cash flow.** Net debt ÷ EBITDA above 3 is heavy for most businesses.
- **Receivables and inventory vs sales.** Growing much faster than sales means trouble
  collecting or selling.
- **Goodwill**: the premium paid in acquisitions. Large write-offs later mean overpayment.
- **Contingent liabilities** in the notes: lawsuits and guarantees that are not on the
  balance sheet yet.
""")
    sec("The cash flow statement", """
Where the cash actually went:
- **Operating cash flow (CFO)**: cash from the business. Over several years it should be
  close to net profit. Profit without cash is the oldest warning sign in accounting.
- **Investing cash flow**: capital spending, acquisitions, investments.
- **Financing cash flow**: borrowing, repaying, dividends, buybacks, share issues.
- **Free cash flow (FCF)** = CFO − capital spending. This is what can be returned to
  shareholders or used to repay debt.

A company that keeps raising money or borrowing to pay dividends is paying you with your
own money.
""")
    sec("The ratios that matter", """
| Ratio | What it tells you | Rough guide |
|---|---|---|
| Revenue growth (3–5 yr) | Is the business getting bigger? | Compare with nominal GDP (~10–11%) |
| Operating (EBITDA) margin | Profit per ₹100 of sales | Compare within the sector only |
| ROE | Profit per ₹ of shareholders' money | Above 15% is good |
| ROCE | Profit per ₹ of all capital, debt included | Above 15%; above the cost of debt |
| Debt / equity | How much it leans on borrowing | Below 1 for most non-financials |
| Interest coverage | EBIT ÷ interest | Below 3 is fragile |
| CFO / net profit | Are profits turning into cash? | Close to 1 over 3–5 years |
| Receivable days | How fast customers pay | Rising every year is a warning |
| Asset turnover | Sales per ₹ of assets | Shows capital intensity |

**ROE can be flattered by debt**: a company can borrow to raise ROE. ROCE avoids that. Use
both.
""")
    sec("Banks and NBFCs are read differently", """
Lenders borrow money and lend it; their "raw material" is money, so the usual ratios don't
work.

| Measure | Meaning | Good sign |
|---|---|---|
| **NIM** (net interest margin) | Interest earned − interest paid, ÷ loans | Stable or rising |
| **Gross / net NPA** | Loans not being repaid | Low and falling |
| **Provision coverage** | Share of bad loans already written off | Above 70% |
| **Credit cost** | New provisions ÷ loans | Low and steady |
| **CASA ratio** | Cheap current and savings deposits | High (cheaper funding) |
| **Capital adequacy (CRAR)** | Capital ÷ risky assets | Well above the minimum |
| **Loan growth vs deposit growth** | Can it fund its lending? | Not far apart |

Value lenders with price-to-book (P/B) against their ROE: a bank with 18% ROE deserves a
higher P/B than one with 10%.
""")
    sec("Sector cheat sheet: what to watch", """
| Sector | Watch |
|---|---|
| IT services | Deal wins, revenue growth in dollars, margins, attrition, US demand and visa rules |
| Pharma | USFDA inspections and approvals, US generic price erosion, launches, R&D |
| Autos | Monthly volumes (1st of the month), inventory, commodity costs, EV shift |
| Cement | Volumes, realisation per tonne, power and fuel costs, capacity additions |
| Steel and metals | Global prices, China, import duties, coking coal cost |
| FMCG | Volume growth, rural demand, monsoon, raw-material inflation |
| Banks | Loan growth, NIM, NPAs, deposit costs, RBI rates |
| Telecom | Tariffs, ARPU (revenue per user), subscriber adds, spectrum costs |
| Power and utilities | Demand, tariff orders, capacity, coal supply |
| Capital goods, defence | Order book vs revenue, execution, government capex |
| Real estate | Pre-sales (bookings), collections, launches, debt |
| Oil marketing (OMCs) | Crude vs fuel price, marketing margins, government policy |

The **Stock news** page tracks sector and policy news on these themes and maps it to the
stocks it affects.
""")
    sec("Valuation with multiples", """
- **P/E** (price ÷ earnings per share): how many years of today's profit you pay. A P/E of 60
  needs years of fast growth just to stand still.
- **EV/EBITDA**: like P/E but counts debt; better for comparing companies with different
  borrowing.
- **P/B**: banks, NBFCs and asset-heavy businesses.
- **PEG** (P/E ÷ growth %): a P/E of 30 with 30% growth can be cheaper than 15 with 5%.
- **Dividend yield, FCF yield**: the cash you get now.

**How to use them**
- Compare with the company's own 5–10 year range and its closest peers, never in isolation.
- "Cheap on everything" often means cheap for a reason: falling profits, debt, governance,
  or a dying product.
- Expensive stocks can stay expensive while growth lasts and fall hardest when it slows.

**The evidence across decades and countries.** Three things have beaten the market on
average over long periods, with long dry spells:
- **Momentum**: the biggest risers of the past 12 months, excluding the last month.
- **Quality**: profitable, low-debt companies.
- **Value**: cheap relative to fundamentals.

This system's long-term model uses all three together.
""")
    with sec("Valuation calculator: what is it worth to you?"):
        st.caption("A simple earnings model: today's earnings per share grow for some years, then the "
                   "market pays an exit P/E; discount it back at the return you want.")
        c1, c2, c3 = st.columns(3)
        eps = c1.number_input("Earnings per share now (₹)", 0.1, 10_000.0, 50.0)
        g = c2.slider("Growth a year (%)", -10, 40, 15)
        yrs = c3.slider("Years", 1, 15, 5)
        c4, c5, c6 = st.columns(3)
        exit_pe = c4.slider("P/E at the end", 5, 80, 25)
        want = c5.slider("Return you want a year (%)", 6, 25, 12)
        price = c6.number_input("Price today (₹)", 1.0, 1_000_000.0, 1_500.0)
        future = eps * (1 + g / 100) ** yrs * exit_pe
        fair = future / (1 + want / 100) ** yrs
        implied = (future / price) ** (1 / yrs) - 1
        UI.cards([UI.card("Price in the end year", f"₹{future:,.0f}", f"EPS ₹{eps * (1 + g / 100) ** yrs:,.1f} × {exit_pe}", 0.0),
                  UI.card("Worth to you today", f"₹{fair:,.0f}", f"at {want}% a year", 0.0,
                          "up" if fair >= price else "down"),
                  UI.card("Return at today's price", f"{implied:+.1%} a year", "if these assumptions hold",
                          min(1.0, max(0.0, implied / 0.3)), "up" if implied >= want / 100 else "down")])
        st.caption("Change one assumption at a time to see which matters most; usually it is growth "
                   "and the exit P/E. If a small change flips the answer, the price leaves no margin "
                   "of safety.")
    sec("Quality and red flags", """
**Good signs**
- Steady ROCE above 15% through a full business cycle
- Operating cash flow close to profit
- Low or falling debt
- Promoters holding steady or buying
- Clean, unchanged auditors
- Few related-party transactions
- Dividends or buybacks paid from free cash flow

**Red flags**
- **Pledged promoter shares**: if the price falls, lenders sell, and the price falls more.
- **The auditor resigns** or qualifies the accounts.
- **Receivables or inventory growing much faster than sales.**
- **Frequent equity raises**, warrants to promoters, preferential allotments.
- **Big related-party deals**: money moving between the company and the promoters' other
  businesses.
- **Large contingent liabilities** in the notes.
- **Steady promoter selling** while management talks up the future.
- **A sudden change of business** into whatever is fashionable.
- **Profits rising but cash flow falling** for several years.
""")
    sec("The shareholding pattern", """
Published every quarter on NSE and BSE:

| Holder | What changes tell you |
|---|---|
| **Promoters** | Rising: confidence. Falling or pledged: ask why |
| **FPIs** (foreign funds) | Big, fast-moving money; heavy selling weighs on large caps |
| **DIIs** (mutual funds, insurers) | Rising holdings mean domestic analysts are buying in |
| **Public / retail** | Rising retail share while institutions sell is a warning |

Also check for **bulk and block deals** (big trades reported the same day) and **insider
trading disclosures** (SAST and PIT filings) when promoters or directors buy or sell.
""")
    sec("Quarterly results and conference calls", """
Companies report within 45 days of each quarter's end (60 days for the March quarter).
**What moves the price is the surprise, not the number**: profit up 20% can send a stock
down if the market expected 30%.

Read, in order:
1. **Revenue growth** and its mix (price vs volume, segments)
2. **Margins**: rising costs show up here first
3. **One-offs**: exceptional items, tax write-backs, other income
4. **Management commentary and guidance**, in the investor presentation and the earnings
   call. Listen for what they stop talking about.
5. **Order book** for capital goods and defence; **asset quality** for banks

**After the results.** Stocks that surprise positively tend to keep drifting the same way for
weeks (post-earnings announcement drift, Bernard and Thomas 1989). Results-day moves are
large and hard to predict, so the intraday model's rules can skip stocks reporting that day.
""")
    sec("Analysing an IPO, step by step", """
1. **Read the red herring prospectus (RHP).** It covers the business, the risk factors,
   three years of financials and the "objects of the issue". It is on SEBI's and NSE's sites.
2. **Fresh issue or offer for sale?** A fresh issue raises money for the company. An offer
   for sale (OFS) is existing owners selling. A large OFS by private-equity holders deserves
   a harder look at the price.
3. **Price it against listed peers.** At the top of the band, compare P/E, EV/EBITDA, P/B,
   ROE and growth. Paying more than peers needs faster growth.
4. **Check ownership after listing.** Look at promoter holding, any pledges, and lock-ins.
5. **Read the anchor book.** Institutions buy the day before opening; their names and the
   30/90-day lock-in show who is committed.
6. **Watch subscription by category, especially on the last day.** Institutions (QIBs) bid
   mostly on the final day. In this system's data (NSE mainboard IPOs), the share that listed
   above the issue price was:
   - QIB demand under 3×: about 26% (median listing −4%)
   - QIB 10–50×: about 62% (median +5%)
   - QIB 50–100×: about 92% (median +29%)
   - QIB over 100×: about 96% (median +36%)
7. **Read HNI demand for what it is.** Big HNIs (bids above ₹10 lakh) often borrow to apply
   and sell on listing day. Very high HNI demand supports the listing pop, not the business.
8. **Treat the grey-market premium as gossip.** It is unofficial and easy to move.
9. **Know your allotment odds.** Retail allotment is a lottery of one lot per applicant;
   apply for one lot only.
10. **Decide your listing-day plan in advance.** On average, IPOs make most of their gain on
    listing day and lag the market over the following years (Ritter 1991).

The **IPO analysis** page shows all of this live for every open IPO, with a final call.
Tested year by year on IPOs the model had not seen, its "Apply" calls listed up 87% of the
time (median +28%), against 68% for all IPOs.
""")
    sec("Funds and ETFs vs picking stocks", """
- **Index funds and ETFs** (Nifty 50, Nifty Next 50, Midcap 150) give the market's return
  at 0.05–0.3% a year. Most active funds fail to beat their index after fees over 10 years
  (SPIVA India reports).
- **Factor index funds** (momentum, quality, low volatility, value) package the evidence-
  backed tilts cheaply.
- **Picking stocks** makes sense if you have an edge: time, knowledge of an industry, or a
  tested system. Measure yourself against the index honestly.

A sensible core for most people: index funds through a monthly SIP, with stock picking or
trading in a smaller satellite part you can afford to get wrong.
""")


# --- 3. Reading the market ----------------------------------------------------------------

def market() -> None:
    st.markdown("What moves prices from day to day: news, the economy, money flows, mood, "
                "the calendar, and what charts can and cannot tell you.")
    sec("News that moves stocks, and news that doesn't", """
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
| Buybacks, bonus issues, splits, dividends | Small; mostly known in advance |
| Sector policy (duties, tariffs, rates, prices) | Moves the whole sector together |

**Noise this system drops**
- Tips and targets, brokerage upgrades and downgrades
- "Likely to rise" and "should you buy" pieces
- Price reports and generated ratings
- Lists of stocks and chart or open-interest commentary
- Job adverts and admin filings
- Opinion mills

**How the system reads news**
- **Fetching**: GitHub fetches Google News for every stock and for 15 sector and policy
  themes at 8:30, 10:00, 12:00, 14:00 and 17:00 on weekdays.
- **Right company**: each headline must name its company.
- **Tone**: FinBERT gives each headline a tone from −1 to +1.
- **Weight**: a second model, trained on more than 40,000 past headlines, weights each by how
  much such headlines moved their stock (out-of-sample correlation 0.20, against 0.15 for
  tone alone).

**The evidence.** Moves that come with news tend to continue; moves without news tend to
reverse (Chan 2003; Savor 2012). Negative words in news predict lower returns over the next
day or two (Tetlock 2007).
""")
    sec("The economy: rates, inflation, growth, rupee, oil", """
| Driver | Why it matters | Who it hits most |
|---|---|---|
| **RBI repo rate** | Cost of money; lower rates lift valuations and loan demand | Banks, NBFCs, real estate, autos |
| **Inflation (CPI)** | Drives the RBI's decisions; squeezes margins | Consumer companies |
| **GDP growth, PMI, GST collections** | Demand in the economy | Cyclicals, capital goods |
| **Rupee (USD/INR)** | Weak rupee helps exporters and hurts importers | IT and pharma gain; oil and electronics importers lose |
| **Crude oil** | India imports ~85% of its oil | OMCs, paints, airlines, chemicals |
| **US Fed and bond yields** | Higher US yields pull foreign money out of India | Large caps through FPI flows |
| **Monsoon** | Rural incomes | FMCG, tractors, two-wheelers, fertilisers |
| **Government capex and the Budget** | Orders for infrastructure | Capital goods, cement, defence, railways |
""")
    sec("Money flows: FPIs and DIIs", """
- **FPIs** (foreign portfolio investors) own about a sixth of Indian shares, mostly large
  caps. Heavy selling for weeks, often when US yields rise or the dollar strengthens, can pull
  the Nifty down even while companies do well.
- **DIIs** (mutual funds, insurers, pension funds) are fed by monthly SIPs of over ₹25,000
  crore. They have often bought what FPIs sold, which has made Indian falls shallower since
  2020.
- Daily provisional figures come out every evening; this system downloads them.

Flows explain the market's direction better than they predict it.
""")
    sec("Market mood: breadth, VIX, sector rotation", """
- **Breadth**: how many stocks rose vs fell. An index up on narrow breadth (a few big stocks)
  is weaker than one up on broad breadth.
- **Advance/decline ratio** and **% of stocks above their 200-day average**: above 70% is
  strong, below 30% is a washed-out market.
- **India VIX**: rising VIX means fear and larger daily moves; option premiums rise with it.
- **Sector rotation**: money moves from defensives (FMCG, pharma, IT) to cyclicals (banks,
  metals, autos, capital goods) when growth looks better, and back when it doesn't.
- **Euphoria signs**: record demat openings, IPOs oversubscribed hundreds of times, small
  caps far outrunning large caps. **Fear signs**: VIX above 25, heavy FPI selling, most
  stocks below their 200-day average.
""")
    sec("The calendar", """
| When | What | Effect |
|---|---|---|
| Results season (mid-Jan, Apr, Jul, Oct) | Companies report | Big single-stock moves |
| RBI policy (about every 2 months) | Rates and liquidity | Banks, rate-sensitives |
| Union Budget (1 February) | Taxes, spending, duties | Whole market, sector policies |
| US Fed meetings (8 a year) | US rates | FPI flows, IT |
| Nifty weekly expiry (Tuesday) | Options expire | Volatile last hours |
| Monthly expiry (last Tuesday) | Stock F&O expire, rollover | Positioning shifts |
| MSCI / FTSE index reviews | Passive funds buy or sell | Large moves on review day in affected stocks |
| 1st of the month | Auto sales, GST collections | Autos, macro mood |
| Holidays | Exchange closed | Check NSE's holiday list |
""")
    sec("Global cues and gaps", """
Before 9:15, the market reacts to what happened overnight:
- **US close**: the S&P 500 and Nasdaq, especially for IT stocks.
- **Asian markets**: Nikkei, Hang Seng, and the GIFT Nifty futures in Gandhinagar.
- **Crude oil, the dollar and US bond yields.**

A stock that opens far from yesterday's close has **gapped**:
- **With news**, the gap often holds or extends.
- **Without news**, it often fills back.

The first 30 minutes show which, which is why every intraday pick here waits until 9:46.
The lab's *Gap fade without news* book tests the reversal side.
""")
    sec("Market cycles, crashes and bubbles", """
- **Corrections** (10–20% falls) happen most years. **Bear markets** (over 20%) a few times a
  decade.
- **Indian history**:
  - 2008: the Nifty fell about 60%.
  - 2015–16: about 23%.
  - March 2020: about 38% in five weeks, then recovered within a year.
  - Small caps fall 50–70% in bad cycles.
- **Bubbles** share features: a new-era story, easy money, retail euphoria, IPO frenzy, and
  valuations justified by "this time is different".
- **What survives**:
  - Quality businesses bought at sensible prices
  - Diversification
  - No leverage you can't hold through a 40% fall
  - Steady SIPs that buy more when prices are low

A market below its 200-day average has historically had worse returns and more crashes. The
long-term book halves new positions in that case.
""")
    sec("Charts, honestly", """
- **Candles**: open, high, low and close for each period; green closes above its open.
- **Support and resistance**: prices where buyers or sellers appeared before. Useful for
  placing stops; not reliable predictions.
- **Trend**: higher highs and higher lows. **Moving averages** (50 and 200 days) are a
  simple trend filter.
- **Volume**: moves on heavy volume are more meaningful than moves on thin volume.
- **VWAP**: the day's volume-weighted average price, the benchmark large traders are judged
  against.
- **Opening range**: the high and low of the first 15–30 minutes. Breakouts from it on heavy
  volume are the *Stocks in play* book.

**What the research says.** Most named chart patterns (head and shoulders, flags, triangles)
show weak or no edge once tested out of sample and after costs. Trend (momentum) and volume
are the parts with consistent evidence. Use a chart to decide where you are wrong (your stop),
not as a forecast.
""")
    sec("Indicators, explained", """
| Indicator | Measures | Common reading |
|---|---|---|
| **RSI (14)** | Speed of recent gains vs losses, 0–100 | Above 70 stretched up, below 30 stretched down; in strong trends it stays stretched |
| **MACD** | Gap between a fast and a slow moving average | Crossing up/down signals momentum turning |
| **Bollinger Bands** | Price vs its 20-day average ± 2 standard deviations | Touching a band = stretched; narrow bands = a big move often follows |
| **ATR** | Average daily range | Size stops and positions to it |
| **ADX** | Strength of a trend | Above 25 = trending |
| **OBV** | Volume flowing in or out | Divergence from price can warn |

All of them are transformations of price and volume, so they overlap. This system feeds
several to its models as features and lets the data decide their weight, rather than using
any one as a rule.
""")
    sec("Intraday patterns with evidence", """
- **Market intraday momentum.** The Nifty's move from the previous close to the first half
  hour predicts the direction of the last half hour (Gao, Han, Li and Zhou 2018). This is the
  lab's Nifty futures book.
- **Overnight vs intraday.** Much of the long-run return comes overnight; intraday returns
  behave differently (Lou, Polk and Skouras 2019).
- **Stocks in play.** Stocks with unusual morning volume trend more often than quiet ones
  (Zarattini, Barbon and Aziz 2024).
- **U-shaped volume and volatility.** Most volume and the biggest moves come in the first and
  last hour; the middle of the day is quieter and noisier.
- **Expiry days.** Index options expiring that day pin prices near big strikes, then can whip
  in the last hour.
""")


# --- 4. Trading and risk --------------------------------------------------------------------

def risk() -> None:
    st.markdown("Why most traders lose, and the habits, numbers and tools that keep losses "
                "small enough to survive long enough to find an edge.")
    sec("The odds, from SEBI's own studies", """
- **Intraday cash trading:** about 70% of individual intraday traders lost money in FY23.
  Among those trading more than 500 times a year, about 80% lost (SEBI, July 2024).
- **Futures and options:** 93% of more than 1 crore individual F&O traders lost money over
  FY22–FY24, about ₹2 lakh each on average including costs. Only about 1% made more than ₹1
  lakh (SEBI, September 2024).

**Why most lose**
- Costs on every trade
- Trading too often and too big
- No measured edge
- Letting losses run and cutting winners

**What wins over time**
- Low costs, few trades
- A small, honestly tested edge
- Strict loss limits
- Long-term ownership of good businesses
""")
    sec("Trading styles compared", """
| Style | Holding time | What drives returns | Main enemy |
|---|---|---|---|
| Long-term investing | Years | Business growth, compounding | Impatience, overpaying |
| Positional / factor | Weeks to months | Momentum, quality, value | Long dry spells |
| Swing trading | Days to weeks | Trends, news drift | Gaps against you overnight |
| Intraday | Minutes to hours | Small intraday edges | Costs and noise |
| Scalping | Seconds to minutes | Order flow, speed | Costs; professionals with faster systems |
| Option selling | Days | Volatility risk premium | Rare large losses |

The shorter the holding time, the bigger costs are relative to the move you catch, and the
more you compete with professional firms that have faster systems and lower costs.
""")
    sec("Building a trading plan", """
Write it down before you trade:
1. **Markets and instruments**: which stocks or contracts, and which never.
2. **Setup**: the exact conditions for a trade, and why they should work.
3. **Entry**: when and how (limit or market).
4. **Exit**: stop-loss, target and a time exit (for example, "close by 15:15 regardless").
5. **Size**: from the stop distance and the risk per trade (calculator below).
6. **Limits**: maximum loss per day and per week, and maximum open trades.
7. **Review**: a journal with every trade, its reason and the result, reviewed weekly.
8. **Evidence bar**: how many trades before you judge the plan, and what result means stop.

A plan you cannot write down is a feeling, not a plan.
""")
    with sec("Position sizing calculator"):
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
- **Risk 0.5–1% of capital per trade.** Ten losses in a row, which will happen, then cost
  about 10%, not everything.
- **Size from the stop, not from conviction.** A wider stop means fewer shares.
- **Cap the total.** Limit open risk and set a daily loss limit. Every paper book here stops
  for the day at −1.5%.
""")
    sec("Stop-losses: kinds and placement", """
- **Fixed %**: simple. It ignores how much the stock normally moves.
- **ATR-based** (for example 1.5× the average daily range): adapts to each stock's
  volatility. Usually better.
- **Structure-based**: below a support level or the opening range low. Logical, but obvious
  levels attract stop-hunting.
- **Time stop**: exit if the trade hasn't worked by a set time. Intraday books here always
  close by 12:30 or 15:15.
- **Trailing stop**: moves up as the price rises, locking in gains.
- **Stop to cost**: once a trade is well in profit, move the stop to the entry price. The
  option-selling books do this after one leg stops.

Stops can be jumped by gaps (overnight or on news) and fill worse than the trigger. Size for
that.
""")
    with sec("Expectancy calculator: does the strategy make money?"):
        c1, c2, c3, c4 = st.columns(4)
        win = c1.slider("Win rate (%)", 10, 90, 45)
        aw = c2.number_input("Average win (₹)", 1.0, 1_000_000.0, 600.0)
        al = c3.number_input("Average loss (₹)", 1.0, 1_000_000.0, 400.0)
        cost = c4.number_input("Costs per trade (₹)", 0.0, 10_000.0, 35.0)
        e = win / 100 * aw - (1 - win / 100) * al - cost
        be = (al + cost) / (aw + al) if aw + al else 0
        UI.cards([UI.card("Expectancy per trade", UI.money(e), "average result after costs", 0.0,
                          "up" if e > 0 else "down"),
                  UI.card("Over 250 trades", UI.money(250 * e), "about a year of daily trading", 0.0,
                          "up" if e > 0 else "down"),
                  UI.card("Break-even win rate", f"{be:.0%}", "with these wins, losses and costs", be)])
        st.caption("A 40% win rate is fine if winners are twice the size of losers. A 70% win rate loses "
                   "money if losses are four times the wins: the trap in option selling. And costs "
                   "decide whether a thin edge survives.")
    with sec("Drawdowns: why losses matter more than gains"):
        loss = st.slider("A fall of", 5, 90, 30, 5, format="%d%%")
        need = 1 / (1 - loss / 100) - 1
        st.markdown(f"needs a gain of **{need:.0%}** to get back to where you started.")
        x = np.arange(5, 91, 5)
        df = pd.DataFrame({"Fall": x / 100, "Gain needed": 1 / (1 - x / 100) - 1})
        st.altair_chart(alt.Chart(df).mark_line(point=True, color=UI.NEUTRAL).encode(
            x=alt.X("Fall:Q", axis=alt.Axis(format="%")),
            y=alt.Y("Gain needed:Q", axis=alt.Axis(format="%"), scale=alt.Scale(type="log"))
        ).properties(height=240), width="stretch")
        st.caption("A strategy's deepest fall (maximum drawdown) matters as much as its return: few "
                   "people keep following a strategy through a 50% drawdown.")
    sec("Diversification and correlation", """
- Five bank stocks are close to one big bet on banks. **Count independent bets, not
  stocks.**
- Correlations rise in crashes: everything falls together when it matters most.
- Mixing styles helps more than adding stocks: momentum with value, long-term with intraday,
  equity with some bonds or gold.
- The strategy lab is diversification across ideas. Strategies that make money on different
  days smooth the whole.
""")
    sec("Portfolio construction and rebalancing", """
- **Equal weight**: simple, and avoids over-betting on one idea. The long-term book holds 5
  stocks equally.
- **Volatility weighting**: give calmer stocks more money so each contributes similar risk.
- **Sector caps**: limit how much one sector can take.
- **Rebalancing**: on a schedule (weekly for the long-term book), sell what fell out of the
  top ranks and buy the new leaders. Not too often: each change costs.
- **Cash**: uninvested cash is drag in a rising market and a cushion in a falling one.
  Decide deliberately how much to hold.
""")
    sec("Psychology: the mistakes everyone makes", """
- **Loss aversion**: a loss hurts about twice as much as an equal gain pleases (Kahneman and
  Tversky). So people hold losers, hoping, and sell winners early: the disposition effect.
- **Overconfidence**: after a few wins people trade bigger and more often. Trading volume and
  returns are negatively related for individuals (Barber and Odean 2000).
- **Revenge trading**: trying to win back a loss the same day, usually with larger size.
- **Confirmation bias**: reading only the news that agrees with your position.
- **Recency**: expecting the last few weeks to continue forever.
- **FOMO**: buying because everyone else is, usually near tops.

**What helps**
- Write the plan before the trade and follow it.
- Keep a journal of reasons and feelings.
- Set hard limits.
- Let rules, or a tested system, do the trading.
""")
    sec("From paper to real money: what changes", """
- **Fills**: paper fills at the last price; real orders face the spread, queue position and
  partial fills. Compare paper with what the real fills would have been. This system checks
  paper against simulation every evening.
- **Emotion**: losing real money changes behaviour; start smaller than you think.
- **Costs**: real contract notes include every charge; reconcile them with the paper model.
- **Operational risk**: internet drops, app outages, a forgotten MIS position. Use stop
  orders at the broker, not only in your head.
- **Rule of thumb**: go live only with strategies that beat their costs on paper with a
  t-statistic above 2 over 60+ trading days, and at a fraction of the final size.
""")
    sec("When to stop a strategy", """
Decide in advance:
- A maximum drawdown (for example 1.5× the worst in testing) that switches it off.
- A minimum sample before judging (20+ days for daily strategies; more is better).
- A check that the live results resemble the backtest. A big gap means the backtest was
  wrong, not that the market is "unlucky".

Every strategy has losing streaks. The skill is telling a normal bad run from a broken
edge, and the only way is numbers set before the run started.
""")


# --- 5. Futures and options -------------------------------------------------------------------

def derivatives() -> None:
    st.markdown("Futures, options, the Greeks, option selling, open interest, and the SEBI "
                "rules that govern them.")
    sec("Futures", """
A future is an agreement to buy or sell at a fixed price on an expiry date.
- Stock and index futures expire on the last Tuesday of the month.
- They trade in **lots**. Nifty is 65 units, so one lot at 26,000 is about ₹17 lakh of
  exposure.
- You put up a **margin** of roughly 12–20% for stocks (less for the index). That is
  **leverage**: a 5% move can be a 25–40% gain or loss on your margin.
- Profit and loss are settled daily (**mark-to-market**); losses must be covered the next
  morning.
- **Basis** = future price − spot. Usually positive (the cost of carry); it shrinks to zero
  at expiry.
- **Rollover**: closing this month's contract and opening next month's.

Futures are a cheap way to trade direction (low costs, easy shorting) and a fast way to lose
money (leverage).
""")
    sec("Margins: SPAN, exposure, peak and MTM", """
- **SPAN margin**: the exchange's estimate of a position's one-day worst loss.
- **Exposure margin**: an extra buffer on top.
- **Peak margin**: margin is checked at random moments during the day; you must have the
  full margin at all times, not only at the close.
- **Option buyers** pay the full premium upfront. **Option sellers** post SPAN + exposure,
  often ₹1–2 lakh per Nifty lot, much less when hedged with bought wings.
- **MTM**: futures losses are debited daily. A big move can call for more money than you
  have, and the broker will close the position.
""")
    with sec("Options and their payoffs"):
        st.markdown("""
- A **call** is the right to buy at the strike. A **put** is the right to sell at the strike.
- The **buyer** pays the premium and can lose at most that. The **seller** receives it and
  can lose much more.
- **Premium = intrinsic value + time value.** Time value decays to zero by expiry: slowly at
  first, fast in the last days.
- **Moneyness**:
  - At the money (ATM): strike near the price
  - In the money (ITM): has intrinsic value
  - Out of the money (OTM): only time value
""")
        kind = st.segmented_control("Payoff of", ["Buy a call", "Sell a call", "Buy a put", "Sell a put"],
                                    default="Buy a call", key="payoff_kind")
        strike, prem = 26000, 150.0
        x = np.arange(25000, 27001, 10)
        call, put = np.maximum(x - strike, 0) - prem, np.maximum(strike - x, 0) - prem
        y = {"Buy a call": call, "Sell a call": -call, "Buy a put": put, "Sell a put": -put}[kind or "Buy a call"] * 65
        st.altair_chart(alt.Chart(pd.DataFrame({"Nifty at expiry": x, "Profit or loss (₹)": y}))
                        .mark_line(color=UI.NEUTRAL).encode(
                            x=alt.X("Nifty at expiry:Q", scale=alt.Scale(zero=False)),
                            y="Profit or loss (₹):Q").properties(
                                height=220, title=f"{kind}: strike 26,000, premium ₹150, 1 lot of 65"),
                        width="stretch")
    sec("The Greeks", """
| Greek | Measures | In practice |
|---|---|---|
| **Delta** | Premium change for a ₹1 move in the underlying | ATM ≈ 0.5; also a rough chance of expiring ITM |
| **Gamma** | How fast delta changes | Highest ATM near expiry; makes expiry days violent |
| **Theta** | Time decay per day | The buyer pays it, the seller earns it |
| **Vega** | Sensitivity to implied volatility | Options get pricier when fear rises |
| **Rho** | Sensitivity to interest rates | Small for short-dated options |
""")
    sec("Implied volatility, VIX and how options are priced", """
- An option's price depends on the underlying's price, the strike, time to expiry, interest
  rates and **implied volatility (IV)**: how much movement the market expects.
- The Black-Scholes model turns these into a price. Traders use it backwards: from the market
  price they read the IV.
- **India VIX** is the Nifty's 30-day IV. It usually sits above the volatility that actually
  follows. That gap is the volatility risk premium that option sellers collect.
- **IV crush**: after an event (results, the Budget), IV falls sharply. Options bought for
  the event lose value even if the stock moves.
- **Skew**: out-of-the-money puts usually carry higher IV than calls, because protection
  against crashes is in demand.
""")
    sec("Strategies, from simple to hedged", """
| Strategy | How | When | Risk |
|---|---|---|---|
| Covered call | Own shares, sell a call | Flat to slightly rising | Gains capped |
| Protective put | Own shares, buy a put | Insurance | Premium cost |
| Bull call spread | Buy a call, sell a higher call | Moderately bullish | Limited both ways |
| Bear put spread | Buy a put, sell a lower put | Moderately bearish | Limited both ways |
| Long straddle | Buy ATM call + put | Expect a big move either way | Time decay |
| Short straddle | Sell ATM call + put | Expect calm | Unlimited |
| Short strangle | Sell OTM call + put | Expect a range | Unlimited |
| Iron fly / condor | Short straddle / strangle + bought wings | Range, capped risk | Limited |
| Calendar spread | Sell near expiry, buy far | Calm now, movement later | Moderate |

Hedged structures (spreads, iron flies, condors) need far less margin and cannot blow up an
account. That is why every option-selling book here is hedged.
""")
    with sec("Option selling"):
        st.markdown("""
**Why it can work.** Index options are usually priced for more movement than happens: the
volatility risk premium (Coval and Shumway 2001; Bakshi and Kapadia 2003). Nifty's VIX
typically sits 2–4 points above the volatility that follows.

**Why it is dangerous.** Gains are small and frequent; losses are rare and large. A few
big-move days decide the year. One public backtest of a time-based Nifty straddle made
₹19,405 in a year, but −₹94,377 without its 25 best days.

**The paper books.** Five hedged structures run side by side on the *Option selling* page:
- 9:20 iron fly
- 11:00 iron fly
- Iron condor with a 50% profit target
- The same condor, but only when VIX is above realized volatility
- Expiry-day iron fly
""")
        x = np.arange(25200, 26801, 10)
        credit, lot = 252.0, 65
        pnl = (credit - np.abs(x - 26000) + np.maximum(np.abs(x - 26000) - 500, 0)) * lot
        st.altair_chart(alt.Chart(pd.DataFrame({"Nifty at the close": x, "Profit or loss (₹)": pnl}))
                        .mark_area(opacity=0.25, color=UI.NEUTRAL, line={"color": UI.NEUTRAL}).encode(
            x=alt.X("Nifty at the close:Q", scale=alt.Scale(zero=False)),
            y="Profit or loss (₹):Q").properties(
                height=220, title="Iron fly at 26,000: wings 500 points away, ₹252 credit, 1 lot"),
            width="stretch")
    sec("Expiry-day dynamics", """
- Time value collapses through the day; ATM options can lose most of their value by 14:00.
- **Gamma is extreme**: a 0.5% move can multiply an ATM option's price.
- Prices often **pin** near strikes with the largest open interest, then can break sharply in
  the last hour.
- SEBI charges extra margin (2% ELM) on short options on expiry day. Calendar spreads lose
  their margin benefit that day.
- Most retail losses in options happen on expiry days, buying cheap OTM options that expire
  worthless.
""")
    sec("Open interest and the put-call ratio", """
**Open interest (OI)** is the number of contracts still open. Read it together with price:

| Price | OI | Usual reading |
|---|---|---|
| Up | Up | **Long build-up**: new buyers, strength |
| Down | Up | **Short build-up**: new sellers, weakness |
| Up | Down | **Short covering**: sellers exiting; the rise may fade |
| Down | Down | **Long unwinding**: buyers exiting; the fall may fade |

**Put-call ratio (PCR)** = put OI ÷ call OI. A very high PCR is often read as too much fear
(contrarian bullish). It is noisy on its own.

**This system's evidence.** OI changes, build-up, futures basis and PCR, as features in the
intraday model, improved its daily ranking (IC +0.0072, t = 9.0 over 2,433 days). "Max pain"
and single-day OI levels have little evidence behind them.
""")
    sec("SEBI's F&O rules (2024–26)", """
SEBI tightened index derivatives from late 2024 to curb retail losses:
- **Bigger contracts**: a minimum contract value of ₹15 lakh, so lot sizes went up.
- **One weekly expiry per exchange**: the Nifty on the NSE (Tuesday), the Sensex on the BSE.
- **Upfront premium**: option buyers pay the full premium upfront.
- **Expiry-day margin**: an extra 2% margin on short options on expiry day.
- **No calendar-spread benefit** on expiry day.
- **Intraday monitoring** of position limits.

Lot sizes are revised periodically: check the current NSE circular before trading. Taxes:
F&O profits are business income at your slab rate; see *Tax on gains* in Part 1.
""")


# --- 6. Algos and this system -------------------------------------------------------------------

def algos() -> None:
    st.markdown("How systematic trading is done properly, how a professional desk is "
                "organised, and how this system works day to day.")
    sec("What an algorithm is", """
An algorithm is a rule a computer follows exactly: what to buy or sell, how much, and when to
get out. The value is not speed (retail cannot compete on speed) but **discipline and
measurement**:
- Every decision is recorded.
- Every result can be counted.
- Emotions are out of the loop.

Writing a rule is easy. Knowing whether it has an edge, after costs and out of sample, is
the whole job.
""")
    sec("Data: the raw material", """
| Data | Used for | In this system |
|---|---|---|
| Daily prices (OHLC, volume) | Trends, volatility, momentum | Since 2005, split-adjusted |
| Intraday bars (1–5 minute) | First-30-minute features, exits | About 2,500 days from Angel One |
| Delivery % | Conviction behind volume | Daily NSE files |
| F&O bhavcopy (OI, basis) | Positioning | Since 2012 |
| Results calendar and reactions | Earnings effects | Yes |
| FPI / DII flows | Market direction | Daily |
| News | Events, sentiment | Company + 15 sector themes, scored |
| Fundamentals | Quality and value | Snapshots |
| Corporate actions | Adjusting prices and positions | Splits, bonuses, dividends |

**Data problems that ruin backtests**
- **Survivorship bias**: only today's stocks.
- **Look-ahead**: using information before it was public.
- **Unadjusted splits.**
- **Wrong timestamps.**
""")
    sec("Features and labels", """
- A **label** is what you want to predict. Here, for each stock and day: did it beat the
  market from 9:45 to 12:30 (or to the close, or over the next week)?
- **Features** are what you know at decision time: the first 30 minutes' move, gap, range,
  volume vs normal, VWAP distance, sector and market moves, overnight global cues, recent
  trend, F&O positioning, news tone.
- **Relative features beat absolute ones.** A 2% rise means little on a day everything
  rose 2%.
- Every feature must be computable **at the moment of the decision**, with no look-ahead.
""")
    sec("Models", """
- **Linear models**: simple and robust; good baselines.
- **Gradient-boosted trees (LightGBM)**: this system's main model. They handle non-linear
  effects and interactions and tolerate missing data. Research on many stock markets
  (Gu, Kelly and Xiu 2020) finds trees and neural networks beat linear models, mostly
  through interactions.
- **Ranking, not forecasting.** The model ranks stocks against each other each day. Its
  skill is measured by the **IC** (information coefficient): the correlation between
  predicted and actual ranks. An IC of 0.03–0.05 is useful in stock selection; 0.1 is
  exceptional.
- **Ensembles**: averaging several models (different seeds, the 12:30 and close models
  blending each other) is more stable than any one.
""")
    sec("Testing properly", """
1. **Walk-forward**: train on the past, test on the next period, roll forward. Never test on
   data the model saw.
2. **Purging and embargo**: drop training rows whose labels overlap the test period (López
   de Prado).
3. **Combinatorial purged cross-validation (CPCV)**: many train/test splits, so one lucky
   period doesn't decide. This project requires an idea to win in at least 80% of paths.
4. **Multiple-testing correction**: try 20 settings and one will look great by chance. The
   bar (a paired t-statistic of at least 2) rises with the number of settings tried. The
   *deflated Sharpe ratio* formalises this.
5. **Costs and realistic fills** in every test.
6. **Paper trading** on live prices before any real money.

**Traps this project hit, with numbers**
- **Survivorship**: adding delisted stocks cut the long-term backtest from 21.6% to 12.4% a
  year.
- **Learning from its own mistakes overfits**: replays that weighted wrong picks more made
  results worse (₹−386 → ₹−413 a day).
- **New information beat cleverer models**: ten price-only ideas failed before F&O
  positioning data clearly helped (t = 9.0).
""")
    sec("Costs and execution", """
- **Model costs honestly**: brokerage (with minimums), taxes, charges, spread and slippage.
  Here: about 0.37% per intraday round trip.
- **Edge per trade must beat costs.** An average move in your favour of 0.2% with 0.37% costs
  loses money however accurate the direction calls are.
- **How professionals cut costs**:
  - Fewer, larger trades (minimum charges matter less)
  - Limit orders instead of market orders
  - Trading the most liquid stocks
  - Futures instead of cash for short-term trades (STT is lower)
  - Negotiated brokerage
- **Execution algorithms** (VWAP, TWAP) split big orders over time to avoid moving the
  price. Retail sizes rarely need them.
""")
    sec("The risk engine", """
Every professional desk has a risk layer that the strategy cannot override:
- **Kill switch**: stale data, broken predictions, or a book's weekly loss limit stops new
  trades.
- **Daily loss limit**: −1.5% of a book's capital closes it for the day.
- **Position limits**: at most 3–5 stocks a side, at most the capital (no hidden leverage).
- **Event filters**: results days and expiry days can be skipped by the learned rules.
- **Split protection**: a sudden 40%+ jump or drop in a held stock is treated as a probable
  corporate action, not a stop.
- **Monitoring**: paper fills are compared with the simulation every evening, and alerts go
  to the dashboard.
""")
    sec("How a professional trading desk works", """
A trading firm is organised so that no single good idea or bad day decides its fate:

1. **Research** generates many ideas from data and papers, tests them with strict rules, and
   kills most of them. This is the *experiment log*: E-001 to E-013 so far.
2. **Portfolio of strategies.** Many small, different edges (the *Strategy lab*), with money
   moved towards what works and away from what stops working, by rules set in advance.
3. **Execution** minimises costs: who you trade with, how, when, at what size.
4. **Risk** sets limits per trade, per strategy, per day, independent of the researchers.
5. **Attribution.** Every rupee of profit or loss is explained: market, sector, stock
   picking, costs. See *Where the money went* on the Strategy lab page.
6. **Monitoring and post-mortems.** Every loss day is reviewed; every live-vs-backtest gap
   is explained.

This system follows the same structure at small scale. The current gap it reports: stock
picking before costs is close to zero in the intraday books, and costs (0.37% per round
trip) are larger than the edge. So the work now is better information (news, positioning) and
lower costs, not more trades.
""")
    sec("How this system works, in one day", """
| Time | Where | What |
|---|---|---|
| 8:30 | GitHub | Overnight company and sector news fetched and scored |
| 9:15 – 9:45 | Mac | Live prices every minute; the first 30 minutes recorded |
| 9:20 – 11:00 | Mac | Option-selling books enter |
| 9:46 | Mac | Models rank every stock; 25 buys + 25 sells saved and judged; 3–5 a side traded after the news, sector and market checks; lab books trade their own rules |
| All day | Mac | Stops, targets and loss limits checked every minute |
| 12:30 / 15:15 | Mac | Books square off; every pick judged |
| 14:55 – 15:25 | Mac | Nifty last-half-hour momentum book |
| 15:40 | Mac | Paper results to GitHub |
| 18:37 | GitHub | Market, F&O, delivery, results and IPO data updated |
| 21:07 | GitHub | Nightly training (below) |
| ~04:00 | GitHub | US book after the US close |

**Every night on GitHub**
1. Retrain each model on all history up to that day, plus every judged paper pick.
2. Tune the trading rules (stop, target, number of trades) by paper profit on recent history.
3. Replay history as paper trading, keeping lessons only if they improve profit and accuracy.
4. Try new model settings; adopt them only if they win on data they never saw.
""")
    sec("The strategy lab", """
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
""")
    sec("SEBI's rules for retail algos", """
From 2025, SEBI requires algorithms that place orders through a broker's API to be:
- registered with the exchange through the broker, with a unique algo ID;
- run with the broker's risk checks.

Strategies whose logic is not disclosed ("black box") need the provider to register as a
research analyst. Personal, order-by-order use of your own logic within a broker's order-rate
limits is treated more lightly. This system only paper trades, so none of this applies yet;
check the current circular before automating real orders.
""")
    sec("Reading list", """
**Papers**
- Coval and Shumway (2001), [Expected Option Returns](https://business.baylor.edu/don_cunningham/Option%20Returns.pdf), Journal of Finance.
- Bakshi and Kapadia (2003), *Delta-Hedged Gains and the Negative Market Volatility Risk Premium*, Review of Financial Studies.
- Jegadeesh and Titman (1993), *Returns to Buying Winners and Selling Losers*, Journal of Finance.
- Bernard and Thomas (1989), *Post-Earnings-Announcement Drift*, Journal of Accounting Research.
- Ritter (1991), *The Long-Run Performance of Initial Public Offerings*, Journal of Finance.
- Tetlock (2007), *Giving Content to Investor Sentiment*, Journal of Finance.
- Savor (2012), *Stock Returns after Major Price Changes: The Role of News*, Journal of Financial Economics.
- Gao, Han, Li and Zhou (2018), [Market Intraday Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866), Journal of Financial Economics.
- Lou, Polk and Skouras (2019), [A Tug of War: Overnight Versus Intraday Expected Returns](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650), Journal of Financial Economics.
- Gu, Kelly and Xiu (2020), [Empirical Asset Pricing via Machine Learning](https://dachxiu.chicagobooth.edu/download/ML.pdf), Review of Financial Studies.
- Barber and Odean (2000), *Trading Is Hazardous to Your Wealth*, Journal of Finance.

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
- NSE corporate announcements, shareholding patterns and F&O circulars at [nseindia.com](https://www.nseindia.com).

**Practitioner material**
- AlgoTest, [The 9:20 straddle strategy](https://algotest.in/blog/the-920-straddle-strategy-defined/).
- Option Alpha, [0DTE strategies: insights from 25k trades](https://optionalpha.com/blog/0dte).
""")


# --- Glossary ------------------------------------------------------------------------------

GLOSSARY = {
    "Alpha": "Return beyond what the market (or a benchmark) gave; the part due to skill.",
    "Anchor investors": "Institutions that buy part of an IPO the day before it opens, with a lock-in.",
    "ASM / GSM": "Exchange surveillance lists for unusually volatile or suspicious stocks.",
    "ATR": "Average true range: a stock's typical daily range.",
    "Basis": "Future price minus spot price.",
    "Beta": "How much a stock moves with the market (1 = same as the market).",
    "Bid / ask": "Best price a buyer will pay / best price a seller will accept.",
    "Block deal": "A large trade (₹10 crore+) in a special window, reported the same day.",
    "Bonus issue": "Free extra shares to existing holders; the price adjusts down.",
    "Book value": "Assets minus liabilities: shareholders' equity.",
    "CAGR": "Compound annual growth rate.",
    "CASA": "Current and savings account deposits: a bank's cheapest funding.",
    "Circuit": "The daily price limit of a stock.",
    "CNC / MIS / NRML": "Delivery / intraday / carried-forward F&O product types.",
    "Cost of carry": "The interest cost reflected in a future's premium over spot.",
    "Delivery %": "Share of traded volume taken into demat accounts.",
    "Delta": "Option price change per ₹1 move in the underlying.",
    "Drawdown": "Fall from a previous peak.",
    "EBITDA": "Earnings before interest, tax, depreciation and amortisation.",
    "EPS": "Earnings per share.",
    "Ex-date": "From this date the buyer no longer gets the dividend, split or bonus.",
    "Expectancy": "Average profit per trade after costs.",
    "FPI / DII": "Foreign portfolio investors / domestic institutional investors.",
    "Free float": "Shares available to trade (excluding promoters' and locked-in holdings).",
    "Gamma": "How fast an option's delta changes.",
    "Gap": "The difference between today's open and yesterday's close.",
    "GMP": "Grey-market premium for an IPO: unofficial and unreliable.",
    "HNI / NII": "High-net-worth / non-institutional IPO investors (bids above ₹2 lakh).",
    "IC": "Information coefficient: correlation of predicted and actual ranks.",
    "Implied volatility": "The volatility an option's price assumes.",
    "Iron condor / fly": "Short strangle / straddle with bought wings that cap the loss.",
    "Leverage": "Trading more than your money through margin or derivatives.",
    "Lot size": "The minimum quantity of an F&O contract.",
    "Market cap": "Share price × number of shares.",
    "Mark-to-market": "Daily settlement of futures profits and losses.",
    "NIM": "Net interest margin of a lender.",
    "NPA": "Non-performing asset: a loan not being repaid.",
    "OFS": "Offer for sale: existing shareholders selling (in an IPO or later).",
    "Open interest": "Number of open F&O contracts.",
    "P/E, P/B, EV/EBITDA": "Valuation multiples: price vs earnings, book value, operating profit.",
    "PCR": "Put-call ratio of open interest.",
    "Pledge": "Shares given as collateral for a loan (by promoters, a warning sign).",
    "Promoter": "The founder or controlling group of a listed company.",
    "QIB": "Qualified institutional buyers: mutual funds, FPIs, banks, insurers.",
    "Record date": "The date that decides who gets a dividend or corporate action.",
    "RHP": "Red herring prospectus: the IPO offer document.",
    "ROCE / ROE": "Return on capital employed / on equity.",
    "Rollover": "Moving a futures position to the next expiry.",
    "Sharpe ratio": "Return per unit of volatility (annualised).",
    "Slippage": "Difference between the expected and the actual fill price.",
    "SPAN": "The exchange's margin model for F&O.",
    "Split": "One share divided into several; value unchanged.",
    "Spread": "Gap between the best bid and ask.",
    "STT": "Securities transaction tax.",
    "Survivorship bias": "Testing only on stocks that still exist, hiding the failures.",
    "Theta": "Daily time decay of an option.",
    "t-statistic": "How many standard errors a result is from zero; above 2 is unlikely to be luck.",
    "Vega": "Option price change per 1-point change in implied volatility.",
    "VIX": "Expected 30-day volatility from option prices (India VIX for the Nifty).",
    "VWAP": "Volume-weighted average price of the day.",
    "Walk-forward": "Testing on each period with a model trained only on earlier periods.",
}


def glossary() -> None:
    q = st.text_input("Search the glossary", placeholder="e.g. delta, QIB, drawdown",
                      label_visibility="collapsed")
    rows = [(k, v) for k, v in GLOSSARY.items()
            if not q or q.lower() in k.lower() or q.lower() in v.lower()]
    st.dataframe(pd.DataFrame(rows, columns=["Term", "Meaning"]), hide_index=True,
                 width="stretch", height=min(38 * (len(rows) + 1), 1400),
                 column_config={"Meaning": st.column_config.TextColumn(width="large")})
