# IPO Trading Resources & Learning

## Repositories (Open Source)

### Direct IPO Trading
- **NSEPython / NSE Data** — Indian IPO data, GMP tracking, allotment status
  - https://github.com/NSEPython/NSEPython (license: MIT)
  - Provides NSE IPO calendar, GMP history, subscription tracking

### Portfolio & Risk Management (for IPO position sizing)
- **PyPortfolioOpt** (already studied) — position sizing, risk constraints
- **cvxportfolio** (already studied) — trading costs in portfolio context
- **Riskfolio-Lib** (already studied) — risk budgeting for positions

### Market Microstructure (for first-print analysis)
- **pylob** — limit order book simulation, volume analysis
  - https://github.com/neutronst/pylob (MIT)
- **orderbook** — real-time order book analysis
  - https://github.com/bmoscon/orderbook (Apache-2.0)

### Sentiment & Social Data
- **TextBlob / VADER** — sentiment from news, Twitter IPO discussions
  - Part of NLTK; tracks IPO hype during quiet period
- **Tweepy** — track IPO mentions on Twitter (if doing sentiment)

## Books & Resources

1. **"Malkiel's Random Walk Down Wall Street"** — chapter on IPOs (survivorship, pricing)
   - Why most retail traders lose on IPOs (overpriced, hype-driven)

2. **"Market Microstructure Theory"** (O'Hara) — limit orders, first-print mechanics
   - How order book dynamics affect IPO opening moments

3. **Indian IPO Guides**
   - **"How to Invest in IPOs"** (Zerodha, free blog series)
   - **NSE Official IPO Manual** (NSE website) — technical rules, allotment, settlement
   - **BSE IPO Guidelines** (same)

4. **Case Studies**
   - Flipkart IPO (2023, delayed, then strong open) — sentiment vs reality
   - Zomato IPO (2021, strong open, then fade) — GMP accuracy
   - Nykaa IPO (2021, huge pop, later correction) — typical arc

## What to Track Before Trading an IPO

### During Quiet Period (no IPO marketing)
- [ ] **Business fundamentals:** revenue, growth, unit economics, margins
- [ ] **Peer comparison:** valuations vs similar listed companies (P/E, EV/sales)
- [ ] **Competitive position:** market share, moat, growth rate vs peers
- [ ] **Management quality:** founder track record, board composition
- [ ] **Sector health:** tailwinds (e.g., fintech growth) or headwinds (e.g., coal)

### Right Before Listing (T-3 to T-1)
- [ ] **Subscription data:** final alloc ratio (released D-1)
- [ ] **GMP trajectory:** is GMP rising or falling? (social channels, forums, news)
- [ ] **Nifty/Sector momentum:** is the market rally intact?
- [ ] **Recent IPO comps:** similar IPOs from past 6 months — did they pop or fade?
- [ ] **Options implied vol:** what's the market pricing for listing-day moves?

### Listing Day (T)
- [ ] **First 30 seconds:** at 09:15, check first-print volume (need >=5x lot size)
- [ ] **First 5 minutes:** does price hold GMP or fade? Quick momentum signal
- [ ] **Volume profile:** is volume sustained or dying? (illiquid = can't exit)

## IPO Trading Strategy Template

### Entry
```
At 09:15 (listing open):
  IF first-print volume > 5x lot_size AND price >= 95% GMP:
    Buy 1 lot market (or limit 1-2% above GMP)
  ELSE:
    Skip this IPO
```

### Exit
```
Target 1: +5% within 5 mins → Exit half (lock in gains)
Target 2: +8% within 10 mins → Exit rest
Loss stop: -2% → Exit immediately (IPO fading)
Time stop: If no +5% by T+15 mins → Exit (momentum lost)
```

### Position Size
```
Max per IPO: 2-3 lots (test 1 lot first, add if conviction)
Max portfolio allocation: 10% at any time (IPOs are risky short-term bets)
```

## Known IPO Trading Pitfalls

- **Hype vs reality:** GMP can be wildly optimistic; listen to institutional demand (alloc ratio)
- **First-print illiquidity:** if volume dies, you're trapped (can't exit at market)
- **Sector/market timing:** IPO pops in rallies, fades in corrections
- **Overallocation:** retail traders buy too much; professional traders exit quickly → dump
- **Tax implications:** short-term gains on IPOs are taxed as income (no indexation benefit like stocks)

## Next Steps for Implementation

1. **Integrate NSEPython** — live IPO calendar, alloc ratio, GMP data
2. **Build first-print volume monitor** — at 09:15, fetch NSE live data
3. **Sentiment tracker** — track GMP trajectory from forums (if automated)
4. **Backtest recent IPOs** — validate entry/exit rules on 2021-2024 IPO history
5. **Paper trade** — live track 5-10 IPOs before real money
