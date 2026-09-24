# Stock Predictor — Project Plan

Personal-use AI stock predictor for the Indian market (Nifty 100).
Everything runs **locally on my laptop** and costs **₹0** (no paid APIs, no cloud).

---

## 1. Goals

- Train on historical Nifty 100 data (daily + intraday).
- **Intraday:** ~30 minutes after market open (9:45 AM), predict **5 stocks that will go up** and **5 that will go down** for the day.
- **Long-term:** weekly picks to buy / hold / exit, holding period 1–3 months.
- Use candle patterns, technical indicators, strategy signals, market context and **company news** as model inputs.
- Paper-trade both strategies and track **accuracy** and **profit/loss** honestly (after costs).
- Simple UI + Telegram alerts.

## 2. Key decisions

| Topic | Decision |
|---|---|
| Universe | Nifty 100 |
| Intraday prediction time | 9:45 AM (uses first 30 min of trading) |
| Intraday holding | Entry ~9:45, exit by ~3:15 PM (or stop-loss / target) |
| Long-term | Weekly predictions, 1–3 month horizon |
| Broker (real trades) | Groww — trades entered **manually** in the app |
| Market data | **Free** real-time API (Angel One SmartAPI — data only), plus yfinance / NSE for daily history |
| Paper capital | ₹1,00,000 intraday + ₹1,00,000 long-term |
| Real capital | To be set later in Settings |
| News analysis | Free, local (FinBERT) |
| Alerts | Telegram bot (free) |
| Runs on | My laptop for training, live monitoring and the app (catches up if opened late) |
| Data storage | Market data on git (`market-data` branch), updated daily by free GitHub Actions; Mac keeps only a shallow snapshot |
| Long-term decisions | Official buy / hold / exit **daily after close**; live monitoring all day; emergency exits on stop-loss or strongly negative news |
| Live tracking | Paper **and** real (Groww, manual) portfolios monitored live with stop-loss alerts |
| Cost | ₹0 |

## 3. Data sources (all free)

- **Angel One SmartAPI** (free with an Angel One account, used only for data):
  - Real-time live ticks via WebSocket (no delay)
  - Historical 1-min / 5-min / daily candles
- **yfinance / NSE bhavcopy:** long daily history, backup source.
- **NSE:** holiday calendar, corporate announcements, results, FII/DII data, India VIX.
- **News:** Google News RSS per company, Moneycontrol / Economic Times RSS, NSE announcements, GDELT (historical).

## 4. Model inputs (features)

| Group | Examples |
|---|---|
| Candle patterns | 60+ TA-Lib patterns (engulfing, hammer, doji, morning star, …) |
| Technical indicators | RSI, MACD, Bollinger Bands, VWAP, ATR, moving averages |
| Strategy signals | Momentum, mean reversion, gap-fill, opening-range breakout |
| First 30 min (intraday) | 9:15–9:45 return, volume vs average, range, position vs VWAP |
| Market context | Nifty & sector index moves, India VIX, GIFT Nifty gap, FII/DII flows |
| News | FinBERT sentiment, news count, results / event-day flag |
| Fundamentals (long-term) | P/E, ROE, profit growth — where free data is available |

## 5. Models

Two separate **LightGBM ranking** models (train on CPU in minutes):

| | Intraday | Long-term |
|---|---|---|
| Runs | Daily 9:45 AM | Weekly |
| Target | Return 9:45 → 3:15 | Return over 1–3 months |
| Output | Top 5 up, top 5 down | Buy / hold / exit list |

Rules:
- **Walk-forward validation** only (train on past, test on the following period) — no look-ahead bias.
- **Realistic costs** in every backtest and P&L: brokerage, STT, exchange charges, GST, stamp duty, slippage.
- **Confidence threshold** — the model can say "skip today" (budget day, extreme VIX, etc.).
- **Stop-loss / target** per pick based on the stock's volatility (ATR).
- **Automatic weekly retraining.**
- Every pick shows **why** it was chosen (top contributing signals).

## 6. App (Streamlit, local)

1. **Intraday — Today's Picks:** 5 up / 5 down, confidence, reasons, stop-loss/target, related news.
2. **Long-term Picks:** buy / hold / exit with reasons.
3. **Paper Trading** (separate Intraday & Long-term tabs, ₹1 lakh each):
   prediction vs reality table, P&L after costs, cumulative P&L chart.
4. **My Portfolio** (separate Intraday & Long-term tabs):
   manual entry of real Groww trades, live prices, invested amount, realized / unrealized P&L.
5. **Accuracy:**
   - Direction accuracy (overall, up-picks, down-picks)
   - % of trades profitable after costs
   - Rolling 30-day accuracy trend
   - Accuracy by confidence level
   - Comparison vs a random-pick baseline
6. **Model:** backtest results, feature importance, last retrain date.
7. **Settings:** capital amounts, stop-loss %, confidence threshold, Telegram setup.

## 7. Automation & live design

**Long-term — live monitoring, disciplined decisions**

| Layer | Frequency | What it does |
|---|---|---|
| Live monitor | Every 1 min (market hours) | Live prices, paper + real portfolio P&L, stop-loss / target hit → immediate exit alert |
| Live scores | Every 15 min | Re-score all Nifty 100 on live prices; rank movement shown as *provisional* |
| News watch | Every 15 min | New headlines → FinBERT; strongly negative news on a held stock → review / exit alert |
| Official decisions | Daily after close | Final buy / hold / exit on complete data; paper portfolio trades on these |
| Data update | Daily 16:30 IST | GitHub Actions updates the `market-data` branch |
| Retraining | Weekly | Model learns from the newest data |

Live prices: Yahoo Finance (may lag a few minutes) until Angel One is active, then Angel One real-time feed.

**Intraday**

- **9:45 AM (market days):** fetch data + news → intraday picks → save → Telegram alert.
- **3:20 PM:** record actual results → update paper P&L and accuracy.
- **Weekly:** long-term picks + model retraining.
- NSE holidays skipped automatically; missed runs caught up when the laptop is opened.

## 8. Tech stack

Python · Angel One SmartAPI · yfinance · SQLite · pandas · TA-Lib · LightGBM · FinBERT · Streamlit · APScheduler · Telegram Bot API

## 9. Build phases

Order: **long-term first** (needs only free daily data, available now),
then **intraday** (needs Angel One), **Telegram last**.

### ✅ Done

| # | Phase | Delivered |
|---|---|---|
| 1 | Setup | Project structure, SQLite database, `.env` config, Angel One data-only client, CLI |
| 2 | Data pipeline | Daily prices since 2010 for Nifty 100 + indices, split/bonus handling, incremental updates, data-check, intraday downloader (ready for Angel One) |
| 2b | Git data store | CSV store on `market-data` branch, daily GitHub Actions updater, shallow `sync-data` on the Mac |
| 3 | Long-term features | 51 features: momentum, trend, 52-week range, volatility/drawdown, RSI/MACD/ADX/Bollinger, volume, beta, strength vs Nifty & sector peers, weekly candles, market regime, ranks; look-ahead tests |
| 4 | Fundamentals & news | Google News RSS + FinBERT sentiment in GitHub Actions (4×/day); weekly fundamentals snapshots |
| 5 | Long-term model + backtest | LightGBM ranker, walk-forward with embargo, next-day execution, delivery costs, benchmarks, momentum baseline |
| 6 | Paper trading + accuracy | ₹1 lakh account, queued orders filled at next price, predictions judged after 3 months vs random baseline |
| 7 | Streamlit app | Picks, Paper trading, My portfolio, Accuracy, Model, Settings; live auto-refresh; light/dark |
| 8 | My Portfolio | Manual Groww trades, average cost, realized/unrealized P&L, stop-loss alerts |
| 9 | Live monitor | 1-min prices & stop-losses, 15-min news & provisional scores, after-close decision, weekend retrain, holiday detection, catch-up |

**Track A is complete.** Long-term paper trading starts now and runs while Track B is built.

### Backtest findings (2015 – Sep 2026, walk-forward, costs included)

| Strategy | Yearly return | Sharpe | Worst fall |
|---|---|---|---|
| Model, weekly rebalance | 28.4% | 1.47 | −48% |
| Model, daily rebalance | 27.3% | 1.34 | −49% |
| Momentum only (no ML) | 28.0% | 1.11 | −45% |
| Equal-weight Nifty 100 | 19.0% | 1.09 | −38% |
| Nifty 50 | 9.0% | 0.56 | −38% |

- Top-10 picks beat Nifty over 3 months 57% of the time (all stocks: 52%); IC 0.044 (t≈2).
- **Weekly rank rebalancing is the default** (better risk-adjusted, ~⅓ lower costs); stop-loss and news exits stay daily.
- ⚠️ **Survivorship bias:** only today's Nifty 100 members are in the data, so absolute returns are overstated (equal-weight shows 19% vs Nifty's 9%). Compare against equal-weight, not Nifty. The model roughly matches plain momentum on return with smoother results. Paper trading is the real test.
- News and fundamentals are an overlay (skip/exit on strongly negative news; P/E and ROE shown) until enough history exists to train on them.

**Possible improvements:** broader universe (Nifty 200/500) with point-in-time selection to reduce survivorship bias; add news and fundamentals as model inputs after ~6-12 months of collection.

### Track B — Intraday (after Angel One is active)

**Phase 10 — Intraday data & features**
- 5-min / 1-min history from Angel One (`prices-intraday`)
- Intraday candle patterns, first-30-min behaviour (9:15–9:45 return, volume surge, range), VWAP position, gap from previous close, opening-range breakout
- Market context at 9:45: Nifty / sector moves, VIX, GIFT Nifty gap, overnight news sentiment

**Phase 11 — Intraday model + backtest**
- Target: return from 9:45 to 3:15; LightGBM ranker → top 5 up / top 5 down
- Intraday costs (brokerage, STT, charges, slippage), ATR stop-loss / target, "skip today" threshold
- Walk-forward backtest vs random baseline — *proves whether an intraday edge exists*

**Phase 12 — Intraday paper trading, app pages, scheduler jobs**
- ₹1,00,000 intraday paper account; 9:45 AM picks job and 3:20 PM results job
- Intraday Picks page, Intraday Paper Trading tab, intraday accuracy

### Final

**Phase 13 — Telegram alerts**
- Weekly long-term picks and daily 9:45 intraday picks, plus end-of-day P&L summary

Then: **paper trade for 2–3 months** before trusting real money to the picks.

## 10. Ground rules

- Personal use only (sharing picks publicly needs SEBI Research Analyst registration).
- No automatic order placement — I place trades myself on Groww.
- No paid services, no cloud compute.
- The model is a decision aid, not a guarantee; realistic intraday accuracy is ~51–55%.
