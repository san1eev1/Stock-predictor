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
| Runs on | My laptop (catches up on missed data if opened late) |
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

## 7. Automation

- **9:45 AM (market days):** fetch data + news → intraday picks → save → Telegram alert.
- **3:20 PM:** record actual results → update paper P&L and accuracy.
- **Weekly:** long-term picks + model retraining.
- NSE holidays skipped automatically; missed runs caught up when the laptop is opened.

## 8. Tech stack

Python · Angel One SmartAPI · yfinance · SQLite · pandas · TA-Lib · LightGBM · FinBERT · Streamlit · APScheduler · Telegram Bot API

## 9. Build phases

1. **Setup** — project structure, database, config, Angel One connection. ✅
2. **Data pipeline** — Nifty 100 daily + intraday history, split/bonus handling.
3. **Features** — candle patterns, indicators, first-30-min signals, market context.
4. **Intraday model + backtest** — walk-forward, with costs. *Proves whether there is a real edge.*
5. **News pipeline** — fetch + FinBERT sentiment, added as features.
6. **Long-term model + backtest.**
7. **Paper trading engine + accuracy tracking.**
8. **Streamlit UI** — all pages.
9. **Scheduler** — daily / weekly jobs, holiday calendar, catch-up.
10. **Portfolio page** — manual entry of real trades.
11. **Telegram alerts** — last step.

Then: **paper trade for 2–3 months** before trusting real money to the picks.

## 10. Ground rules

- Personal use only (sharing picks publicly needs SEBI Research Analyst registration).
- No automatic order placement — I place trades myself on Groww.
- No paid services, no cloud compute.
- The model is a decision aid, not a guarantee; realistic intraday accuracy is ~51–55%.
