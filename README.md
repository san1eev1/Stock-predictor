# Stock Predictor

Personal AI stock predictor for the **Nifty LargeMidcap 250** (Nifty 100 + Midcap 150), with two
tracks:

- **Long-term:** 1-week predictions — the top 10 buy candidates after every close, judged
  after a week against Nifty 50; a ₹1 lakh buy-only paper portfolio holding the 5 best.
- **Intraday — two competing models:** at 9:45 each lists the top 10 buys and top 10 sells
  (separate tables) from the first 30 minutes and paper-trades only the 5 best buys + 5 best sells (fewer
  trades, lower costs; all 10 + 10 are still judged and learned from). One trades **until 12:30**, the other **until the close (15:15 square-off)**, each
  on its own ₹1 lakh a day. After the close they are compared, both learn from the full session,
  and each can learn from the other.

Both have live prices every minute, paper trading, accuracy tracking (buy and sell picks
separately), your real portfolio and a local dashboard. See [PLAN.md](PLAN.md) for the build
history and [DAILY_GUIDE.md](DAILY_GUIDE.md) for step-by-step daily use.

> Market data only — this project never places orders. You trade manually on Groww.
> Runs free: your Mac + GitHub Actions (public repo).

## Requirements (what we have agreed so far)

| Area | Requirement |
|---|---|
| **Universe** | Both models **train on and pick from the Nifty LargeMidcap 250** (Nifty 100 + Midcap 150), refreshed from NSE's official list daily. |
| **History** | Long-term model trains on **all daily history since 2005**. Intraday model trains on ~2 years of Angel One 5-minute history plus Yahoo's daily-growing summaries. |
| **Continuous training** | **All learning runs on GitHub**, on all history since 2005: **every day at 21:00 IST** (data update, then training) **and 23:00 IST** — retraining, self-tuning, historical paper-trading replays and trading rules chosen by profit, for long-term and both intraday models. **The Mac trains nothing** (keeps it cool and its memory free): live prices, picks, paper trading, downloading the models; it keeps only the last 4 years of prices on disk and 3 years in memory. (`MAC_DAILY_TRAINING=1` adds a daily 16:00 Mac retrain; `MAC_TRAINING=1` the old background tuning.) |
| **Learning from the live market** | The model **learns from the market until it closes**: after 15:30 the day's full live Angel One session is added and the intraday model retrains. |
| **Intraday: two trades** | Picks at 9:45. **Book 1 trades until 12:30**, **book 2 until the close (15:15 square-off)**, each with its own model (predicting 9:45 → 12:30 and 9:45 → 15:15), its own ₹1 lakh and stop-loss / target. Separate pages: *Intraday — until 12:30* and *Intraday — until close*. |
| **Compete and learn from each other** | After the close both books are judged; a daily winner and running score are shown (🏆 Competition). Each model's tuning may blend in the other's ranking (`peer_weight`), kept only if it improves accuracy on unseen days. A walk-forward backtest compares 12:30 vs close exits on the same days (refreshed daily). |
| **Train on buy and sell** | Both models rank every stock from strongest to weakest, so they learn both ends. Judged paper picks — **buy and sell** — feed back into training (wrong calls weigh 2×, right 1.5×); intraday tuning can also focus on the biggest risers and fallers. |
| **Paper trading** | Intraday: **two sections, each starting fresh with ₹1 lakh every day** (until 12:30 / until close); each day's result is saved and **compared day by day** (P&L, trades won, buy/sell accuracy vs random). Long-term: one running ₹1 lakh portfolio. Buying and selling shown in **separate tables**. |
| **Accuracy** | Shown on every picks and paper page, for that page's model, with **predicted UP and predicted DOWN in separate tables**; always compared with random picks. Long-term pages include a live "today" row. |
| **Live data** | Live prices **every minute** for all tradable stocks (Angel One, Yahoo fallback); the dashboard refreshes itself every minute. **Today %**, share move since entry and **P&L after costs** are shown side by side; live price on long-term picks. |
| **Chart methods** | Candlestick patterns and classic technical / statistical methods are available to the model; only groups that improve out-of-sample accuracy are used (see below). |
| **News** | **No buy/sell tips** (target prices, "stocks to buy/watch", broker calls, forecasts) and **no plain price-move reports** ("shares fall 2%", "share price today", 52-week highs, index moves) unless they name a company event — only news directly about the company (results, orders, deals, management, regulators…). A model **learns which headlines really move each stock** and weights news by it. |
| **Storage** | Big history stays **on git**, not on the Mac: the Mac keeps only the last 4 years of daily prices (~13 MB) and downloads trained models (~2 MB). |
| **Honesty** | Every change to the models is measured on data they never trained on and only kept if it helps; results always shown next to random picks. No method predicts markets with high certainty — expect a few points above 50%. |
| **Secrets** | Angel One keys only in `.env` (git-ignored), never in `.env.example`, never logged. |

## Run it from VS Code

The dashboard is a **local web page** at <http://localhost:8501> (any browser, or
`Cmd+Shift+P` → **Simple Browser: Show**).

**One-time setup**

1. Terminal: `brew install python@3.12 libomp git gh`
2. `git clone https://github.com/san1eev1/Stock-predictor.git`, then in VS Code
   **File → Open Folder… → Stock-predictor** (branch `claude/vibrant-heisenberg-yabdry`).
3. Install the recommended **Python** extension when VS Code asks.
4. `Cmd+Shift+P` → **Tasks: Run Task** → **1. First-time setup**.
5. Put your Angel One keys in `.env` (real-time prices + intraday history), then run the task
   **Angel One: check login**. The first `start` downloads ~2 years of intraday history in the
   background (~15 min).
6. Optional: `gh auth login` so you can start GitHub training runs from the terminal.

**Every day** — press **F5** (▶ *Start Stock Predictor*) or `Cmd+Shift+B`, or turn on
**Autostart** (starts 09:00 Mon–Fri). That one command:

- opens the dashboard and runs the live monitor (prices every minute, stop-losses, news)
- 9:46 intraday picks → paper trades on a fresh ₹1 lakh → **12:30 square-off**
- 15:30 close → learns from today's session → after-close long-term decision
- background training all day; downloads the newest GitHub-trained models every 15 minutes

`Ctrl+C` stops everything. Keep the Mac awake during market hours (the task uses `caffeinate`).

**Other tasks** (`Cmd+Shift+P` → Tasks: Run Task): *Training: ON/OFF* (daily background job for
days the app isn't open), *Keep training now*, *Get latest data*, *Backtest long-term*,
*Backtest intraday*, *Web dashboard only*, *Run tests*.

## How the models learn

### Where training happens

| Where | What | How often |
|---|---|---|
| **GitHub Actions** (free) | Long-term model on all history since 2005 + judged paper predictions, then self-tuning for the rest of a ~150-minute run; news relevance model (each tuning round is also checked on paper picks: new settings are kept only if the 5 stocks it would buy beat Nifty at least as often); **long-term historical paper trading** (walk-forward since 2015 with the live rules — the 5 best held — plus a 10-holding comparison, costs included) once a day on the first run after the 15:30 close. Published to the `models` branch. | **Twice each evening, every day**: 21:00 IST (data update, then training) and 23:00 IST; started by the Mac on time (`update-market-data.yml` → `train-models.yml`) |
| **GitHub Actions**, same runs | **Intraday (both models):** Angel One 5-minute history (keys in GitHub Secrets; the data stays in GitHub's private Actions cache, never committed) topped up with each new day → retrain on all of it + judged intraday paper picks from the Mac → **trading rules chosen by profit after costs** (0–5 buys and 0–5 sells per day, skip weak days, stop-loss/target; must earn more on the last 120 days and not less on the 120 before) → 3 rounds of historical paper-trading replays → 12:30-vs-close comparison → self-tuning with a paper check each round. Results appear on *Paper trading — Intraday* → 🔁 | Same runs |
| **Mac (only with `MAC_TRAINING=1`; off by default), background thread** | Both intraday models take turns self-tuning on their history (uses your Angel One data, which stays on the Mac). **After every round a paper-trading check** trades the last 120 days with the settings in use (5 best buys + 5 best sells, ₹1 lakh a day, costs); new settings are kept only if they also paper-trade at least as well (🧪 on *Paper trading — Intraday* → 🔁) | Every ~5 minutes in market hours, back to back after the close |
| **Mac (only with `MAC_TRAINING=1`; off by default), after the close** | Today's full live session is added; both intraday models retrain with it; the 12:30-vs-close comparison is refreshed | Every trading day, 15:32 |
| **Mac (only with `MAC_TRAINING=1`; off by default), after the close** | **Historical paper-trading replays:** each intraday model trades the last 120 days again in **3 rounds**, each round learning from the previous round's judged buy/sell picks; kept for the live model only if the last round beats the first (🔁 tab on *Paper trading — Intraday*) | Daily, 15:40 (weekends: any time) |
| **Mac, after the close** | Live prices stop at 15:30; the Mac saves today's Angel One session for live features and sends the judged paper picks to GitHub. No training (only with `MAC_DAILY_TRAINING=1`: one retrain at 16:00 in its own process) | Every trading day |
| **Mac, after each close** | Judged paper predictions (stock, date, right/wrong) sent to the `paper-feedback` branch for cloud training | Daily |

New settings are adopted only if they beat the current ones out-of-sample over the last 3 years
**and** are not worse over 6 years (intraday: last 120 and 250 days), so a setting that only fits
one period by luck is rejected. Each model is an **average of 3 LightGBM models** with early
stopping. Set `CLOUD_TRAINING=0` in `.env` to train everything on the Mac instead.

### What they learn from

- **Long-term (~100 inputs):** momentum, trend, 52-week range, volatility, RSI / MACD / ADX /
  Bollinger, volume, relative strength vs Nifty and sector, weekly candles, market regime.
- **Chart methods** (`features/technical.py`): trend systems (Supertrend, Ichimoku, Aroon,
  Donchian breakouts, Heikin-Ashi, Keltner squeeze) are **on** — in walk-forward tests they
  raised the top-10 weekly excess return (3 years: 0.67% → 0.85%; 6 years: 0.75% → 0.92%).
  Candlestick patterns, oscillators, volume flow and statistical signals **lowered or didn't
  change** accuracy, so they are off; hourly self-tuning keeps testing them and switches a group
  on only if it proves itself.
- **Intraday:** the first 30 minutes (gap, move, range, VWAP, volume) vs the whole market
  **and vs the stock's own sector**, plus yesterday's daily context.
- **Live race (long-term):** AI model vs 50/50 blend with momentum vs momentum only, each paper
  predicting; after 20+ judged days the best live variant takes over if it leads by 5+ points.

### Accuracy and profit work (Sep 2026)

Every change below was checked walk-forward (each period predicted by a model trained only
on earlier data) before it was kept.

| Change | Result |
|---|---|
| **Price-data cleaning** (`data/clean.py`): bad rows dropped, 22 unadjusted splits/bonuses/demergers (TMPV, VEDL, TRENT…) back-adjusted, real crashes kept; intraday prices put on the same basis | Long-term IC over 6 years 0.037 → 0.039 |
| **Sector-relative and event-calendar features** (intraday): move vs own sector at 9:45, F&O expiry days (Thu → Tue from Sep 2025), ex-dividend days | Intraday IC 12:30 0.025 → 0.033, close 0.032 → 0.038 |
| **Long-term trading score: 90% momentum + 10% model** (was 50/50) | Backtest 2015–2026, 5 holdings, costs: **13.9% → 22.6% a year** (Sharpe 0.58 → 0.83) vs Nifty 9.0%; momentum alone 22.2% (0.73) |
| **Intraday trading rules chosen by profit** (trades per side, skip weak days, stop/target; must trade on ≥1 day in 5) | History: 12:30 book −₹381 → −₹114/day, close book −₹394 → −₹37/day. **Not yet profitable.** |
| **Realistic paper trading**: slippage from each stock's first-30-min range, positions ≤1% of volume, daily loss limit 1.5% per book | Paper results closer to what real trading would give |
| Trade-outcome label, "take this trade?" meta-model, sector caps, volatility sizing | Tested; not better yet. They stay as options the tuner can pick if they start winning |

New inputs (most start mattering once their data has built up on GitHub):

- **NSE delivery %** (`data/delivery.py`, history back to 2005, downloaded in chunks).
- **Results dates** (`data/earnings.py`, ~20 years): days since/to results, results day, and
  **post-results drift** (how the stock reacted to its last results, kept for a quarter).
- **Overnight cues**: S&P 500, Nasdaq, US VIX, Nikkei, Hang Seng, USD/INR, crude. Long-term
  uses only closes before the decision day; intraday uses the last close before 9:15.
- **1-minute and 3-minute opening features** (`data/fine.py`): first 5/15-minute moves,
  first-15-minute range breakout, volume burst, up-minute share, last 3-minute move.
- **NSE pre-open auction** (`data/preopen.py`): auction price, buy vs sell order imbalance,
  auction volume. NSE keeps no history, so it is collected daily from now on.
- **Ridge model mixed into LightGBM**, recency weighting and ranking objective as tuning options.

### Safety net (Sep 2026)

- **Data-quality gate**: before each GitHub training run the data store is checked (stale or
  missing prices, broken rows, duplicates); on errors the run stops and the last good models
  stay in use. Issues are shown on the dashboard.
- **Promotion gates**: new model settings must beat the current ones significantly (paired
  t-statistic ≥ 2 on daily IC); new trading rules are chosen on older days, must also win on
  the untouched newest 60 days, then run 5+ trading days in a **shadow period** before going live.
- **Kill-switches** (no new trades; predictions still saved and judged): stale data or live
  prices, broken predictions, an intraday book down 3% in a week, the long-term book 25% below
  its peak.
- **Health banner** on every page: failed GitHub runs, data issues, old models, active
  kill-switches, **model decay** (live picks worse than random), paper vs simulation gaps.
- **Paper vs simulation**: after each close the day's paper trades are re-run through the
  backtest simulator; a big difference means live and backtest disagree.
- **Stress tests**: `scripts/research_stress.py` (strategy through real crises) and a live
  "Nifty falls 10%" estimate on *Paper trading — Long-term*.
- **Research on GitHub** (`research.yml`, scripts in `scripts/`): experiments never run on the Mac.

### News

Google News headlines for each stock, scored with FinBERT on GitHub 4× per trading day.
Tips are dropped (~16% of headlines). A LightGBM model (`nlp/relevance.py`) learns from history
which headlines came with real moves in their stock — on unseen headlines it matches real
reactions better than tone alone (correlation 0.15 vs 0.11; 0.10 vs 0.04 excluding price-move
reports). Relevant news counts more in the news mood and "sell on severe bad news"; lists,
namesakes and noise count less. News is a live signal, not a model input yet (history only
starts in 2026).

## Data and storage

| What | Where | On the Mac |
|---|---|---|
| Daily prices, 250 stocks since 2005 | git `market-data` branch, updated 16:30 IST by GitHub Actions | Last 4 years only (git partial clone) |
| Intraday summaries (one row per stock per day from 5-min bars: first 30 min, 12:30 exit, stop/target hit times) | Yahoo → git; Angel One → local file | ~20 MB |
| News + FinBERT scores | git | ~6 MB |
| Trained long-term + news models | git `models` branch | ~2 MB (`trained-models/`) |
| Intraday model, paper trades, your portfolio | local only (`models/`, `data/stockpredictor.db`) | small |

## Dashboard pages

- **Long-term picks** — long-term accuracy (UP table only — long-term predicts only UP; live today row); top 10 buy candidates
  (no sell candidates) with live price, Today %, Since pick %, reasons, news, P/E, ROE; *Sell now*
- **Intraday — until 12:30** / **Intraday — until close** — each model's accuracy (UP / DOWN);
  top 10 buy / top 10 sell (🧪 = the 5 + 5 traded) at 9:45 with entry, stop-loss, target, exit, live and 15:30 close prices;
  the close page also compares 12:30 vs close exits
- **Paper trading — Long-term** — accuracy; **Buying** (holdings + queued buys) and **Selling**
  (queued sells + sold) tables with P&L after costs
- **Paper trading — Intraday** — two books (until 12:30 / until close), each on a fresh ₹1 lakh
  a day with buy and sell trades and a **Day by day** comparison; **🏆 Competition** tab
- **My portfolio** — your Groww trades; live P&L, Today %, stop-loss alerts
- **Accuracy** — both models in detail, vs random; live strategy race
- **Model** — what the model relies on, cloud training status, backtest, every training run
- **Settings** — paper trading rules, stop-losses, paper account reset

## Commands

```bash
python -m stockpredictor start            # everything: dashboard + live monitor + training
python -m stockpredictor app              # dashboard only
python -m stockpredictor autostart [--off]         # start 09:00 Mon-Fri (macOS)
python -m stockpredictor schedule install|remove   # daily background job (18:00, 21:30)
python -m stockpredictor today            # one-shot daily run
python -m stockpredictor improve --tune   # retrain + self-tune now (on the Mac)
python -m stockpredictor cloud-train      # what GitHub runs every hour
python -m stockpredictor backtest | backtest-intraday
python -m stockpredictor sync-data | status | check-angel
gh workflow run train-models.yml          # start a GitHub training run now
```

## Tests

```bash
pytest          # 98 tests
```

## Layout

```
src/stockpredictor/
  config.py, db.py, store.py (git data, models and feedback branches), universe.py, costs.py
  data/        daily.py, intraday.py (daily summaries of 5-min bars), angelone.py, news.py,
               fundamentals.py, quality.py
  nlp/         sentiment.py (FinBERT, on GitHub), relevance.py (which news matters)
  features/    longterm.py, technical.py (chart methods), intraday.py (9:45 features), labels.py
  models/      longterm.py, intraday.py, engine.py (LightGBM), trainer.py (retrain + tuning)
  backtest/    run.py + portfolio.py (long-term), intraday.py (rules + backtest, 12:30 exit)
  paper/       engine.py + daily.py (long-term), intraday.py (daily ₹1 lakh), scoreboard.py
  portfolio/   real.py (your Groww trades)
  live/        prices.py, monitor.py (live loop), background.py (background training)
  app/         main.py (Streamlit dashboard), charts.py
.github/workflows/  update-market-data.yml, update-news.yml, train-models.yml
```
