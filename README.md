# Stock Predictor

Personal AI stock predictor for the Nifty LargeMidcap 250 (Nifty 100 + Midcap 150), with two tracks, both complete:

- **Long-term:** 1-week predictions (10 buy / 10 sell candidates, judged after a week); buy-only paper portfolio
- **Intraday:** at 9:45, 5 longs and 5 shorts from the first 30 minutes, squared off at 15:15

Both have live monitoring, paper trading, accuracy tracking, your real portfolio and a local
dashboard. See [PLAN.md](PLAN.md).

> Market data only — this project never places orders. You trade manually on Groww.
> Runs on your Mac at zero cost; market data and news are collected free by GitHub Actions.

> 📘 **Step-by-step daily instructions: [DAILY_GUIDE.md](DAILY_GUIDE.md)**
> (autostart at 9:00, press F5, or a one-shot quick daily run).

## Run it from VS Code (no app to install)

Everything runs from VS Code. The dashboard is a **local web page** at
<http://localhost:8501>: open it in any browser, or inside VS Code with
`Cmd+Shift+P` → **Simple Browser: Show** → `http://localhost:8501`.

**One-time setup**

1. Install tools (Terminal): `brew install python@3.12 libomp git`
2. Get the project: `git clone https://github.com/san1eev1/Stock-predictor.git`, then in VS Code
   **File → Open Folder… → Stock-predictor** (on branch `claude/vibrant-heisenberg-yabdry`).
3. Install the recommended **Python** extension when VS Code asks.
4. `Cmd+Shift+P` → **Tasks: Run Task** → **1. First-time setup**
   (creates `.venv`, installs everything, downloads the data).
5. Optional: put your Angel One keys in `.env` (real-time prices + intraday history), then run the
   tasks **Angel One: check login** and **Angel One: download intraday history (one-time)**.

**Every day**

Press **F5** (▶ *Start Stock Predictor*), or `Cmd+Shift+B`. That one command:

- opens the web dashboard
- runs the live monitor (prices, stop-losses, news, 9:46 intraday picks, 15:15 square-off)
- makes the after-close decision and **retrains both models on the newest data**
- **self-tunes every weekend** (tries new model settings, keeps them only if they test better)

**Training every day, weekends included**

Run the task **Training: ON** once (or `python -m stockpredictor schedule install`). A macOS
background job then runs every day at 18:00 and 21:30: on weekdays it syncs the data, makes the
after-close decision and retrains both models; at weekends it self-tunes. If the Mac was asleep,
it runs when the Mac wakes. It does nothing while the live monitor (`start` / autostart) is
running, since the monitor already does this, so it works alongside autostart: autostart runs
the trading day, this makes sure training never skips a day. Log: `logs/auto.log`.
Turn it off with **Training: OFF** (`schedule remove`).

`Ctrl+C` in the terminal stops everything. Keep the Mac awake during market hours
(the build task uses `caffeinate`).

**Other tasks** (`Cmd+Shift+P` → Tasks: Run Task): *Keep training now (retrain + self-tune)*,
*Get latest data*, *Backtest long-term*, *Backtest intraday*, *Web dashboard only*, *Run tests*.

Terminal equivalents: `python -m stockpredictor start`, `... improve --tune`, `... backtest`.

## Data and continuous learning

| What | How much | Where |
|---|---|---|
| Daily prices | **Nifty 250** since **2005** (~1M rows) | git `market-data` branch, updated 16:30 IST by GitHub Actions; the Mac keeps the last 4 years |
| Universe | Nifty LargeMidcap 250 — both models train on it and pick from it | |
| Intraday | Daily summaries of 5-min bars: 200 stocks, Yahoo (60 days, growing daily) + Angel One backfill (~2 years) | git + small local file |
| News | Google News + FinBERT, 4× per trading day | git |

**Chart methods the model combines** (`features/technical.py`): daily candlestick patterns
(hammer, shooting star, engulfing, piercing / dark cloud, morning / evening star, three white
soldiers / black crows, harami, marubozu, doji, gaps, net pattern score), oscillators
(stochastic, Williams %R, CCI, money flow index), trend systems (Supertrend, Ichimoku, Aroon,
Donchian breakouts, Heikin-Ashi, Keltner squeeze), volume flow (OBV, Chaikin money flow) and
statistics (trend slope and R², autocorrelation, variance ratio, efficiency ratio, skew,
z-score). LightGBM learns how much each is worth and how they combine; nothing is a fixed rule.

### Where training happens

| Where | What | How often |
|---|---|---|
| **GitHub Actions** (free) | Long-term model: retrains on **all history since 2005** plus judged paper predictions (wrong ones weigh 2×, right ones 1.5×), then self-tunes for the rest of a ~45-minute run. News relevance model. Weekly backtest. Results go to the `models` branch (~1.5 MB). | **Every hour** (`.github/workflows/train-models.yml`) |
| **Mac, background thread** | Intraday model: self-tuning rounds on its history (uses your Angel One data, which stays on the Mac) — keeps running during market hours without pausing live prices | Every ~5 minutes, all day |
| **Mac, live market** | Right after the 15:15 square-off, today's live Angel One 5-minute session is added and the intraday model retrains on it | Every trading day, 15:17 |
| **Mac** | Downloads the newest cloud models; sends judged paper predictions to the `paper-feedback` branch so cloud training learns from them | Every 15 min / after each close |

The Mac keeps only the last **4 years** of daily prices (~13 MB; the signals need ~1 year of
warm-up) — the full history lives on git and is only downloaded by GitHub Actions. Set
`CLOUD_TRAINING=0` in `.env` to train everything on the Mac instead (downloads all history).

New settings are adopted only if they beat the current ones out-of-sample over the last 3 years
**and** are not worse over 6 years (intraday: 120 and 250 days), so a setting that only fits one
period by luck is rejected.

**Live race:** three variants — AI model, 50/50 blend with momentum, momentum only — each make
paper predictions. After 20+ judged days, the system switches to the variant with the best
**live** accuracy if it leads by 5+ points.

**News:** buy/sell tips, target prices, "stocks to watch" lists and forecasts are ignored — the
model makes its own calls. A LightGBM model learns from history which headlines actually move
their stock (`nlp/relevance.py`); relevant news counts more in the news mood and bad-news exits,
lists and namesakes count less. On unseen headlines its relevance ranks real reactions better
than FinBERT tone alone (correlation 0.15 vs 0.11).

The training engine uses **early stopping** and an **average of 3 models** with different random
seeds. Features are cached in `data/cache` (turn off with `FEATURE_CACHE=0` in `.env`).

The **Accuracy** page shows the live race; the **Model** page shows every retrain and tuning run.

**Angel One:** with keys in `.env`, `start` checks the login, uses real-time prices, downloads ~2
years of intraday history in the background the first time (~15 min), and after each close saves
the day's Angel One intraday data.

## Dashboard pages

- **Long-term picks** — 10 buy and 10 sell candidates for the next week with **live price, Today % and Since pick %** (updated every minute), *Sell now* for your holdings, reasons, news, P/E, ROE; live intraday + long-term accuracy at the top
- **Intraday picks** — 10 buy and 10 sell candidates at 9:45 with entry, stop-loss, target, live Today % and share move since 9:45, result (🧪 = paper-traded); live accuracy at the top
- **Paper trading — Long-term** — buy-only ₹1 lakh portfolio (steady version of the weekly signal)
- **Paper trading — Intraday** — 10 buy trades and 10 sell (short) trades a day, separate sections
- **My portfolio** — add your Groww trades; live P&L, stop-loss alerts, allocation (Long-term / Intraday tabs)
- **Accuracy** — long-term picks judged after 3 months vs Nifty; intraday picks at 15:15; both vs random picks
- **Model** — what each model relies on, backtest results, retrain button
- **Settings** — long-term and intraday rules, stop-losses, paper account reset

## Commands

```bash
python -m stockpredictor start         # everything: web dashboard + live monitor + training
python -m stockpredictor app           # web dashboard only
python -m stockpredictor run           # live monitor only
python -m stockpredictor improve --tune  # sync, retrain, self-tune now
python -m stockpredictor daily         # run the after-close decision by hand
python -m stockpredictor schedule install|remove|status  # automatic daily training
python -m stockpredictor auto          # one pass of the daily background job
python -m stockpredictor sync-data     # fetch latest data snapshot
python -m stockpredictor train         # retrain the model now
python -m stockpredictor backtest      # walk-forward backtest (~5 min), shown on the Model page
python -m stockpredictor intraday-backfill / train-intraday / backtest-intraday
python -m stockpredictor features --symbol RELIANCE
python -m stockpredictor status
# Angel One (after filling .env):
python -m stockpredictor universe && python -m stockpredictor tokens
python -m stockpredictor check-angel
```

## Tests

```bash
pytest
```

## Layout

```
src/stockpredictor/
  config.py, db.py, store.py, universe.py, costs.py
  data/        daily.py, intraday.py (daily summaries of 5-min bars), angelone.py, news.py,
               fundamentals.py, quality.py
  nlp/         sentiment.py (FinBERT, used in GitHub Actions)
  features/    longterm.py (~100 features), technical.py (candlestick patterns, oscillators,
               trend systems, statistical signals), intraday.py (9:45 features), labels.py
  models/      longterm.py, intraday.py (LightGBM ranking models, walk-forward)
  backtest/    portfolio.py + run.py (long-term), intraday.py (intraday rules + backtest)
  paper/       engine.py + daily.py (long-term), intraday.py (intraday paper trading)
  portfolio/   real.py (your Groww trades)
  live/        prices.py (Angel One / Yahoo), intraday_bars.py, monitor.py (live loop)
  app/         main.py (Streamlit dashboard), charts.py
.github/workflows/  update-market-data.yml, update-news.yml
```
