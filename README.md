# Stock Predictor

Personal AI stock predictor for the Nifty 100. **Track A (long-term) is complete:** weekly-rebalanced
long-term picks with live monitoring, paper trading, your real portfolio, accuracy tracking and a
local dashboard. Intraday (Track B) comes next. See [PLAN.md](PLAN.md).

> Market data only — this project never places orders. You trade manually on Groww.
> Runs on your Mac at zero cost; market data and news are collected free by GitHub Actions.

## One-time setup (macOS)

```bash
brew install python@3.12 libomp git          # libomp is needed by LightGBM
git clone https://github.com/san1eev1/Stock-predictor.git
cd Stock-predictor
git checkout claude/vibrant-heisenberg-yabdry
git config remote.origin.fetch "+refs/heads/claude/vibrant-heisenberg-yabdry:refs/remotes/origin/claude/vibrant-heisenberg-yabdry"

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env        # optional: Angel One keys for real-time prices
```

Every new terminal: `source .venv/bin/activate` (or select the `.venv` interpreter in VS Code).

## Daily use

Two terminals:

```bash
# Terminal 1 - live monitor (leave running; caffeinate keeps the Mac awake)
caffeinate -i python -m stockpredictor run

# Terminal 2 - dashboard (opens in your browser)
python -m stockpredictor app
```

The monitor does everything on its own:

| When | What |
|---|---|
| Every minute, 9:15-15:30 IST | Live prices; fills queued paper orders; stop-loss exits (paper) and stop-loss alerts (your portfolio) |
| Every 15 min, market hours | Pulls new scored news from git; negative-news alerts / paper exits; provisional live re-ranking |
| From 17:15 IST | Syncs the day's data, makes the official decision, evaluates past predictions |
| Weekends | Retrains the model if it is older than 6 days |

If the Mac was off, it catches up on missed days when started. NSE holidays are detected automatically.

## Dashboard pages

- **Long-term picks** — top 10 / weakest 10 with confidence, reasons, news mood, P/E, ROE; live provisional ranking
- **Paper trading** — Rs 1 lakh virtual portfolio: holdings, queued orders, value vs Nifty, closed trades
- **My portfolio** — add your Groww trades; live P&L, stop-loss alerts, allocation (Long-term / Intraday tabs)
- **Accuracy** — each prediction judged after 3 months vs Nifty, compared with random picks
- **Model** — what the model relies on, backtest results, retrain button
- **Settings** — rules (stocks held, exit rank, stop-loss, weekly/daily rebalance, news exits), stop-losses, reset

## How data works

GitHub Actions (free) keep the `market-data` branch up to date:

| Job | When (IST) | What |
|---|---|---|
| Update market data | 16:30 Mon-Fri | Daily prices for Nifty 100 + indices; fundamentals on Fridays |
| Update news | 10:00, 12:00, 14:00, 17:00 Mon-Fri | Google News headlines scored by FinBERT |

The Mac fetches only the latest snapshot (`sync-data`, ~35 MB, no history). Features are built in
memory; the local SQLite database holds only app state.

## Commands

```bash
python -m stockpredictor run           # live monitor
python -m stockpredictor app           # dashboard
python -m stockpredictor daily         # run the after-close decision by hand
python -m stockpredictor sync-data     # fetch latest data snapshot
python -m stockpredictor train         # retrain the model now
python -m stockpredictor backtest      # walk-forward backtest (~5 min), shown on the Model page
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
  data/        daily.py, intraday.py, angelone.py, news.py, fundamentals.py, quality.py
  nlp/         sentiment.py (FinBERT, used in GitHub Actions)
  features/    longterm.py (51 features), labels.py (plain-English names)
  models/      longterm.py (LightGBM ranking model, walk-forward)
  backtest/    portfolio.py (rules + simulator), run.py (report)
  paper/       engine.py (paper trading, evaluation), daily.py (after-close job)
  portfolio/   real.py (your Groww trades)
  live/        prices.py (Angel One / Yahoo), monitor.py (live loop)
  app/         main.py (Streamlit dashboard), charts.py
.github/workflows/  update-market-data.yml, update-news.yml
```
