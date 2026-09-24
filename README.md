# Stock Predictor

Personal AI stock predictor for the Nifty 100, with two tracks, both complete:

- **Long-term:** weekly-rebalanced picks (top 10 expected to beat Nifty over 3 months)
- **Intraday:** at 9:45, 5 longs and 5 shorts from the first 30 minutes, squared off at 15:15

Both have live monitoring, paper trading, accuracy tracking, your real portfolio and a local
dashboard. See [PLAN.md](PLAN.md).

> Market data only — this project never places orders. You trade manually on Groww.
> Runs on your Mac at zero cost; market data and news are collected free by GitHub Actions.

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

`Ctrl+C` in the terminal stops everything. Keep the Mac awake during market hours
(the build task uses `caffeinate`).

**Other tasks** (`Cmd+Shift+P` → Tasks: Run Task): *Keep training now (retrain + self-tune)*,
*Get latest data*, *Backtest long-term*, *Backtest intraday*, *Web dashboard only*, *Run tests*.

Terminal equivalents: `python -m stockpredictor start`, `... improve --tune`, `... backtest`.

## Data and continuous learning

| What | How much | Where |
|---|---|---|
| Daily prices | **Nifty 200** (training) since **2005**, ~830k rows | git `market-data` branch, updated 16:30 IST by GitHub Actions |
| Trading universe | Nifty 100 (picks, paper trading, news) | |
| Intraday | Daily summaries of 5-min bars: 200 stocks, Yahoo (60 days, growing daily) + Angel One backfill (~2 years) | git + small local file |
| News | Google News + FinBERT, 4× per trading day | git |

The more days pass, the more data the models have: they retrain after every close and self-tune
weekly. The **Model** page shows each tuning run (prediction quality before/after) so you can see
whether accuracy actually improves.

## Dashboard pages

- **Long-term picks** — top 10 / weakest 10 with confidence, reasons, news mood, P/E, ROE; live provisional ranking
- **Intraday picks** — today's longs and shorts with entry, stop-loss, target, live move and result
- **Paper trading** — Rs 1 lakh each for long-term and intraday: holdings, orders, value, closed trades
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
  features/    longterm.py (51 features), intraday.py (9:45 features), labels.py
  models/      longterm.py, intraday.py (LightGBM ranking models, walk-forward)
  backtest/    portfolio.py + run.py (long-term), intraday.py (intraday rules + backtest)
  paper/       engine.py + daily.py (long-term), intraday.py (intraday paper trading)
  portfolio/   real.py (your Groww trades)
  live/        prices.py (Angel One / Yahoo), intraday_bars.py, monitor.py (live loop)
  app/         main.py (Streamlit dashboard), charts.py
.github/workflows/  update-market-data.yml, update-news.yml
```
