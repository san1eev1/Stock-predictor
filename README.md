# Stock Predictor

Personal AI stock predictor for the Nifty 100: daily intraday picks (5 up / 5 down at 9:45 AM)
and weekly long-term picks, with paper trading, accuracy tracking and a local dashboard.
Runs entirely on your own Mac at zero cost. See [PLAN.md](PLAN.md) for the full plan.

> Market data only — this project never places orders. Trades are made manually on Groww.

## Setup (macOS)

```bash
# 1. Tools
brew install python@3.12 ta-lib git

# 2. Get the code
git clone https://github.com/san1eev1/Stock-predictor.git
cd Stock-predictor
git checkout claude/vibrant-heisenberg-yabdry

# 3. Virtual environment + dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 4. Secrets (stay on your Mac, never committed)
cp .env.example .env
open -e .env        # fill in values when your Angel One account is active
```

## How data works

Market data is **not downloaded on the Mac**. A free GitHub Actions job
(`.github/workflows/update-market-data.yml`) runs every weekday at 16:30 IST,
downloads prices for the Nifty 100 and indices, and commits compact CSV files to the
`market-data` branch. The Mac fetches only the latest snapshot (no history) for training:

```bash
python -m stockpredictor sync-data    # shallow fetch of market-data branch (~tens of MB)
python -m stockpredictor features     # build long-term features in memory (~10 s)
python -m stockpredictor features --symbol RELIANCE   # latest values for one stock
```

The local SQLite database only holds app state (predictions, paper trades, your portfolio).

First-time data load: GitHub → **Actions** → *Update market data* → **Run workflow**.

## Other commands

```bash
python -m stockpredictor init         # create the local app database
python -m stockpredictor status       # show configuration
# after Angel One keys are in .env:
python -m stockpredictor universe     # Nifty 100 list into the local database
python -m stockpredictor tokens       # map stocks to Angel One tokens
python -m stockpredictor check-angel  # test login + one live price
```

Used by the GitHub Action (can also be run locally): `universe`, `prices`, `data-check`,
`import-store`, `export-store`.

## Tests

```bash
pytest
```

## Layout

```
src/stockpredictor/
  config.py          settings from .env
  db.py              SQLite schema
  universe.py        Nifty 100 constituents
  data/angelone.py   Angel One SmartAPI client (data only)
  data/daily.py      daily prices, indices, splits (Yahoo Finance)
  data/intraday.py   intraday candles (Angel One)
  data/quality.py    data coverage report
  store.py           git CSV market-data store + sync
  features/longterm.py  long-term features (momentum, trend, risk, RS, weekly candles, regime)
  __main__.py        command line
tests/
data/                local app database (git-ignored)
market-data/         synced data snapshot (git-ignored, from the market-data branch)
```
