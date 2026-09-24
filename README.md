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

## Commands

```bash
python -m stockpredictor init         # create the local database
python -m stockpredictor universe     # download Nifty 100 list from NSE
python -m stockpredictor prices       # daily prices since 2010 for Nifty 100 + indices (~5-10 min first time)
python -m stockpredictor data-check   # coverage / gaps report
python -m stockpredictor status       # show configuration and row counts
# after Angel One keys are in .env:
python -m stockpredictor tokens       # map stocks to Angel One tokens
python -m stockpredictor check-angel  # test login + one live price
python -m stockpredictor prices-intraday --interval FIVE_MINUTE --days 365
```

Run `prices` again any time (e.g. daily after 4 PM) — it only downloads what is new.
Try a single stock first with `--symbols RELIANCE`.

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
  __main__.py        command line
tests/
data/                local database (git-ignored)
```
