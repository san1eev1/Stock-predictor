# Technique catalog

Status: discovered → prototyped → validated → integrated (or rejected).

| Technique | Source | Defect / opportunity | Status |
|---|---|---|---|
| Circuit-limit guard (no buy at the upper band, no sell at the lower band) | qlib `Exchange(limit_threshold)` | Backtest could trade locked stocks | **Integrated** (`backtest/portfolio._at_circuit`, test `test_no_buy_at_upper_circuit`) |
| Alpha158 extras: CORR, CORD, RSV, QTLU/QTLD, IMXD, CNTD (20/60 days) | qlib `Alpha158DL` | Possibly missing price-volume / timing information | **Rejected** (E-001: t = 0.34) |
| Cross-sectional z-score label | qlib handler `CSZScoreNorm` | Market-wide moves add label noise | Discovered; we already train on excess-vs-market targets and have a rank-objective option |
| DoubleEnsemble | qlib `double_ensemble.py` | Robustness to noisy samples | **Rejected** (E-002: t = 0.79) |
| CPCV (combinatorial purged CV) | skfolio | Single-path walk-forward gives noisy promotion gates | **Validated** as the research harness (E-002); not yet used in the nightly gate |
| Block-bootstrap CIs for CAGR / Sharpe; Sortino, Calmar | pybroker (idea), quantstats | Reports gave point estimates only | **Integrated** (`portfolio.bootstrap_ci`, `performance`; backtest report) |
| Square-law market impact | zipline VolumeShareSlippage | Impact grows with size | Candidate (low impact at ₹1 lakh) |
| Stop fills beyond the stop price | general practice | Stops filled exactly at the stop | **Integrated** (`STOP_SLIPPAGE` 0.1%) |
| Frozen monthly holdout + trial-count promotion bar | López de Prado; deflated Sharpe | Rolling holdout reused nightly | **Integrated** (`tune_rules`, `trial_t_min`) |
| Point-in-time universe from bhavcopy | general | Survivorship bias | **Validated as a test**, see E-000 |
| Train on point-in-time history | own (E-000 data) | Model is data-limited and trained on survivors | **Rejected** (E-003: t = 0.66, lower CAGR) |
| Trailing stop | LEAN `TrailingStopRiskManagementModel` | −51% worst fall | **Rejected** (E-004: −3.8 pts/yr); option kept in `simulate(trail_stop=)`, off |

## Research round 2026-11-14 (papers, GitHub, books) — ranked by expected value for us
Measured weaknesses: intraday rank skill is real (IC 0.04–0.06, 59% direction) but below costs
(₹10k books: 0.84% per round trip); the long-term model is data-limited (E-001…E-005).

| # | Technique | Source | Fixes | Effort | Status |
|---|---|---|---|---|---|
| 1 | Market intraday momentum (prev close→9:45 predicts 15:00→close), one cheap instrument (Nifty futures / NIFTYBEES) | Gao, Han, Li, Zhou (2018) JFE; "Intraday time series momentum: global evidence" (Reading) | intraday costs (1 trade/day, index costs ≪ stock costs) | low | **E-009 running** |
| 2 | Options/futures positioning as features: open-interest change × price change (long/short build-up), put-call OI ratio | Pan & Poteshman (2006) RFS; "Why does option open interest predict stock returns?"; NSE F&O bhavcopy (archives, OI daily since 2000s) | data-limited model: genuinely new information | medium (downloader + features) | **Integrated for intraday** (E-011b t 9.0); long-term rank gain (CPCV 15/15) not yet in the portfolio |
| 3 | Residual (idiosyncratic) momentum | Blitz, Huij, Martens (2011) JEF; null result on US large caps 2016–26 (github blaquebaux/bespoke) | long-term risk-adjusted return | low | **Rejected** (E-010: t −0.41) |
| 4 | Overnight vs intraday momentum split | Lou, Polk, Skouras (2019) JFE | long-term features; entry timing | low | **Rejected** (E-010 t 1.64; CPCV E-010b t 1.03, 67% of paths) |
| 5 | Cost-aware execution filter / no-trade zone: trade only when the predicted move exceeds the round-trip cost | Jensen, Kelly, Malamud, Pedersen, "Machine Learning and the Implementable Efficient Frontier" (RFS) | intraday costs | low-medium | proposed (we have min_prob; tuner keeps it off) |
| 6 | Trees and shallow (≤3-layer) nets are the best ML return predictors; deeper nets worse | Gu, Kelly, Xiu (2020) RFS | confirms our LightGBM choice; DL already rejected (E-005) | — | no change |
| 7 | Learning-to-rank objective, purged CV | qlib / QRT 2026 challenge repos | already have rank objective + CPCV harness | — | done |

## Intraday profitability after costs (reading, 2026-11-27)
- Huddleston, Liu & Stentoft (J. Financial Econometrics 2023), *Intraday Market Predictability: A
  Machine Learning Approach*: ensembles predict intraday market returns; Sharpe ~0.98 after costs.
  Market-level, not single stocks -> supports the Nifty futures book (lab_mim), not stock picking.
- Aleti, Bollerslev & Siggaard (Management Science 2025), *Intraday Market Return Predictability
  Culled from the Factor Zoo*: lagged high-frequency cross-sectional returns predict the market;
  profitable after costs on liquid ETFs -> again: trade the index, cheaply.
- Cost-aware construction: Hautsch & Voigt (arXiv 1709.06296) and Fan, Medeiros, Yang & Yang (arXiv
  2412.11575): turnover penalties set by costs beat naive rebalancing; with high costs shift
  weight to slower-decaying signals (alpha-decay literature, Qian-Sorensen-Hua 2007).
- Trade size vs speed (CESifo wp8143): post-cost returns fall with size; cost savings must be
  weighed against alpha decay.
- India announcements: NSE-NYU working paper on unscheduled announcements (Nawn & Banerjee 2024):
  HFT/institutions trade first, retail loses intraday -> our filings feed must act within minutes
  or not at all; NSE tick-data paper: prices drift for days after news (slow at daily horizon).
Implication for this system: (1) costs and execution first, (2) index-level intraday signals are
the best-documented after-cost edge, (3) single-stock intraday needs signals that persist past
costs (news drift at daily horizon) rather than faster trading. Tested in E-014.

## Algo and maths menu by paper book (2026-11-28)
Status: LIVE = running as a paper book; TESTED = research run done; NEXT = queued; IDEA.
Every idea is tested on the walk-forward history (research.yml) before it gets a paper book;
adoption rule: net of Zerodha costs (~0.21% intraday round trip) t >= 2 and better than now.

### Intraday stocks (cash books)
- LIVE  ML ranking (LightGBM, first-30-min + daily + F&O features), 12:30 and close books.
- LIVE  Candle re-rank (E-015: candles reverse; now weight 0.25, hold to close).
- LIVE  News drift; gap fade without news; stocks in play (ORB); weekly reversal (lab).
- NEXT  30-minute reversal and candle reversal (E-016, Heston-Korajczyk-Sadka 2010).
- IDEA  VWAP mean reversion: stocks stretched > k sigma from VWAP revert intraday.
- IDEA  Intraday pairs / stat-arb within a sector: spread z-score, hedge ratio by Kalman
        filter, cointegration (Engle-Granger) test (Gatev-Goetzmann-Rouwenhorst 2006).
- IDEA  Lead-lag: big stocks' / sector leaders' first-hour moves predict laggards (Hou 2007).
- IDEA  Pre-open auction imbalance (data already collected) -> open-to-close direction.
- IDEA  Post-results drift, day 2 onwards (Bernard-Thomas 1989), using the results calendar.
### Index / futures
- LIVE  Nifty last-half-hour momentum (Gao-Han-Li-Zhou 2018).
- IDEA  Overnight drift: long Nifty futures close -> open (Lou-Polk-Skouras 2019).
- IDEA  Turn-of-the-month and pre-holiday effects (Ariel 1987; Lakonishok-Smidt 1988).
- IDEA  FII/DII flow signal for the next day's Nifty direction.
- IDEA  Opening-gap fade on Nifty when the gap has no global cue behind it.
### Options
- LIVE  Five hedged VRP sellers (iron fly 9:20 / 11:00, condor 50% target, condor VIX>RV,
        expiry fly) (Coval-Shumway 2001; Bakshi-Kapadia 2003).
- IDEA  IV-rank filter: sell only when IV is in the top third of its year.
- IDEA  Results IV crush: hedged short straddles on stocks reporting results (sell the day
        before, buy back after) - only with defined risk (iron flies).
- IDEA  Delta-hedged straddles (gamma scalping) to measure the VRP cleanly.
- IDEA  Calendar spreads around weekly vs monthly IV term structure.
### Long-term
- LIVE  ML ranking + negative-news filter; live race model / 50-50 / momentum.
- TESTED Residual momentum (E-010, rejected), F&O features (E-011a, rejected by rule).
- IDEA  52-week-high proximity (George-Hwang 2004); quality (profitability, low debt);
        low volatility (Nifty 100 Low Vol 30 logic); value; factor blend with equal risk.
### Maths for sizing, allocation and risk (apply to any book)
- IDEA  Volatility targeting / inverse-vol sizing (Moreira-Muir 2017) - partly in E-016.
- IDEA  Fractional Kelly sizing (edge / variance x 1/4) once a book has 60+ live days.
- IDEA  Capital allocation across strategy books as a multi-armed bandit (Thompson sampling
        on daily returns), with a floor so no book starves of evidence.
- IDEA  Risk parity / hierarchical risk parity across books (Lopez de Prado 2016); Ledoit-
        Wolf shrinkage for the covariance.
- IDEA  Regime model: hidden Markov model on Nifty returns + VIX; GARCH(1,1) volatility
        forecast to scale exposure and option strikes.
- IDEA  Meta-labeling (Lopez de Prado 2018): a second model deciding take / skip per trade
        (code exists, the nightly tuning keeps it off); triple-barrier labels.
- IDEA  Overfitting control: deflated Sharpe ratio, probability of backtest overfitting
        (Bailey-Lopez de Prado); CPCV already in use.
### Execution
- IDEA  Limit-order entries at 9:46 (E-014: worth ~+16 bp/day of costs); TWAP over 5 minutes.
