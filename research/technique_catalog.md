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
| 2 | Options/futures positioning as features: open-interest change × price change (long/short build-up), put-call OI ratio | Pan & Poteshman (2006) RFS; "Why does option open interest predict stock returns?"; NSE F&O bhavcopy (archives, OI daily since 2000s) | data-limited model: genuinely new information | medium (downloader + features) | **E-011: downloader built, data downloading** |
| 3 | Residual (idiosyncratic) momentum | Blitz, Huij, Martens (2011) JEF; null result on US large caps 2016–26 (github blaquebaux/bespoke) | long-term risk-adjusted return | low | **Rejected** (E-010: t −0.41) |
| 4 | Overnight vs intraday momentum split | Lou, Polk, Skouras (2019) JFE | long-term features; entry timing | low | **Rejected** (E-010 t 1.64; CPCV E-010b t 1.03, 67% of paths) |
| 5 | Cost-aware execution filter / no-trade zone: trade only when the predicted move exceeds the round-trip cost | Jensen, Kelly, Malamud, Pedersen, "Machine Learning and the Implementable Efficient Frontier" (RFS) | intraday costs | low-medium | proposed (we have min_prob; tuner keeps it off) |
| 6 | Trees and shallow (≤3-layer) nets are the best ML return predictors; deeper nets worse | Gu, Kelly, Xiu (2020) RFS | confirms our LightGBM choice; DL already rejected (E-005) | — | no change |
| 7 | Learning-to-rank objective, purged CV | qlib / QRT 2026 challenge repos | already have rank objective + CPCV harness | — | done |
