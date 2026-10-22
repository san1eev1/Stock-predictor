# Technique catalog

Status: discovered → prototyped → validated → integrated (or rejected).

| Technique | Source | Defect / opportunity | Status |
|---|---|---|---|
| Circuit-limit guard (no buy at the upper band, no sell at the lower band) | qlib `Exchange(limit_threshold)` | Backtest could trade locked stocks | **Integrated** (`backtest/portfolio._at_circuit`, test `test_no_buy_at_upper_circuit`) |
| Alpha158 extras: CORR, CORD, RSV, QTLU/QTLD, IMXD, CNTD (20/60 days) | qlib `Alpha158DL` | Possibly missing price-volume / timing information | **Prototyped**: E-001 |
| Cross-sectional z-score label | qlib handler `CSZScoreNorm` | Market-wide moves add label noise | Discovered; we already train on excess-vs-market targets and have a rank-objective option |
| DoubleEnsemble | qlib `double_ensemble.py` | Robustness to noisy samples | Discovered; ensembles failed before (E-history), low priority |
| CPCV (combinatorial purged CV) | skfolio | Single-path walk-forward gives noisy promotion gates | Candidate |
| BCa bootstrap CIs for Sharpe / profit factor | pybroker (idea only) | Reports give point estimates only | Candidate (write our own) |
| Square-law market impact | zipline VolumeShareSlippage | Impact grows with size | Candidate (low impact at ₹1 lakh) |
| Stop fills beyond the stop price | general practice | Stops filled exactly at the stop | **Integrated** (`STOP_SLIPPAGE` 0.1%) |
| Frozen monthly holdout + trial-count promotion bar | López de Prado; deflated Sharpe | Rolling holdout reused nightly | **Integrated** (`tune_rules`, `trial_t_min`) |
| Point-in-time universe from bhavcopy | general | Survivorship bias | **Validated as a test**, see E-000 |
