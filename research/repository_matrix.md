# Repository matrix

Studied 2026-10-22: shallow clones, source read (not just READMEs). Nothing was installed into the
project environment. "Use" = what we took or could take.

| Repo @ commit | Licence | Modules read | Useful technique | Compatible? | Status |
|---|---|---|---|---|---|
| microsoft/qlib @5435523 | MIT | `contrib/data/loader.py` (Alpha158DL), `contrib/data/handler.py`, `backtest/exchange.py`, `contrib/model/double_ensemble.py` | Alpha158 feature families; cross-sectional label normalisation (CSZScoreNorm); `limit_threshold` (no trade at a price limit); DoubleEnsemble (sample reweighting + feature selection over LightGBM) | Yes: formulas reimplemented in pandas, no dependency | Circuit guard **integrated**; Alpha158 extras **prototyped** (E-001); DoubleEnsemble **discovered** |
| ranaroussi/quantstats @9ef4c6d | Apache-2.0 | `quantstats/stats.py` | Probabilistic/smart Sharpe, Sortino, Calmar, ulcer index, omega, tail ratio | Yes | We already have deflated Sharpe; ulcer/Sortino/Calmar **candidate** for reports |
| edtechre/pybroker @01f11fd | Apache-2.0 + **Commons Clause** | `src/pybroker/eval.py` | BCa bootstrap confidence intervals for Sharpe / profit factor | Ideas only (Commons Clause: no code copying) | **Discovered** |
| polakowo/vectorbt @dd8eb3a | Apache-2.0 + **Commons Clause** | (structure only) | Vectorised portfolio simulation | Ideas only | Our simulators are already vectorised where it matters |
| stefan-jansen/zipline-reloaded @cf649ac | Apache-2.0 | `finance/slippage.py` | VolumeShareSlippage: impact = k·(volume share)² with a volume cap | Yes | We cap at 1% of 30-minute volume plus range-based slippage; square-law impact **candidate** |
| stefan-jansen/alphalens-reloaded @1bc181d | Apache-2.0 | (structure) | IC by horizon, turnover, quantile spreads | Yes | Our research_diagnose covers IC/deciles; IC decay **candidate** |
| skfolio/skfolio @55e9e04 | BSD-3 | `model_selection/_combinatorial.py`, `_walk_forward.py` | Combinatorial purged CV (CPCV), walk-forward with purge/embargo | Yes | We use yearly walk-forward with an embargo; CPCV **candidate** to reduce variance in the gates |
| stefan-jansen/machine-learning-for-trading @35d6e2b | MIT | (structure) | Textbook: alphalens + LightGBM + walk-forward | Already matches our approach | — |
| AI4Finance-Foundation/FinRL @c0018c2 | MIT | (structure) | RL portfolio agents | Not adopted: earlier evidence says tabular GBDT is the right tool; RL needs far more data | **Rejected** |
| TauricResearch/TradingAgents @e8f5503 | Apache-2.0 | (structure) | LLM multi-agent analysis | Kept out of the numerical path (no evidence of benefit, not reproducible) | **Rejected** for prediction |
| KoshIndia/kite-algo-bot @b589d7f | MIT | `analytics/charges.py`, `docs/LIVE_CHECKLIST.md` | Zerodha-style Indian charges; live checklist | Our `costs.py` already matches (STT 0.1% delivery / 0.025% intraday sell, exchange, SEBI, stamp, GST, DP) | Verified, nothing to add |
| thecoderpanda/openquant @3f8aa22 | Apache-2.0 | README | Indian quant scaffolding | Nothing beyond what we have | — |
| luckyloneranger/backtest-platform @52ddbbc | none found | (structure) | — | **No licence: do not reuse code** | — |
