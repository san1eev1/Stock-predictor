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

## Round 2 — 2026-10-22

| Repo @ commit | Licence | Modules read | Useful technique | Compatible? | Status |
|---|---|---|---|---|---|
| AI4Finance-Foundation/FinRL-Meta @286fef6 | MIT | (structure) | Market environments for RL | RL is rejected for prediction (round 1) | — |
| QuantConnect/Lean @f3d841e | Apache-2.0 | `Common/Orders/Fills/EquityFillModel.cs` (`StopMarketFill`: fill at the open when the bar gaps past the stop), `Algorithm.Framework/Risk/*` (TrailingStop, MaximumDrawdownPercentPerSecurity/Portfolio, MaximumSectorExposure) | Gap-aware stop fills; trailing-stop and sector-exposure risk models | Ideas re-implemented | Stop fill: our 0.1% `STOP_SLIPPAGE` stands in (summaries lack bar opens); sector cap already exists (`Rules.max_per_sector`); trailing stop **prototyped** (E-004) |
| nautechsystems/nautilus_trader @e19fa98 | LGPL-3.0 | tree only (risk engine, Rust/Cython) | Pre-trade risk engine, order state machine | Relevant only if live orders are ever built | Recorded for later |
| Quant-IQ/quantiq @6582de1 | **none found** | README, docs | NSE end-to-end design docs; `backtest/` is empty | No code to learn from, no licence | **Unsuitable** |
| pmorissette/bt @3adebef | MIT | (structure) | Tree-of-algos portfolio backtests | Our simulator already covers our needs | — |
| PyPortfolio/PyPortfolioOpt @a6638d2 | MIT | `pypfopt/hierarchical_portfolio.py` (HRPOpt) | HRP / risk-parity weights | With 5 holdings the gain is small; volatility sizing already tested (worse) | Candidate, low priority |
| dcajasn/Riskfolio-Lib @7b1b26a | BSD-3 | (structure) | CVaR / drawdown-risk portfolio optimisation | As above | Candidate, low priority |
| ceyyy427/finathink @f2520b9 | MIT | (structure) | Point-in-time research tooling | Our bhav-based point-in-time universe covers it | — |

## Round 3 — 2026-10-25 (all 21 remaining watchlist repos)

| Repo @ commit | Licence | What was read | Finding | Status |
|---|---|---|---|---|
| HKUDS/Vibe-Trading @41f333b | MIT | `agent/backtest/engines/india_equity.py`, `china_a.py` (`_blocked_by_limit`), `factor_costs.py` (`apply_adv_capacity`), kill-switch tests | Indian delivery engine: T+1, no overnight shorts, one ±20% circuit band, same cost stack as ours (STT 0.1% both sides, exchange 0.00297%, SEBI, stamp 0.015% buy, GST 18%) | **Verified ours matches**; our circuit guard is finer (5/10/20% bands). Capacity = our 1%-of-volume cap |
| cvxgrp/cvxportfolio @351c782 | GPL-3.0 | `cvxportfolio/costs.py` (`StocksTransactionCost`: a·|z| + b·σ·|z|^1.5/V^0.5) | Square-law/3⁄2-power market impact | Ideas only (GPL). Impact is negligible at ₹1 lakh; matters only at much larger capital. Recorded |
| ricequant/rqalpha @6a996af | own (Chinese) licence | `utils/price_limits.py`, simulation matcher tests | Limit-up/down blocking, volume-share matching | Same ideas as qlib `limit_threshold`, already integrated |
| zvtvz/zvt @8509990 | MIT | factor and trader modules | A-share factor framework with a simulated account | Nothing new for us |
| fasiondog/hikyuu @47e4236 | Apache-2.0 | signal/slippage tests | C++ A-share system | Nothing new |
| kernc/backtesting.py @ca2e261 | **AGPL-3.0** | structure, optimisation example | Single-asset event backtester, parameter heatmaps | Ideas only; our tuning already uses walk-forward and holdouts |
| cloudQuant/backtrader @7196567 | GPL-3.0 | structure | Event-driven backtester | Nothing new |
| Lumiwealth/lumibot @e14453b | GPL-3.0 | broker tests | Multi-broker live trading | Relevant only for real orders (not authorised) |
| coding-kitties/investing-algorithm-framework @f8e82c7 | Apache-2.0 | live-hardening tests | Live-trading hardening patterns (idempotent orders, restart) | Recorded for if real orders are ever built |
| OpenByteInc/QuantDinger @af83394 | Apache-2.0 | resource guard, data sources | LLM-assisted platform | LLMs stay out of the prediction path |
| cuemacro/finmarketpy @afeaef2 | Apache-2.0 | `backtest/tradeanalysis.py` | FX-focused backtests | Not applicable |
| mbk-dev/okama @4d222a8 | MIT | structure | Portfolio analytics on monthly data | Not applicable |
| fortitudo-tech/fortitudo.tech @1d839a9 | GPL-3.0 | structure | Entropy pooling, CVaR | Not applicable at 5 holdings |
| shashankvemuri/Finance @2079903 | MIT | scripts | Collection of TA scripts | Nothing new (we have ~130 features) |
| wangzhe3224/awesome-systematic-trading @1cab7c2 | MIT | README | Curated list | Leads below |
| leoncuhk/awesome-quant-ai @9cae34a | Apache-2.0 | README | Curated list | Leads below |
| hummingbot, pycryptobot, botflow, polymarket-kalshi-weather-bot | various | structure | Crypto / prediction-market bots | **Not applicable** (different markets) |
| bqth29/simulated-bifurcation-algorithm @dcff6d6 | MIT | structure | Combinatorial optimiser | Not applicable |

### New leads from the curated lists (need new data or compute: ask before building)
- **FII/DII daily flows and shareholding patterns** (via Tapetide MCP / NSE): new information we don't use. NSE serves only the current day's FII/DII figures; a history must be collected daily before it can be tested.
- **Kronos** (shiyu-coder/Kronos): OHLCV foundation model pre-trained on 45+ exchanges. Pre-training on outside data is one way around our data limit (E-002, E-003). Test cost: heavy CPU inference on GitHub.
- **PatchTST / Moirai**: time-series transformers; same caution as Kronos.
