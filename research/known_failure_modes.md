# Known failure modes and lessons

- **Survivorship bias**: today's index members projected back inflated long-term CAGR from ~12% to ~22% (E-000).
- **Look-ahead in `align_to_daily`**: fixed (previous-day factor); covered by tests.
- **Rule tuning picked "never trade"**: fixed with MIN_TRADE_DAYS, a holdout and a shadow period.
- **Rolling holdout reused nightly**: fixed (monthly freeze plus trial-count bar).
- **Stops filled exactly at the stop**: fixed (STOP_SLIPPAGE).
- **Live features ≠ training features**: delivery/results inputs were blank in live scoring for weeks (fixed 2026-10-05). Always compare live columns with the model's.
- **DB migration renamed a referenced table**: paper trades failed to save for 4 days (fixed: legacy_alter_table).
- **Operations**: Mac sleep kills the 9:46 picks; Angel One throttles bursts of candle requests.
- **Rejected ideas (worse out-of-sample)**: crash guard, low-vol tilt, Clenow momentum, volatility targeting, week+month+recent ensembles, intraday sell signal as a long-term filter, RL/LLM in the prediction path.
- **Intraday**: ~0.05%/trade gross vs 0.32% costs: no net edge to date.
- **Exit overlays on the momentum book** (crash guard, vol targeting, trailing stops 10–25%): all cut return more than they cut drawdown (E-004).
