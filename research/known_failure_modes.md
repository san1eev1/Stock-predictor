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
- **Silent 2-week paper-trading outage (2026-10-27 → 11-10)**: the Mac could not resolve or keep TLS to apiconnect.angelone.in (613 DNS failures, SSL "record layer"/"bad record MAC" errors) and Yahoo also failed, so no picks were made. Now: a red dashboard banner when live prices aren't from Angel One during market hours. Root cause is the Mac's network (DNS/Wi-Fi), not the code.
- **Deleting .venv under a running app** (2026-11-11): broke Angel One TLS (certifi path) until rebuilt. Never remove the environment of the running service.
- **The Mac's network drops uploads above ~16–48 KB** (2026-11-11: "SSL bad record mac", "remote end hung up"; 16 KB pushes pass, 48 KB fail; downloads fine). Explains Angel One SSL errors and failed feedback pushes. Since 2026-11-11 paper trading runs on GitHub (paper-trading.yml + cloud-evening), so nothing depends on the Mac's uploads; the initial state was seeded in 15 KB pieces (cloud.seed_state).

## Costs larger than the intraday edge (measured 2026-11-27)
P&L attribution of the live intraday paper books over 90 days (market move / sector move /
stock picking / costs): 12:30 book -Rs 3,596 net = +17 market +799 sector -657 picking
-3,755 costs; close book -4,151 = +192 +923 -1,912 -3,355; candle book -2,206 = +23 -97 -54
-2,079. Costs average 0.37% per round trip (0.1% brokerage with Rs 5 minimum each side, STT,
charges, 0.05% slippage per order) on ~Rs 9k trades. Stock picking before costs is about
zero live. Levers: cheaper execution (0.03% brokerage, limit orders, futures for liquid
names), bigger and fewer trades, longer holds; better information before more trades.

## Splits looked like crashes in paper books (fixed 2026-11-27)
US SNDK 8:1 split: bought at $809 pre-split, marked at $94 post-split, stop-loss sold at
-88% (the whole US book's -18%). paper/splits.py now adjusts held positions and re-opened the
wrongly stopped position; the Mac skips stops on 40%+ single moves.
