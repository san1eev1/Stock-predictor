# Integration log

| Date | Change | Files | Tests | Why |
|---|---|---|---|---|
| 2026-10-22 | Learn from every closed paper trade (all books) | `models/trainer.py` (`paper_trade_outcomes`, `judged_predictions`, `intraday_feedback`), `live/monitor.py` (`push_feedback`) | `tests/test_fno.py::test_trade_outcomes_feed_learning` | User requirement: every day's trades teach the models |
| 2026-10-22 | Stop fills 0.1% beyond the stop | `backtest/intraday.py` | `tests/test_intraday_backtest.py` | Audit #2 |
| 2026-10-22 | F&O fills cross half the spread | `paper/fno.py` (`fill`) | `tests/test_fno.py` | Audit #4 |
| 2026-10-22 | Monthly-frozen rule holdout; promotion bar rises with trial count | `models/trainer.py` | `tests/test_fno.py::test_trial_threshold_rises` | Audit #3 |
| 2026-10-22 | Circuit-limit guard in the long-term simulator | `backtest/portfolio.py` | `tests/test_model_backtest.py::test_no_buy_at_upper_circuit` | From qlib `limit_threshold` |
