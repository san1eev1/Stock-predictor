# Experiment log (never overwrite; append)

## E-000 Survivorship-free long-term backtest (2026-10-22, GitHub research.yml)
Script: `scripts/research_survivorship.py`. Data: bhavcopy 2005→ (top 400 by turnover per day).
| Universe | IC | 2015–2026 | Last 5 years |
|---|---|---|---|
| Today's Nifty 250 (biased) | 0.042 | 21.6%/yr, Sharpe 0.84, worst −41.6% | 19.4%/yr, Sharpe 0.70 |
| Point-in-time (incl. stocks that left) | 0.046 | **12.4%/yr, Sharpe 0.42, worst −51.6%** | 12.3%/yr, Sharpe 0.37, worst −51.6% |
Conclusion: about **9 points a year of the old backtest was survivorship bias**. The signal itself
(IC) holds, but the strategy's net edge over Nifty (~9–11%/yr in the same period) is small with
large drawdowns. All earlier long-term CAGR figures must be read with this in mind.

## E-001 Alpha158 extra features (2026-10-22) — queued
Script: `scripts/research_alpha158.py`. Fixed protocol in the script header (paired IC t ≥ 2 and
CAGR/Sharpe not worse). Limitation: evaluated on today's-universe backtest (the script reuses
the standard loader); if accepted, re-check on the point-in-time universe before integrating.
