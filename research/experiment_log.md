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

## E-001 Alpha158 extra features (2026-10-22) — **REJECTED**
Script: `scripts/research_alpha158.py` (run 39735225219). 12 extra features on top of 129.
| | Weekly IC (t) | CAGR | Sharpe | Worst |
|---|---|---|---|---|
| Baseline | 0.0492 (7.4) | 21.6% | 0.85 | −43.0% |
| + Alpha158 extras | 0.0500 (7.7) | 22.4% | 0.87 | −40.2% |
Paired IC difference +0.0007, **t = 0.34** over 614 weeks: indistinguishable from noise, so not
adopted (the small CAGR gain is within the noise; today's-universe backtest).
Lesson: our 129 features already cover these families (trend R², slopes, 52-week range, up days).

## E-002 CPCV + DoubleEnsemble-lite (2026-10-22) — running
Script: `scripts/research_cpcv_double_ensemble.py` (run 39737604311). Rule: better on ≥80% of the
15 CPCV paths and paired t ≥ 2.
