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

## E-002 CPCV + DoubleEnsemble-lite (2026-10-22) — **REJECTED**
Script: `scripts/research_cpcv_double_ensemble.py` (run 39737604311). 15 CPCV paths (6 blocks,
2 test blocks each, 14-day purge), 2008–2026 weekly data.
Baseline IC 0.0747 ± 0.0131 vs DoubleEnsemble-lite 0.0750 ± 0.0137; better on 60% of paths,
**paired t = 0.79** (needed ≥ 80% and t ≥ 2). Not adopted: +3× training cost for no measurable gain.
Lesson: the long-term model is data-limited, not model-limited (consistent with the earlier
ensemble and deep-learning findings). CPCV itself works and gives tighter comparisons: keep the
script as the standard harness for future model changes.

## E-003 / E-004 (2026-10-22) — **both REJECTED** (run 39749205814)
E-003, evaluated on the point-in-time universe (1,930 stocks ever in a day's top 250):
| Trained on | IC | CAGR | Sharpe | Worst |
|---|---|---|---|---|
| Survivors (production) | 0.0452 | 14.6% | 0.49 | −55.3% |
| Point-in-time history | 0.0489 | 10.2% | 0.34 | −51.1% |
Paired IC t = 0.66: no significant ranking gain, and lower CAGR/Sharpe. Keep training on survivors.
E-004 trailing stop: chosen on 2015–20 = 20% (best Sharpe there, only 0.14). Confirmation 2021–26:
worst −46.7% vs −55.3% (better) but CAGR 18.3% vs 22.1% (−3.8 points, over the 1-point limit).
Tighter stops (10–15%) wrecked returns (whipsaw). Same lesson as the crash guard and vol targeting:
cutting drawdowns with exits costs more return than it saves for this momentum book.
Original plan below kept for the record.
Script: `scripts/research_pit_training.py`. Both evaluated on the point-in-time universe.
E-003: train on point-in-time history vs survivors (rule: IC paired t ≥ 2, CAGR and Sharpe not worse).
E-004: trailing stop (LEAN), level chosen on 2015–20 and confirmed on 2021–26 (rule: worst fall
≥ 5 points better, CAGR ≥ −1 point, Sharpe not worse).

## Round 3 study (2026-10-25): no new experiment
All 21 remaining watchlist repos were inspected (see repository_matrix.md, round 3). None offered a
technique that is both new to us and testable on our current data: Indian costs and circuit rules
are confirmed (Vibe-Trading), and impact/capacity models are irrelevant at ₹1 lakh. The open leads
need new data (FII/DII flow history, shareholding) or heavy compute (Kronos), so they wait for
approval rather than becoming a forced experiment.

## E-005 Foundation models (Kronos, Chronos-Bolt, Moirai) vs our model (2026-10-25) — **REJECTED**
Script: `scripts/research_foundation.py` (run 40167589634), workflow `research-dl.yml` (torch only in that job).
44 month-end snapshots (2023-01 to 2026-10), 260 stocks, 256-day context, 20-day forecast window.
Zero-shot ranking skill (cross-sectional Spearman IC per snapshot):
| Model | Mean IC | Positive | vs ours (paired t) |
|---|---|---|---|
| Our walk-forward | +0.0421 | 68% | — |
| Kronos (OHLCV, pre-trained) | +0.0119 | 34% | — |
| Moirai (univariate, pre-trained) | +0.0037 | 32% | — |
| Momentum (6-month) | +0.0072 | 52% | — |
| Ours + Kronos (rank blend) | +0.0357 | 39% | t = 0.07 |
| Ours + Moirai (rank blend) | +0.0316 | 45% | t = −0.18 |
Chronos-Bolt failed to load (dependency issue); conclusion unchanged. No foundation model improves
our score when blended. Our model's data-limited constraint (E-002, E-003) stands; outside pre-training
does not overcome it.

## FII/DII flows collection started (2026-10-25)
`data/fiidii.py`, `fiidii-update` in the evening data workflow → `market-data/fiidii.csv`.
Test as a market-wide feature once ≥ 1 year has accumulated (≈ Nov 2027).

## E-006 IPO listing-gain model (2026-11-11) — **ACCEPTED** (run 42972054312)
297 mainboard IPOs with subscription, 238 with listing-day prices (2023-12 .. 2026-11; older
listings did not match the bhavcopy join — to investigate). Base: 65% listed >1% up, median +9.6%.
QIB demand buckets: <3×: 26% up (median −5%); 3–10×: 36% (−1%); 10–50×: 57% (+5%);
50–100×: 86% (+19%); 100×+: 95% (+29%).
Walk-forward (164 IPOs, each year judged by a model trained on earlier years only):
'Apply' (p ≥ 0.65) 86 IPOs → 88% listed up, median +23%; 'Avoid' (p < 0.45) 64 → 31%, median −2.6%;
all → 65%, +6.9%. +24 points over the base rate with 86 calls: passes the fixed rule.
Coefficients (standardised): QIB 1.12, size 0.34, NII 0.33, retail 0.11, recent listings 0.11,
Nifty 20d 0.07. Caveat: only ~3 years of listings, mostly a strong IPO market. NSE past-issue list + final
subscription by QIB/NII/retail for every mainboard IPO, listing-day prices from the bhavcopy;
logistic regression, walk-forward by year. Rule: 'Apply' calls (p >= 0.65) list up >= 10 points
more often than all IPOs, over >= 30 out-of-sample calls.

## E-007 Candle experiment (live paper, from 2026-11-11)
Own ₹10,000 book (`intraday_candle`): the close model's 25 best buys + 25 best sells re-ranked
50/50 by model rank and five 9:15–9:45 one-minute candle checks; best 10 a side; cut as soon as a
trade slips below entry; rest to 15:15. Candle data from Yahoo (Angel One kept for the main
books). Compare with the main 'until close' book after ≥ 20 trading days (paired daily P&L,
t ≥ 2) before adopting anything. Expected risk: cutting at entry turns many trades into small
cost-sized losses (0.37% round trip).

## E-008 Mid-day review vs holding (2026-11-11) — **SWITCHED OFF** in the main books (run 42976081256)
59 days, same out-of-sample picks (10+10, ₹10k ×5 margin, real costs): hold −₹200/day (total
−₹11,826), review −₹753/day (total −₹44,426); review worse on 98% of days, paired t = −11.31.
Morning winners mostly reverse; every cut and top-up pays costs again. Kept only in the candle
experiment book; a toggle on the intraday page turns it back on for all books.
Also visible: holding 20 trades/day loses ~₹200/day, about the cost of 20 trades (0.37% of
₹50k exposure ≈ ₹185): the intraday model's edge does not cover costs at this trade count.
User rule (live in every intraday book from 2026-11-12): every 30 min from 10:15 cut trades that
lost or moved < 0.3%, add freed buying power to the 1–2 winners up ≥ 0.5% (≤ 40% in one stock,
top-ups stopped at their own entry). Test: `scripts/research_review.py` — same out-of-sample
picks, Yahoo 5-minute candles (~60 days), hold vs review, paired daily P&L. Keep if t ≥ 2 over
≥ 30 days; switch off if t ≤ −2; otherwise keep collecting live.

## Decision (2026-11-11, user): fewer, stronger intraday trades
After E-008 showed 20 trades/day lose about their costs: every intraday book keeps the 25+25
shortlist (all judged and learned from) but trades only the 3–5 strongest a side (the nightly
tuning's number, clamped to 3–5), ₹10,000 total with no margin, held all day (learned stop/target,
square-off). Mid-day review removed from the code. Candle book: same counts, keeps cut-at-entry.

## E-009 Market intraday momentum on Nifty (2026-11-14) — **REJECTED** (run 43555362311)
Gao, Han, Li & Zhou (2018, JFE) on Nifty 50, Angel One 5-min bars 2017-01 → 2026-11 (2,446 days).
corr(prev close→9:45, 15:00→close) = +0.010; strategy gross +0.11 bp/trade (t 0.27, hit 51%);
net of 0.03% (index futures) −2.89 bp (t −7.3); volatile months no better. The US effect is
absent in India in this period.

## E-010 Residual momentum and overnight/intraday momentum (2026-11-14) (run 43555365598)
| | Weekly IC (t) | CAGR | Sharpe | Worst |
|---|---|---|---|---|
| Baseline | 0.0504 (7.6) | 22.5% | 0.88 | −40.5% |
| + residual momentum (Blitz et al. 2011) | 0.0495 (7.5) | 23.7% | 0.92 | −40.7% |
| + overnight/intraday momentum (Lou et al. 2019) | 0.0537 (8.2) | 25.6% | 0.99 | −41.4% |
Residual momentum: paired IC −0.0009, t −0.41 → **REJECTED**. Overnight/intraday: paired IC +0.0034,
t = 1.64 → below the bar (REJECT by the rule) but the most promising result so far; re-tested
through the CPCV harness (E-010b). Backtests are on today's universe (survivorship-biased).

## E-010b Overnight/intraday momentum through CPCV (2026-11-14) — **REJECTED** (run 43555837904)
15 paths: baseline IC 0.0749 vs 0.0763 with the features; better on 67% of paths, paired t = 1.03
(rule: ≥ 80% and t ≥ 2). The walk-forward gain in E-010 was not robust. Every price-derived idea
(E-001…E-010) has now failed: next is new information — F&O open interest (E-011).

## E-011 F&O positioning (open interest) features (2026-11-14) — data downloading
`data/fobhav.py` (NSE F&O bhavcopy 2012→, stock futures/options OI per stock and day),
`scripts/research_fo_oi.py` (long-term: walk-forward + CPCV) and
`scripts/research_fo_oi_intraday.py` (intraday close model, yesterday's positioning).

## E-012 First US (S&P 500) long-term model (2026-11-14, us-daily run 43558962579)
Same model and rules as India, MARKET=us, 503 stocks, Yahoo prices since 2005, $1,000 book.
Weekly IC 0.027 (t = 3.1), positive in 56% of weeks (India 0.050).
| Strategy (2015–2026, today's S&P 500 list) | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Model, daily decisions | 20.6% | 36.0% | 0.57 | −49.9% |
| Model, weekly | 17.6% | 35.3% | 0.50 | −46.4% |
| Model, weekly, max 2 per sector | 21.2% | 35.3% | 0.60 | −51.8% |
| S&P 500 | 11.8% | 17.6% | 0.67 | −33.9% |
| Equal-weight S&P 500 stocks | 14.3% | 19.6% | 0.73 | −34.1% |
Higher return than the index but lower Sharpe and much deeper falls; survivorship-biased (today's
list), so the real edge is smaller still. Top features: market 63-day return, crude, VIX, dollar,
market volatility, sector return, beta. The US book starts trading on Monday's close.
