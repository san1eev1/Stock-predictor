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

## E-011 F&O positioning (open interest) features (2026-11-14) — **intraday ACCEPTED, long-term rejected by rule**
Data: NSE F&O bhavcopy 2012-01 → 2026-11, 454 stocks, 3,664 days (runs 43562018842, 43562022162).
E-011b intraday 'until close' model (yesterday's positioning): daily IC 0.0946 → 0.1018
(+0.0072, t = 9.01, better on 57% of 2,433 days) → **ACCEPTED, integrated** (features/intraday.attach_fo,
used by training and live picks; Mac downloads the last 2 years of fobhav/).
E-011a long-term: walk-forward IC 0.0499 → 0.0565 (t = 2.87); CPCV better on 15/15 paths (t = 10.55);
but the 5-stock backtest did not improve (CAGR 22.9% → 23.0%, Sharpe 0.89 → 0.89, worst −40.5% →
−45.1%) → rejected by the fixed rule. The ranking gain barely reaches the portfolio because the
live score is 90% momentum; next hypothesis: a larger model weight with the F&O features.
First new information source to beat price-only features (E-001…E-010 all failed).
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

## E-013 Option-selling lab (live paper, started 2026-11-27)
Hypothesis: Nifty weekly index options carry a volatility risk premium (Coval & Shumway 2001;
Bakshi & Kapadia 2003; VIX typically 2-4 pts above subsequent realized vol), harvestable by
hedged intraday selling after costs. Five books, Rs 1.5 lakh each, Angel One LTPs, exit 15:15:
9:20 iron fly (25% leg stop, stop-to-cost), 11:00 iron fly, 9:30 iron condor with 50% profit
target / 1x credit stop, same condor only when VIX > 20-day realized vol, expiry-day (Tuesday)
iron fly. Judge after >= 60 traded days each: daily P&L t-stat, worst day, deepest fall.
Caveat: public 9:20-straddle backtests show the year's result depends on a handful of days.

## E-014 Intraday profit levers (2026-11-27, research run 45899279970)
'Until close' model, walk-forward scores, 2,441 out-of-sample days, 545k stock-days. Daily
long-short spread per pair (mean long move - mean short move), equal size:

| Variant | Gross | Net 0.37% (Angel-like) | Net 0.19% (0.03% broker) | Net 0.11% (+limit orders) |
|---|---|---|---|---|
| top/bottom 3, exit 15:15 | +72.8 bp, t 18.6 | -1.2 bp, t -0.3 | **+34.8 bp, t 8.9** | +50.8 bp, t 13.0 |
| top/bottom 5, exit 15:15 (now) | +63.5 bp | -10.5 bp, t -3.5 | +25.5 bp, t 8.3 | +41.5 bp |
| top/bottom 10, exit 15:15 | +52.2 bp | -21.8 bp | +14.2 bp | +30.2 bp |
| top/bottom 5, exit 12:30 | +36.0 bp | -38.0 bp, t -18 | -2.0 bp | +14.0 bp |
| sector pairs 5, exit 15:15 | +55.6 bp | -18.4 bp | +17.6 bp | +33.6 bp |

Last two years gross (top/bottom 5, close): +30.5 bp, t 5.5 (half the full-period edge).
News since the last close vs 9:45->15:15 (187 days, 5,304 stock-days): IC t -0.31; good news
+6.9 bp vs market (t 1.0); bad news +2.1 bp (t 0.2) -> no intraday predictive power.

Conclusions (fixed rule: net t >= 2 and beats now with paired t >= 2):
- The model has a real gross edge; at Angel-like 0.37% round trips NOTHING passes (best: 3 a
  side to the close, break-even). Costs are the binding constraint, not the model.
- At 0.19% (0.03%-brokerage broker) or lower, every close-exit variant passes; 3 a side best.
- Exit at 15:15 beats 12:30 in every case; the 12:30 book loses at any cost >= 0.19%.
- Fewer picks are better (3 > 5 > 10); sector-neutral pairs are worse than plain top/bottom.
- News does not predict the rest of the day: keep it as a veto only, not a ranking input.
Live gap: live stock picking (last 60 days) is ~0 vs +30 bp backtest recently -> watch.

## E-015 Morning candles (2026-11-27, research run 45901706002) — candle book rule changed
'Until close' model, walk-forward, 2,441 days. Each 9:15-9:45 candle check, stocks agreeing
minus disagreeing, 9:45 -> 15:15: closed up -5.1 bp (t -7.1), top half of range -2.8 (t -4.1),
above VWAP -1.6 (t -2.5), ORB-15 up -0.8 (t -0.4), most 1-min green +0.7 (t +0.5); the 0-5
candle score has IC t -11.0. The morning candle tends to REVERSE over the rest of the day
(consistent with the literature: candlestick rules add nothing after costs; Marshall, Young &
Rose; Duvinage et al.; Thai / South African replications).
Candle-book variants, top 3 a side, per position: the live rule (weight 0.5, cut at entry)
-34.6 bp net of 0.37% (6% winners) - the source of the live -22%. Holding to 15:15 is best at
every weight (w 0.25: +0.1 bp at 0.37%, +18.1 bp t 9.7 at 0.19%); 1% stops are worse than
holding. Change applied to the paper candle book: weight 0.25, no cut at entry, 3% safety
stop, no target (paired t vs the old rule +18.9). Main books unchanged (their stops are tuned
nightly by profit).

## E-016 Five intraday ideas (2026-11-28, research run 46123630622) — none adopted
'Until close' model, walk-forward, 2,442 days, 3 a side, held to 15:15. Per position per day,
net of 0.33% / 0.19% round trip (Zerodha's ~0.21% sits near the second column):

| Variant | Net 0.33% | Net 0.19% | vs base (paired t) |
|---|---|---|---|
| base: model top/bottom 3 | +3.2 bp (t 1.6) | +17.2 bp (t 8.8) | - |
| A candle reversal w 0.25 | +1.0 | +15.0 | -1.97 |
| A candle reversal w 0.5 | -2.1 | +11.9 | -3.36 |
| B pure 30-min reversal (no model) | -27.3 (t -11.9) | -13.3 | -10.8 |
| B model + 30-min reversal | -4.2 | +9.8 | -3.5 |
| C inverse-volatility sizing | -1.1 | +12.9 | -5.4 |
| D strong days only (> median, 47% of days) | +12.7 (t 4.2) per traded day | +26.7 | fewer days |
| D strong days only (> 70th pct, 29% of days) | +17.5 (t 4.5) per traded day | +31.5 | fewer days |

E. Regimes: the base edge is about the same in every India VIX tercile and whether the market
was down, flat or up in its first 30 minutes (gross ~+34-37 bp, t 9-11 each): no regime filter.
Conclusions: the model already uses the morning information well - reversing the candle or the
30-minute move, or sizing by volatility, all do worse. Strong-day filtering raises the profit
per traded day but at Zerodha costs lowers the total (26.7 x 47% = 12.6 bp a day vs 17.2): the
nightly rule tuning (skip_quantile, chosen by profit on the new costs) decides it.
