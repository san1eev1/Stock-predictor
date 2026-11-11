# All Heavy Work on GitHub (Nov 11, 2026)

## Current Data Collection (runs nightly)
- ✅ `update-market-data.yml` → prices, intraday (5-min, 7d), FII/DII, delivery, bhavcopy, fundamentals, earnings
- ✅ `repo-scout.yml` → weekly repo search (Sundays)

## Model Training (2 runs daily at 18:30 & 21:00 IST)
- ✅ `train-models.yml` → long-term (weekly), intraday 12:30 & close (daily)

## Research & Testing (on-demand)
- ✅ `research.yml` → backtests (Alpha158, CPCV, point-in-time, trailing stops)
- ✅ `research-dl.yml` → heavy tests (foundation models, torch-based)

## Missing / To Add
- [ ] Different candle aggregations (1-min, 15-min, hourly)
- [ ] Gap-fill analysis (overnight moves)
- [ ] Candle pattern detection (optional)

## Mac Only (Minimal)
- 09:46: live picks (fetch prices, run inference, paper trade)
- 18:29: start train-models.yml
- 21:00: start train-models.yml
