# Daily guide — how to run the Stock Predictor

Everything runs on your Mac from VS Code. The dashboard is a **web page on your own Mac**
(`http://localhost:8501`): nothing to install, nothing sent anywhere.

There are **three ways** to run it each day. Pick one.

| Way | Best if | What you do |
|---|---|---|
| **A. Autostart** (recommended) | Your Mac is usually on during the day | Turn it on once; it starts by itself every weekday at 9:00 |
| **B. Press F5** | You open VS Code every morning | Press F5 before 9:15, leave it running |
| **C. Quick daily run** | You can't leave the Mac on | Run one task once a day, any time after 5:15 PM |

---

## First time only (about 10 minutes)

1. **Terminal** (Spotlight → "Terminal"): `brew install python@3.12 libomp git`
2. **VS Code** → *File → Open Folder…* → **Stock-predictor**.
3. Accept **"Install the recommended Python extension"** if asked.
4. **Terminal → New Terminal**, then: `git pull`
5. `Cmd+Shift+P` → **Tasks: Run Task** → **1. First-time setup**
   (creates `.venv`, installs everything, downloads the data — wait for it to finish).
6. `Cmd+Shift+P` → **Python: Select Interpreter** → choose the one with **`.venv`**.
7. Angel One: open `.env`, fill the four `ANGEL_...` values, save, then run the task
   **Angel One: check login**. You should see `Login OK`.

---

## A. Autostart (set once, then nothing to do)

1. `Cmd+Shift+P` → **Tasks: Run Task** → **Autostart: ON (09:00 Mon-Fri, stops itself at night)**.
2. Keep the MacBook **plugged in with the lid open** on trading days
   (System Settings → Battery → Options → *Prevent automatic sleeping on power adapter* helps).
3. Each weekday at 9:00 it starts in the background (or as soon as the Mac wakes), runs all day,
   makes the after-close decision, learns, and stops after 21:30.
4. To look at results: run **Web dashboard only** (or `python -m stockpredictor app`) any time.
5. Check it worked: task **Autostart: show today's log**. Turn off: **Autostart: OFF**.

## B. Press F5 each morning

1. Open VS Code with the Stock-predictor folder **before 9:15 AM**.
2. Press **F5** → choose **▶ Start Stock Predictor** (first time only it asks).
   Alternatively double-click **`Start Stock Predictor.command`** in Finder.
3. The dashboard opens in your browser. Leave VS Code running until ~6 PM.
4. `Ctrl+C` in the terminal to stop.

## C. Quick daily run (Mac doesn't need to stay on)

1. Any time **after 5:15 PM** on a trading day: `Cmd+Shift+P` → **Tasks: Run Task** →
   **3. Quick daily run**.
2. It downloads the day's data, retrains on it and on judged paper trades, makes the long-term
   decision, updates paper trading, and prints the **top 10 buy** candidates.
3. If you run it between 9:46 and 10:15 AM it also makes the intraday picks.
4. Limits: no minute-by-minute stop-losses or intraday square-off while the Mac is off, and
   missed days are caught up next time (up to 30 days).

---

## What you see when you start it (any time of day)

Within about a minute the terminal shows something like:

```
Starting up: getting data, training and catching up...
Long-term model retrained on data up to 2026-09-17
Decision for 2026-09-24: buy DIVISLAB, TITAN, ... ; sell none
Intraday picks made (late start): LONG LTM 2 @ 4076.20; ...        <- only if the market is open
===== Accuracy now (2026-09-25 11:08) =====
Intraday today : buys 6/10 up, sells 7/10 down -> 65% right (random picks: 52%)
Intraday judged: 58% right on 120 picks over 6 days (random: 50%)
Long-term open : 12/20 picks on track (judged after 1 week)
Long-term judged: 55% right on 40 picks over 2 days (random: 50%)
```

So whenever you run it, it **trains if it hasn't today, catches up on missed days, makes any
missing picks** (intraday picks even on a late start, until 2:30 PM, at the prices of that moment),
and prints the **accuracy right now**. The scoreboard is printed again every 15 minutes and is
always visible in the dashboard sidebar (📊 Accuracy now) and at the top of the Accuracy page.

If GitHub's evening data update is late, the program downloads the day's prices itself after 5:45 PM.

## What happens during a trading day (A and B)

| Time (IST) | What the program does | Where to see it |
|---|---|---|
| 9:15 | Starts watching live prices (Angel One real-time) | sidebar: *Live monitor 🟢* |
| 9:20 | Fills last night's long-term paper orders at market prices | Paper trading → Long-term |
| **9:46** | **Intraday: top 10 buy + top 10 sell candidates**, top 3 + 3 paper-traded on a fresh ₹1 lakh | **Intraday picks** |
| every minute | Stop-loss / target checks (paper) · stop-loss alerts (your portfolio) | 🔔 Alerts |
| every 15 min | New news, live re-ranking of long-term picks | Long-term picks → *Live ranking* |
| **12:30** | **Intraday square-off** and scoring (trading stops; learning doesn't) | Paper trading → Intraday → *Day by day* |
| all day | Background training of the intraday model; newest GitHub-trained models downloaded every 15 min | Model → *Continuous training* |
| **15:32** | Today's full session added to history, intraday model retrains | Model |
| **17:15+** | Data update → **retrain** → **next week's top 10 buys** → paper orders → light self-tuning | **Long-term picks** |
| every hour | GitHub retrains + self-tunes the long-term and news models on all history since 2005 | Model (☁️ status) |

## What to look at in the dashboard

1. **Long-term picks** — top 10 buy candidates (expected to beat Nifty **next week**; no sell
   candidates), plus **Sell now** for holdings the model expects to fall. Paper holds the top 3.
2. **Intraday picks** — top 10 buy + top 10 sell for today (3 + 3 traded), entry / stop-loss / target and live result.
3. **Paper trading — Long-term** (buy-only) and **Paper trading — Intraday** (3 buy + 3 sell
   trades a day, separate sections), each with P&L after costs.
4. **My portfolio** — enter your real Groww trades (➕ Add a trade); live P&L and stop-loss alerts.
5. **Accuracy** — how often picks were right (long-term judged after 1 week, intraday same day)
   vs random picks; the live strategy race.
6. **Model** — what the model relies on, backtests, and every retrain / tuning run.

## Weekly (optional, 5 minutes)

- `git pull` in the VS Code terminal to get improvements, then
  `.venv/bin/pip install -e .`
- Task **Keep training now (retrain + self-tune)** if you want an extra tuning round.
- Glance at **Accuracy** and **Model → Continuous training** to see whether accuracy is improving.

## If something goes wrong

| Problem | Fix |
|---|---|
| `command not found: python` | Run `source .venv/bin/activate` first, or use the VS Code tasks |
| Dashboard shows old dates | Task **Get latest data** |
| `Angel One: login FAILED` | Check the 4 values in `.env`; the TOTP secret is the text, not the 6-digit code |
| Sidebar says *Live monitor ⚪* | It isn't running — press F5 or check the autostart log |
| Anything else | Copy the last lines of the terminal (or `logs/daily.log`) and ask |

> Paper trading only. Nothing places real orders. Predictions are probabilities, not guarantees.
