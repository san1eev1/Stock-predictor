"""Experiment E-006 (GitHub, research.yml): can subscription data predict IPO listing gains?
Fetches NSE's IPO list and every mainboard IPO's final subscription, joins listing-day
prices from the bhavcopy and walk-forward tests the IPO model (models/ipo.py).
Acceptance (fixed before running): IPOs the model says 'Apply' (prob >= 0.65) must list up
more often than all IPOs, by >= 10 points, over >= 30 'Apply' IPOs out of sample."""
import sys
from pathlib import Path

import pandas as pd

from stockpredictor import store
from stockpredictor.data import bhav
from stockpredictor.data import ipo as IPO
from stockpredictor.models import ipo as M

d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
print("update:", IPO.update(d, backfill=700, pause=0.5, progress=lambda m: None), flush=True)
data = M.dataset(IPO.load_issues(d), IPO.load_subscription(d), bhav.load(d), store.load_indices(d))
listed = data.dropna(subset=["gain_close"])
print(f"mainboard IPOs with subscription: {len(data)}, with listing price: {len(listed)} "
      f"({listed['listing_date'].min():%Y-%m} .. {listed['listing_date'].max():%Y-%m})", flush=True)
print(f"listed up >1%: {(listed['gain_close'] > M.WIN).mean():.0%}, median listing-day gain "
      f"{listed['gain_close'].median():+.1%}", flush=True)
model, info = M.train_and_report(data)
for b in info["qib_buckets"]:
    print(f"QIB {b['lo']:>5g}-{b['hi']:<5g}x: {b['n']:4} IPOs, {b['listed_up']:.0%} listed up, "
          f"median {b['median_gain']:+.1%}", flush=True)
wf = info.get("walk_forward", {})
print("walk-forward:", wf, flush=True)
print("coefficients:", dict(zip(["const"] + M.FEATURES, [round(x, 2) for x in model.w])), flush=True)
ok = wf and wf.get("apply_n", 0) >= 30 and wf["apply_listed_up"] - wf["all_listed_up"] >= 0.10
print("ACCEPT" if ok else "REJECT", flush=True)
