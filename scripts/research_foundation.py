"""Experiment E-005 (GitHub, research-dl.yml): pre-trained time-series foundation models vs
our long-term model. Zero-shot (no training on our data): they bring outside data, the one
way around our data limit (E-002, E-003).

Models: Kronos-small (shiyu-coder/Kronos@67b630e, MIT; OHLCV, 45+ exchanges),
Chronos-Bolt-small (amazon, Apache-2.0; univariate), Moirai-small (Salesforce uni2ts,
Apache-2.0) if it installs. PatchTST needs training per dataset, so it isn't zero-shot and
is left out.

Protocol (fixed before running): month-end snapshots 2023-01 .. latest-1 month, today's
Nifty 250, context 256 trading days, forecast 20 trading days. Score = forecast 20-day
return. Target = realised 20-day return. Metric = cross-sectional rank IC per snapshot.
Compare with our walk-forward model score and plain 6-month momentum on the same
snapshots. A foundation model is useful only if blending it with our score (rank average)
raises IC with paired t >= 2.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.backtest import run as bt
from stockpredictor.models import longterm as M

CTX, H = 256, 20
d = Path(sys.argv[1] if len(sys.argv) > 1 else "market-data")
daily, indices, feats, labeled = bt.load_all(d)
px = daily.pivot(index="date", columns="symbol", values="close").sort_index()
cols = {c: daily.pivot(index="date", columns="symbol", values=c).sort_index()
        for c in ("open", "high", "low", "close", "volume")}
dates = px.index
month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).last()
snaps = [t for t in month_ends if t >= pd.Timestamp("2023-01-01")
         and dates.get_loc(t) + H < len(dates)]
print(f"{len(snaps)} snapshots, {px.shape[1]} stocks", flush=True)

ours = M.walk_forward(labeled, feats, 2023)
ours = ours[ours["date"].isin(snaps)].set_index(["date", "symbol"])["score"]


def realised(t):
    i = dates.get_loc(t)
    return px.iloc[i + H] / px.iloc[i] - 1


def contexts(t):
    i = dates.get_loc(t)
    win = slice(i - CTX + 1, i + 1)
    ok = px.iloc[win].notna().all() & cols["volume"].iloc[win].notna().all()
    return [s for s in px.columns if ok.get(s, False)], win


# --- models -------------------------------------------------------------------------
def kronos():
    sys.path.insert(0, os.environ.get("KRONOS_SRC", "kronos-src"))
    from model import Kronos, KronosPredictor, KronosTokenizer
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=512)

    def score(t):
        syms, win = contexts(t)
        x_ts = pd.Series(dates[win])
        y_ts = pd.Series(pd.bdate_range(dates[win][-1] + pd.Timedelta(days=1), periods=H))
        dfs = [pd.DataFrame({c: cols[c][s].iloc[win].to_numpy() for c in cols})
               .assign(amount=lambda f: f["close"] * f["volume"]) for s in syms]
        out = pred.predict_batch(dfs, [x_ts] * len(dfs), [y_ts] * len(dfs), pred_len=H,
                                 T=1.0, top_p=0.9, sample_count=1, verbose=False)
        return pd.Series([o["close"].iloc[-1] / f["close"].iloc[-1] - 1 for o, f in zip(out, dfs)],
                         index=syms)
    return score


def chronos():
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu",
                                               torch_dtype=torch.float32)

    def score(t):
        syms, win = contexts(t)
        ctx = torch.tensor(px[syms].iloc[win].to_numpy().T, dtype=torch.float32)
        q, mean = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        last = ctx[:, -1].numpy()
        return pd.Series(q[:, -1, 0].numpy() / last - 1, index=syms)
    return score


def moirai():
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    model = MoiraiForecast(module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
                           prediction_length=H, context_length=CTX, patch_size=32, num_samples=20,
                           target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)

    def score(t):
        syms, win = contexts(t)
        x = torch.tensor(px[syms].iloc[win].to_numpy().T, dtype=torch.float32)[..., None]
        obs = torch.ones_like(x, dtype=torch.bool)
        pad = torch.zeros(x.shape[:2], dtype=torch.bool)
        fc = model(past_target=x, past_observed_target=obs, past_is_pad=pad)
        med = fc.median(dim=1).values[:, -1]
        return pd.Series(med.numpy() / x[:, -1, 0].numpy() - 1, index=syms)
    return score


def ic(score: pd.Series, t) -> float:
    y = realised(t)
    j = pd.concat([score, y], axis=1, join="inner").dropna()
    return float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")) if len(j) > 30 else np.nan


models = {}
for name, make in (("kronos", kronos), ("chronos", chronos), ("moirai", moirai)):
    try:
        models[name] = make()
        print(f"{name}: loaded", flush=True)
    except Exception as exc:
        print(f"{name}: NOT available ({type(exc).__name__}: {exc})"[:300], flush=True)

rows = []
for t in snaps:
    base = ours.xs(t, level="date") if t in ours.index.get_level_values(0) else pd.Series(dtype=float)
    mom = px.loc[t] / px.iloc[dates.get_loc(t) - 126] - 1
    row = {"date": t, "ours": ic(base, t), "momentum": ic(mom, t)}
    for name, f in models.items():
        t0 = time.time()
        try:
            s = f(t)
        except Exception as exc:
            print(f"{t:%Y-%m} {name} failed: {exc}"[:200], flush=True)
            continue
        row[name] = ic(s, t)
        blend = pd.concat([base.rank(pct=True), s.rank(pct=True)], axis=1, join="inner").mean(axis=1)
        row[f"ours+{name}"] = ic(blend, t)
        row[f"{name}_sec"] = round(time.time() - t0)
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

r = pd.DataFrame(rows).set_index("date")
print("\nmean rank IC over snapshots:", flush=True)
for c in [c for c in r.columns if not c.endswith("_sec")]:
    print(f"  {c:16} {r[c].mean():+.4f}  (positive in {(r[c] > 0).mean():.0%})", flush=True)
for name in models:
    if f"ours+{name}" in r:
        diff = (r[f"ours+{name}"] - r["ours"]).dropna()
        t = diff.mean() / diff.std() * np.sqrt(len(diff)) if len(diff) > 2 and diff.std() else 0
        print(f"ours+{name} vs ours: {diff.mean():+.4f}, paired t = {t:.2f} -> "
              f"{'ACCEPT' if t >= 2 else 'REJECT'}", flush=True)
