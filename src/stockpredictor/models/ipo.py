"""IPO model: from past NSE mainboard IPOs, how likely is a new one to list above its issue
price? Trained on GitHub (needs the bhavcopy for listing-day prices), used on the Mac for
the issues open now.

Inputs, all known when bidding closes (no grey-market premium: it has no reliable source):
  times subscribed by QIBs, non-institutional (NII) and retail investors, issue size,
  Nifty's last 20 trading days, and how the previous 10 mainboard IPOs listed.
Target: listing-day close above the issue price by more than ~1% (costs).
Logistic regression (standardised inputs, small L2): each input's push in the decision can
be shown as a reason. Validation: walk-forward by year (train on earlier listings only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["log_qib", "log_nii", "log_rii", "log_size_cr", "nifty_20d", "recent_listings"]
LABELS = {"log_qib": "QIB (institutions) demand", "log_nii": "HNI (NII) demand",
          "log_rii": "retail demand", "log_size_cr": "issue size",
          "nifty_20d": "Nifty's last 20 days", "recent_listings": "recent IPO listings"}
WIN = 0.01                       # listing close must beat the issue price by 1% (costs)
QIB_BUCKETS = [0, 3, 10, 50, 100, np.inf]
APPLY_AT, AVOID_AT = 0.65, 0.45


def dataset(issues: pd.DataFrame, subs: pd.DataFrame, bhav: pd.DataFrame,
            indices: pd.DataFrame) -> pd.DataFrame:
    """One row per mainboard IPO with inputs and listing-day result (if listed)."""
    d = issues[(issues["segment"] == "mainboard") & issues["issue_price"].gt(0)] \
        .merge(subs, on="symbol", how="inner")
    d = d[d["total_x"].gt(0)].copy()
    d = add_inputs(d, indices)
    lst = bhav.merge(d[["symbol", "listing_date"]], left_on=["symbol", "date"],
                     right_on=["symbol", "listing_date"])[["symbol", "open", "close"]]
    d = d.merge(lst.drop_duplicates("symbol"), on="symbol", how="left")
    d["gain_open"] = d["open"] / d["issue_price"] - 1
    d["gain_close"] = d["close"] / d["issue_price"] - 1
    d = d.sort_values("listing_date").reset_index(drop=True)
    # recent_listings: median listing gain of the 10 previous IPOs listed before bidding closed
    done = d.dropna(subset=["gain_close"])
    d["recent_listings"] = [done.loc[done["listing_date"] < e, "gain_close"].tail(10).median()
                            for e in d["end"]]
    return d


def add_inputs(d: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for k in ("qib", "nii", "rii"):
        d[f"log_{k}"] = np.log1p(d[f"{k}_x"].astype(float).clip(lower=0))
    size = d["shares_offered"].astype(float) * d["issue_price"].fillna(d["price_high"]).astype(float) / 1e7
    d["log_size_cr"] = np.log1p(size.where(size > 0))
    n = indices[indices["symbol"] == "NIFTY50"].set_index("date")["close"].sort_index()
    ret20 = n / n.shift(20) - 1
    d["nifty_20d"] = [ret20[:e].iloc[-1] if pd.notna(e) and len(ret20[:e]) else np.nan
                      for e in d["end"]]
    return d


def _fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Logistic regression by Newton steps; X has a leading column of ones."""
    w = np.zeros(X.shape[1])
    reg = l2 * np.eye(X.shape[1])
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + reg @ w
        h = (X * (p * (1 - p))[:, None]).T @ X + reg
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


class IPOModel:
    def __init__(self, mean, std, w, info: dict):
        self.mean, self.std, self.w, self.info = np.array(mean), np.array(std), np.array(w), info

    @classmethod
    def train(cls, d: pd.DataFrame, info: dict | None = None) -> "IPOModel":
        t = d.dropna(subset=["gain_close"])
        X = t[FEATURES].astype(float)
        mean, std = X.mean(), X.std().replace(0, 1)
        Z = ((X.fillna(mean) - mean) / std).to_numpy()
        w = _fit(np.c_[np.ones(len(Z)), Z], (t["gain_close"] > WIN).to_numpy(float))
        return cls(mean.to_numpy(), std.to_numpy(), w, info or {})

    def contributions(self, row: pd.Series) -> dict[str, float]:
        x = np.array([float(row.get(f)) if pd.notna(row.get(f)) else m
                      for f, m in zip(FEATURES, self.mean)])
        return dict(zip(FEATURES, (x - self.mean) / self.std * self.w[1:]))

    def prob(self, row: pd.Series) -> float:
        return float(1 / (1 + np.exp(-(self.w[0] + sum(self.contributions(row).values())))))

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text(json.dumps(
            {"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
             "w": self.w.tolist(), "info": self.info}, indent=1, default=float))

    @classmethod
    def load(cls, folder: Path) -> "IPOModel | None":
        p = Path(folder) / "model.json"
        if not p.exists():
            return None
        j = json.loads(p.read_text())
        return cls(j["mean"], j["std"], j["w"], j.get("info", {}))


def walk_forward(d: pd.DataFrame, first_year: int = 2019) -> pd.DataFrame:
    """Each year's IPOs scored by a model trained only on IPOs listed in earlier years."""
    t = d.dropna(subset=["gain_close"])
    out = []
    for y in sorted(t["listing_date"].dt.year.unique()):
        if y < first_year:
            continue
        past, now = t[t["listing_date"].dt.year < y], t[t["listing_date"].dt.year == y]
        if len(past) < 40 or now.empty:
            continue
        m = IPOModel.train(past)
        out.append(now.assign(prob=[m.prob(r) for _, r in now.iterrows()]))
    return pd.concat(out) if out else pd.DataFrame()


def buckets(d: pd.DataFrame) -> list[dict]:
    t = d.dropna(subset=["gain_close"])
    b = pd.cut(t["qib_x"], QIB_BUCKETS, right=False)
    rows = []
    for k, g in t.groupby(b, observed=True):
        rows.append({"lo": float(k.left), "hi": float(k.right), "n": int(len(g)),
                     "listed_up": float((g["gain_close"] > WIN).mean()),
                     "median_gain": float(g["gain_close"].median())})
    return rows


def train_and_report(d: pd.DataFrame) -> tuple["IPOModel", dict]:
    wf = walk_forward(d)
    info = {"ipos": int(d["gain_close"].notna().sum()), "since": f"{d['listing_date'].min():%Y-%m-%d}",
            "base_rate": float((d["gain_close"].dropna() > WIN).mean()),
            "qib_buckets": buckets(d)}
    if not wf.empty:
        app, avoid = wf[wf["prob"] >= APPLY_AT], wf[wf["prob"] < AVOID_AT]
        info["walk_forward"] = {
            "ipos": int(len(wf)), "all_listed_up": float((wf["gain_close"] > WIN).mean()),
            "all_median_gain": float(wf["gain_close"].median()),
            "apply_n": int(len(app)),
            "apply_listed_up": float((app["gain_close"] > WIN).mean()) if len(app) else None,
            "apply_median_gain": float(app["gain_close"].median()) if len(app) else None,
            "avoid_n": int(len(avoid)),
            "avoid_listed_up": float((avoid["gain_close"] > WIN).mean()) if len(avoid) else None,
            "avoid_median_gain": float(avoid["gain_close"].median()) if len(avoid) else None}
    return IPOModel.train(d, info), info


def verdict(m: IPOModel, row: pd.Series, closes_today: bool = False,
            still_open: bool = True) -> tuple[str, float, list[str]]:
    """'Apply' / 'Avoid' / 'Neutral' (or 'Wait' while QIB demand isn't in yet) + reasons."""
    p = m.prob(row)
    con = m.contributions(row)
    reasons = []
    for f, c in sorted(con.items(), key=lambda kv: -abs(kv[1]))[:3]:
        if abs(c) < 0.1:
            continue
        reasons.append(f"{LABELS[f]} {'helps' if c > 0 else 'hurts'} ({_show(f, row)})")
    q = row.get("qib_x")
    for b in m.info.get("qib_buckets", []):
        if pd.notna(q) and b["lo"] <= q < b["hi"] and b["n"] >= 5:
            hi = "∞" if not np.isfinite(b["hi"]) else f"{b['hi']:g}"
            reasons.append(f"past IPOs with QIB {b['lo']:g}–{hi}×: {b['listed_up']:.0%} listed "
                           f"up, median {b['median_gain']:+.0%} ({b['n']} IPOs)")
    if still_open and not closes_today:
        return "Wait", p, ["QIBs bid mostly on the last day: decide on the closing day"] + reasons
    label = "Apply" if p >= APPLY_AT else "Avoid" if p < AVOID_AT else "Neutral"
    return label, p, reasons


def _show(f: str, row: pd.Series) -> str:
    if f in ("log_qib", "log_nii", "log_rii"):
        v = row.get(f[4:] + "_x")
        return f"{v:.1f}× subscribed" if pd.notna(v) else "no data"
    if f == "log_size_cr":
        return f"₹{np.expm1(row.get(f)):,.0f} cr" if pd.notna(row.get(f)) else "unknown"
    v = row.get(f)
    return f"{v:+.1%}" if pd.notna(v) else "no data"
