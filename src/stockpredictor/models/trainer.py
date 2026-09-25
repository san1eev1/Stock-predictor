"""Continuous training: daily retraining on the newest data, weekly self-tuning.

Tuning tries variations of the current model settings and scores each one
out-of-sample (walk-forward: every test period is predicted by a model trained
only on earlier data). A variant is adopted only if it beats the current
settings by a margin, which limits chasing noise. Every run is logged to the
model_runs table so the app can show whether prediction quality improves.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from stockpredictor.features import longterm as F
from stockpredictor.models import intraday as MI
from stockpredictor.models import longterm as M

GRID = {
    "num_leaves": [15, 31, 63],
    "min_data_in_leaf": [50, 100, 200, 400],
    "learning_rate": [0.02, 0.03, 0.05],
    "feature_fraction": [0.6, 0.7, 0.8],
    "lambda_l2": [1.0, 5.0, 10.0],
    "num_rounds": [200, 300, 400, 600],
}
LONGTERM_GRID = {**GRID, "mom_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
                 "recency_half_life": [0.0, 3.0, 5.0, 10.0], "tail_weight": [0.0, 1.0, 2.0],
                 "rank_objective": [False, True], "top_features": [0, 25, 35],
                 # which chart-method groups (features/technical.py) the model may use
                 "chart_groups": [["trend"], ["trend", "volume"], ["trend", "oscillators"],
                                  ["trend", "candles"], ["trend", "statistics"],
                                  ["trend", "candles", "oscillators", "volume", "statistics"]]}
FEEDBACK_WEIGHT = {1: 1.5, 0: 2.0}   # paper predictions: right / wrong
# IC gain needed to switch settings. Re-running the same settings with another random
# seed moves IC by about +/-0.004, so smaller "gains" are noise.
MARGIN = 0.01


def log_run(conn: sqlite3.Connection | None, horizon: str, kind: str, train_to: str,
            metrics: dict) -> None:
    if conn is None:
        return
    conn.execute("INSERT INTO model_runs (horizon, version, train_from, train_to, metrics) "
                 "VALUES (?, ?, ?, ?, ?)",
                 (horizon, datetime.now().isoformat(timespec="seconds"), kind, train_to,
                  json.dumps(metrics, default=float)))
    conn.commit()


def candidates(current: dict, n: int, seed: int | None = None,
               grid: dict | None = None) -> list[dict]:
    """The current settings plus `n` variations that each change one or two values."""
    grid = grid or GRID
    rng = random.Random(seed)
    out, seen = [dict(current)], {json.dumps(current, sort_keys=True)}
    for _ in range(n * 10):
        if len(out) > n:
            break
        c = dict(current)
        for key in rng.sample(list(grid), rng.choice([1, 2])):
            c[key] = rng.choice(grid[key])
        k = json.dumps(c, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _save_params(model_dir, params: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "params.json").write_text(json.dumps(params, indent=2))


# --- Long-term ---------------------------------------------------------------------

def longterm_labeled(ctx) -> pd.DataFrame:
    return M.add_labels(M.training_snapshots(ctx.feats), ctx.daily, ctx.indices)


def paper_feedback(ctx, conn, labeled: pd.DataFrame) -> pd.DataFrame:
    """Add the judged paper-trading predictions to the training data with extra weight
    (more for the ones it got wrong), so the model concentrates on its own decisions."""
    if conn is None:
        return labeled
    judged = pd.read_sql("SELECT symbol, date, correct FROM predictions WHERE horizon = 'longterm' "
                         "AND correct IS NOT NULL AND horizon_days = ?", conn, params=(M.HORIZON,))
    if judged.empty:
        return labeled
    judged["date"] = pd.to_datetime(judged["date"])
    rows = ctx.feats.merge(judged, on=["symbol", "date"])
    if rows.empty:
        return labeled
    rows = M.add_labels(rows.drop(columns="correct"), ctx.daily, ctx.indices) \
        .merge(judged, on=["symbol", "date"])
    rows["fb_weight"] = rows["correct"].map(FEEDBACK_WEIGHT)
    key = set(zip(rows["symbol"], rows["date"]))
    rest = labeled[[k not in key for k in zip(labeled["symbol"], labeled["date"])]]
    return pd.concat([rest, rows.drop(columns="correct")], ignore_index=True)


def evaluate_longterm(labeled: pd.DataFrame, params: dict, years: int = 3) -> dict:
    last = labeled.dropna(subset=["target"])["date"].max().year
    scores = M.walk_forward(labeled, labeled, last - years + 1, last, params=params)
    if scores.empty:
        return {"ic": float("nan"), "ic_positive": float("nan"), "top10_hit": float("nan"),
                "top10_excess": float("nan"), "weeks": 0}
    ic = M.information_coefficient(scores, labeled)
    m = scores.merge(labeled[["symbol", "date", "fwd_excess"]], on=["symbol", "date"]).dropna()
    top = m[m.groupby("date")["score"].rank(ascending=False) <= 10]
    return {"ic": float(ic.mean()), "ic_positive": float((ic > 0).mean()),
            "top10_hit": float((top["fwd_excess"] > 0).mean()),
            "top10_excess": float(top["fwd_excess"].mean()), "weeks": int(len(ic))}


def retrain_longterm(ctx, conn=None, force: bool = False) -> M.LongTermModel | None:
    """Daily retrain on all history + judged paper predictions (once per day)."""
    if not force and (M.MODEL_DIR / "meta.json").exists():
        meta = json.loads((M.MODEL_DIR / "meta.json").read_text())
        if meta["trained_at"][:10] == datetime.now().strftime("%Y-%m-%d") \
                and meta.get("horizon", 63) == M.HORIZON:
            return None
    labeled = paper_feedback(ctx, conn, longterm_labeled(ctx))
    model = M.LongTermModel.train(labeled)
    model.save(M.MODEL_DIR)
    fb = int(labeled["fb_weight"].notna().sum()) if "fb_weight" in labeled else 0
    log_run(conn, "longterm", "retrain", model.train_to,
            {"rows": int(labeled["target"].notna().sum()), "feedback_rows": fb,
             "params": M.current_params()})
    return model


def tune_longterm(ctx, conn=None, n_candidates: int = 5, years: int = 3,
                  seed: int | None = None, check_years: int = 6) -> dict:
    """Try variations of the current settings on history (walk-forward). A winner must beat
    the current settings over the last `years` AND not be worse over `check_years`, so a
    setting that only fits one period by luck is not adopted."""
    labeled = longterm_labeled(ctx)
    current = M.current_params()
    results = []
    for i, params in enumerate(candidates(current, n_candidates, seed, LONGTERM_GRID)):
        res = evaluate_longterm(labeled, params, years)
        results.append({"params": params, **res, "current": i == 0})
    valid = [r for r in results if not np.isnan(r["ic"])] or results
    best = max(valid, key=lambda r: -np.inf if np.isnan(r["ic"]) else r["ic"])
    base = results[0]
    adopted = (not best["current"] and not np.isnan(best["ic"])
               and (np.isnan(base["ic"]) or best["ic"] >= base["ic"] + MARGIN))
    check = None
    if adopted and check_years > years:
        check = {"best": evaluate_longterm(labeled, best["params"], check_years)["ic"],
                 "current": evaluate_longterm(labeled, base["params"], check_years)["ic"]}
        adopted = not np.isnan(check["best"]) and (np.isnan(check["current"])
                                                   or check["best"] >= check["current"])
    if adopted:
        _save_params(M.MODEL_DIR, best["params"])
    chosen = best if adopted else base
    report = {"adopted": adopted, "ic": chosen["ic"], "top10_hit": chosen["top10_hit"],
              "top10_excess": chosen["top10_excess"], "previous_ic": base["ic"],
              "tested": len(results), "params": chosen["params"], "all": results,
              "long_check": check}
    log_run(conn, "longterm", "tune", f"{labeled['date'].max():%Y-%m-%d}", report)
    if adopted:
        retrain_longterm(ctx, conn, force=True)
    return report


LIVE_MIN_DAYS = 20       # judged prediction days needed before switching on live results
LIVE_MIN_GAIN = 0.05     # accuracy lead (5 points) a variant needs to take over


def live_selection(conn) -> dict | None:
    """Switch the long-term strategy blend to the variant that wins the live paper race."""
    from stockpredictor.paper import engine as E

    race = E.strategy_race(conn, horizon_days=M.HORIZON)
    if race.empty or race["days"].min() < LIVE_MIN_DAYS:
        return None
    params = M.current_params()
    current_w = params.get("mom_weight", 0.0)
    weights = E.SHADOW_VARIANTS
    current = min(weights, key=lambda k: abs(weights[k] - current_w))
    acc = race.set_index("variant")["accuracy"]
    best = acc.idxmax()
    switched = best != current and acc[best] >= acc.get(current, 0) + LIVE_MIN_GAIN
    report = {"race": race.to_dict("records"), "current": current, "best": best,
              "switched": bool(switched)}
    if switched:
        _save_params(M.MODEL_DIR, {**params, "mom_weight": weights[best]})
        log_run(conn, "longterm", "live-switch", datetime.now().strftime("%Y-%m-%d"),
                {**report, "ic": None, "adopted": True,
                 "params": {"mom_weight": weights[best]}})
    return report


# --- Intraday -----------------------------------------------------------------------

def intraday_feats(ctx, store_dir) -> pd.DataFrame:
    from stockpredictor import store
    from stockpredictor.data import intraday as I
    from stockpredictor.features import intraday as FI

    return FI.build(I.load_summaries(store_dir), ctx.daily, ctx.feats, store.load_actions(store_dir))


def evaluate_intraday(feats: pd.DataFrame, params: dict, last_days: int = 120) -> dict:
    scores = MI.walk_forward(feats, params=params, last_days=last_days)
    if scores.empty:
        return {"ic": float("nan"), "days": 0}
    m = scores.merge(feats[["symbol", "date", "target_ret"]], on=["symbol", "date"])
    ic = m.groupby("date").apply(lambda g: g["score"].rank().corr(g["target_ret"].rank()),
                                 include_groups=False).dropna()
    r = m.groupby("date")["score"].rank(ascending=False)
    n = m.groupby("date")["score"].transform("size")
    longs, shorts = m[r <= 5], m[r > n - 5]
    acc = pd.concat([longs["target_ret"] > 0, shorts["target_ret"] < 0]).mean()
    return {"ic": float(ic.mean()), "ic_positive": float((ic > 0).mean()),
            "direction_accuracy": float(acc), "days": int(len(ic))}


def retrain_intraday(ctx, store_dir, conn=None, force: bool = False) -> MI.IntradayModel | None:
    """Daily retrain on all intraday history (Yahoo + Angel One), once per day."""
    if not force and (MI.MODEL_DIR / "meta.json").exists():
        meta = json.loads((MI.MODEL_DIR / "meta.json").read_text())
        if meta["trained_at"][:10] == datetime.now().strftime("%Y-%m-%d"):
            return None
    feats = intraday_feats(ctx, store_dir)
    if feats.empty or feats["date"].nunique() < MI.MIN_TRAIN_DAYS:
        return None
    model = MI.IntradayModel.train(feats)
    model.save()
    log_run(conn, "intraday", "retrain", model.train_to,
            {"days": model.train_days, "params": MI.current_params()})
    return model


def tune_intraday(ctx, store_dir, conn=None, n_candidates: int = 5,
                  seed: int | None = None, check_days: int = 250) -> dict | None:
    """Like tune_longterm: must win on the last 120 days and not lose on the last `check_days`."""
    feats = intraday_feats(ctx, store_dir)
    if feats.empty or feats["date"].nunique() < MI.MIN_TRAIN_DAYS + 10:
        return None
    current = MI.current_params()
    results = []
    for i, params in enumerate(candidates(current, n_candidates, seed)):
        results.append({"params": params, **evaluate_intraday(feats, params), "current": i == 0})
    valid = [r for r in results if not np.isnan(r["ic"])]
    if not valid:
        return None
    base = results[0]
    best = max(valid, key=lambda r: r["ic"])
    adopted = not best["current"] and best["ic"] >= base["ic"] + MARGIN
    check = None
    if adopted and feats["date"].nunique() >= check_days + MI.MIN_TRAIN_DAYS:
        check = {"best": evaluate_intraday(feats, best["params"], check_days)["ic"],
                 "current": evaluate_intraday(feats, base["params"], check_days)["ic"]}
        adopted = not np.isnan(check["best"]) and (np.isnan(check["current"])
                                                   or check["best"] >= check["current"])
    if adopted:
        _save_params(MI.MODEL_DIR, best["params"])
    chosen = best if adopted else base
    report = {"adopted": adopted, "ic": chosen["ic"],
              "direction_accuracy": chosen.get("direction_accuracy"),
              "previous_ic": base["ic"], "tested": len(results), "params": chosen["params"],
              "all": results, "long_check": check}
    log_run(conn, "intraday", "tune", f"{feats['date'].max():%Y-%m-%d}", report)
    if adopted:
        MI.IntradayModel.train(feats).save()
    return report
