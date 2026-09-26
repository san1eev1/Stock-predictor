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
from pathlib import Path

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
    "linear_blend": [0.0, 0.2, 0.4],     # mix in a ridge model (a different kind of model)
}
LONGTERM_GRID = {**GRID, "mom_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
                 "recency_half_life": [0.0, 3.0, 5.0, 10.0], "tail_weight": [0.0, 1.0, 2.0],
                 "rank_objective": [False, True], "top_features": [0, 25, 35],
                 # which chart-method groups (features/technical.py) the model may use
                 "chart_groups": [["trend"], ["trend", "volume"], ["trend", "oscillators"],
                                  ["trend", "candles"], ["trend", "statistics"],
                                  ["trend", "candles", "oscillators", "volume", "statistics"]]}
FEEDBACK_WEIGHT = {1: 1.5, 0: 2.0}   # paper predictions: right / wrong
MIN_STOCKS_PER_DAY = 50      # intraday days with fewer known outcomes are not used
# Intraday: also try focusing on the biggest risers and fallers (both ends = buy and sell picks).
# Learning from each other: blend in the other intraday model's ranking (peer_weight).
INTRADAY_GRID = {**GRID, "tail_weight": [0.0, 1.0, 2.0], "peer_weight": [0.0, 0.25, 0.5],
                 "label": ["move", "trade"],      # trade: learn the stop/target trade outcome
                 "recency_half_life_days": [0, 120, 250, 500],
                 "rank_objective": [False, True]}
# IC gain needed to switch settings. Re-running the same settings with another random
# seed moves IC by about +/-0.004, so smaller "gains" are noise.
MARGIN = 0.01


# Set by cloud training: runs are also appended to this JSON-lines file (published with the
# models), so the dashboard on the Mac can show the cloud's training history.
RUN_LOG: Path | None = None
# Set by cloud training: judged paper predictions from the paper-feedback branch.
FEEDBACK: pd.DataFrame | None = None
INTRADAY_FEEDBACK: pd.DataFrame | None = None   # judged intraday picks (cloud: feedback branch)


def log_run(conn: sqlite3.Connection | None, horizon: str, kind: str, train_to: str,
            metrics: dict) -> None:
    if RUN_LOG is not None:
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps({"horizon": horizon, "version": datetime.now().isoformat(
                timespec="seconds"), "kind": kind, "train_to": train_to,
                "metrics": metrics}, default=float) + "\n")
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
        # mostly small steps; sometimes a bigger jump (3-4 settings) to explore further
        for key in rng.sample(list(grid), min(len(grid), rng.choice([1, 1, 2, 2, 3, 4]))):
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
    if FEEDBACK is not None:
        judged = FEEDBACK.copy()
    elif conn is None:
        return labeled
    else:
        judged = judged_predictions(conn)
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


def judged_predictions(conn) -> pd.DataFrame:
    return pd.read_sql("SELECT symbol, date, correct FROM predictions WHERE horizon = 'longterm' "
                       "AND correct IS NOT NULL AND horizon_days = ?", conn, params=(M.HORIZON,))


def load_runs(conn, run_log: Path | None = None) -> pd.DataFrame:
    """Training history: local runs (database) plus cloud runs (runs.jsonl from GitHub)."""
    runs = pd.read_sql("SELECT horizon, version, train_from AS kind, train_to, metrics "
                       "FROM model_runs ORDER BY id", conn)
    path = run_log or (M.MODEL_DIR.parent / "runs.jsonl")
    if path.exists():
        cloud = pd.read_json(path, lines=True, dtype=False)
        if not cloud.empty:
            cloud["metrics"] = cloud["metrics"].map(lambda m: json.dumps(m, default=float))
            runs = pd.concat([runs, cloud[["horizon", "version", "kind", "train_to", "metrics"]]],
                             ignore_index=True).sort_values("version", kind="stable")
    return runs.reset_index(drop=True)


TOP_TRADED = 5          # long-term: the 5 best stocks are bought (backtest.portfolio.Rules)


def evaluate_longterm(labeled: pd.DataFrame, params: dict, years: int = 3) -> dict:
    last = labeled.dropna(subset=["target"])["date"].max().year
    scores = M.walk_forward(labeled, labeled, last - years + 1, last, params=params)
    if scores.empty:
        return {"ic": float("nan"), "ic_positive": float("nan"), "top10_hit": float("nan"),
                "top10_excess": float("nan"), "top5_hit": float("nan"),
                "top5_excess": float("nan"), "weeks": 0}
    ic = M.information_coefficient(scores, labeled)
    m = scores.merge(labeled[["symbol", "date", "fwd_excess"]], on=["symbol", "date"]).dropna()
    rank = m.groupby("date")["score"].rank(ascending=False)
    top, best = m[rank <= 10], m[rank <= TOP_TRADED]
    return {"ic": float(ic.mean()), "ic_positive": float((ic > 0).mean()),
            "top10_hit": float((top["fwd_excess"] > 0).mean()),
            "top10_excess": float(top["fwd_excess"].mean()),
            "top5_hit": float((best["fwd_excess"] > 0).mean()),
            "top5_excess": float(best["fwd_excess"].mean()), "weeks": int(len(ic))}


def retrain_longterm(ctx, conn=None, force: bool = False,
                     model_dir: Path | None = None) -> M.LongTermModel | None:
    """Daily retrain on all history + judged paper predictions (once per day)."""
    model_dir = model_dir or M.MODEL_DIR
    if not force and (model_dir / "meta.json").exists():
        meta = json.loads((model_dir / "meta.json").read_text())
        if meta["trained_at"][:10] == datetime.now().strftime("%Y-%m-%d") \
                and meta.get("horizon", 63) == M.HORIZON:
            return None
    labeled = paper_feedback(ctx, conn, longterm_labeled(ctx))
    model = M.LongTermModel.train(labeled)
    model.save(model_dir)
    fb = int(labeled["fb_weight"].notna().sum()) if "fb_weight" in labeled else 0
    log_run(conn, "longterm", "retrain", model.train_to,
            {"rows": int(labeled["target"].notna().sum()), "feedback_rows": fb,
             "params": M.current_params()})
    return model


def tune_longterm(ctx, conn=None, n_candidates: int = 5, years: int = 3,
                  seed: int | None = None, check_years: int = 6, log_all: bool = True,
                  cache: dict | None = None) -> dict:
    """Try variations of the current settings on history (walk-forward). A winner must beat
    the current settings over the last `years` AND not be worse over `check_years`, so a
    setting that only fits one period by luck is not adopted."""
    labeled = longterm_labeled(ctx)
    current = M.current_params()
    cache = {} if cache is None else cache      # same data + settings -> same score

    def score(params, yrs):
        key = (json.dumps(params, sort_keys=True, default=str), yrs)
        if key not in cache:
            cache[key] = evaluate_longterm(labeled, params, yrs)
        return cache[key]

    results = []
    for i, params in enumerate(candidates(current, n_candidates, seed, LONGTERM_GRID)):
        res = score(params, years)
        results.append({"params": params, **res, "current": i == 0})
    valid = [r for r in results if not np.isnan(r["ic"])] or results
    best = max(valid, key=lambda r: -np.inf if np.isnan(r["ic"]) else r["ic"])
    base = results[0]
    adopted = (not best["current"] and not np.isnan(best["ic"])
               and (np.isnan(base["ic"]) or best["ic"] >= base["ic"] + MARGIN)
               # paper check: the 5 stocks it would buy must beat Nifty at least as often
               and not best.get("top5_hit", np.nan) < base.get("top5_hit", np.nan))
    check = None
    if adopted and check_years > years:
        check = {"best": score(best["params"], check_years)["ic"],
                 "current": score(base["params"], check_years)["ic"]}
        adopted = not np.isnan(check["best"]) and (np.isnan(check["current"])
                                                   or check["best"] >= check["current"])
    if adopted:
        _save_params(M.MODEL_DIR, best["params"])
    chosen = best if adopted else base
    report = {"adopted": adopted, "ic": chosen["ic"], "top10_hit": chosen["top10_hit"],
              "top10_excess": chosen["top10_excess"], "top5_hit": chosen.get("top5_hit"),
              "top5_excess": chosen.get("top5_excess"), "previous_ic": base["ic"],
              "tested": len(results), "params": chosen["params"], "all": results,
              "long_check": check}
    if log_all or adopted:     # continuous rounds only log the ones that changed something
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

def intraday_feats(ctx, store_dir, target: MI.Target = MI.TRADE) -> pd.DataFrame:
    from stockpredictor import store
    from stockpredictor.data import intraday as I
    from stockpredictor.features import intraday as FI

    feats = FI.build_for(I.load_summaries(store_dir), ctx, store_dir, exit_col=target.exit_col)
    if feats.empty:
        return feats
    # A day only teaches ranking if enough stocks have its outcome (e.g. while the 12:30
    # prices are still being downloaded, most days have them for only a few stocks).
    n = feats.groupby("date")["target_ret"].transform("count")
    feats.loc[n < MIN_STOCKS_PER_DAY, ["target", "target_ret"]] = np.nan
    return feats


def evaluate_intraday(feats: pd.DataFrame, params: dict, last_days: int = 120,
                      peer_scores: pd.DataFrame | None = None) -> dict:
    scores = MI.walk_forward(feats, params=params, last_days=last_days)
    if scores.empty:
        return {"ic": float("nan"), "days": 0}
    scores = _with_peer(scores, peer_scores, params.get("peer_weight") or 0)
    m = scores.merge(feats[["symbol", "date", "target_ret"]], on=["symbol", "date"])
    ic = m.groupby("date").apply(lambda g: g["score"].rank().corr(g["target_ret"].rank()),
                                 include_groups=False).dropna()
    r = m.groupby("date")["score"].rank(ascending=False)
    n = m.groupby("date")["score"].transform("size")
    longs, shorts = m[r <= 5], m[r > n - 5]
    acc = pd.concat([longs["target_ret"] > 0, shorts["target_ret"] < 0]).mean()
    return {"ic": float(ic.mean()), "ic_positive": float((ic > 0).mean()),
            "direction_accuracy": float(acc), "days": int(len(ic))}


def _with_peer(scores: pd.DataFrame, peer_scores: pd.DataFrame | None,
               weight: float) -> pd.DataFrame:
    """Blend out-of-sample scores with the peer model's (both walk-forward) when weight > 0."""
    if not weight or peer_scores is None or peer_scores.empty:
        return scores
    m = scores.merge(peer_scores, on=["symbol", "date"], suffixes=("", "_peer"))
    if m.empty:
        return scores
    return m.assign(score=MI.blend(m["score"], m["score_peer"], m["date"], weight).values)[
        ["symbol", "date", "score"]]


def peer_walk_forward(ctx, store_dir, target: MI.Target, last_days: int) -> pd.DataFrame:
    """The other model's out-of-sample scores (its own settings, no blending)."""
    peer = MI.peer_of(target)
    feats = intraday_feats(ctx, store_dir, peer)
    params = {**MI.current_params(peer), "peer_weight": 0.0}
    return MI.walk_forward(feats, params=params, last_days=last_days)


def retrain_intraday(ctx, store_dir, conn=None, force: bool = False,
                     target: MI.Target = MI.TRADE) -> MI.IntradayModel | None:
    """Daily retrain on all intraday history (Yahoo + Angel One), once per day."""
    meta_path = target.model_dir / "meta.json"
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["trained_at"][:10] == datetime.now().strftime("%Y-%m-%d"):
            return None
    feats = intraday_feats(ctx, store_dir, target)
    if feats.empty or feats.dropna(subset=["target"])["date"].nunique() < MI.MIN_TRAIN_DAYS:
        return None
    feats = intraday_feedback(conn, feats, target.horizon)
    model = MI.IntradayModel.train(feats, target=target)
    model.save(target.model_dir)
    fb = int(feats["fb_weight"].notna().sum()) if "fb_weight" in feats else 0
    log_run(conn, target.horizon, "retrain", model.train_to,
            {"days": model.train_days, "feedback_rows": fb, "params": MI.current_params(target)})
    return model


COMPARE_PATH = MI.MODEL_DIR.parent / "intraday_compare.json"


RULES_FILE = "rules.json"      # trading rules chosen by profit, per intraday model
RULE_SIDES = [(5, 5), (3, 3), (2, 2), (1, 1), (5, 0), (3, 0), (2, 0), (1, 0),
              (0, 5), (0, 3), (0, 2), (0, 1), (3, 1), (1, 3)]
RULE_SKIPS = [0.0, 0.3, 0.5, 0.7]
RULE_EXITS = [(1.0, 2.0), (0.5, 1.0), (0.75, 1.5), (1.0, 0.0), (1.5, 3.0), (0.75, 0.0)]
RULE_PROBS = [0.0, 0.5, 0.55, 0.6]     # 'take this trade?' thresholds (models/take.py)
MIN_TRADE_DAYS = 0.2      # a rule set must trade on at least 1 day in 5 (not trading at all
                          # "earns" Rs 0 and would otherwise win against losing rules)


def rules_for(target: MI.Target, base=None):
    """`base` rules (the maximum trades) with this model's profit-tuned rules applied."""
    from stockpredictor.backtest import intraday as B

    base = base or B.IntradayRules()
    path = target.model_dir / RULES_FILE
    try:
        r = json.loads(path.read_text())["rules"] if path.exists() else None
    except (OSError, ValueError, KeyError):
        r = None
    if not r:
        return base
    return B.IntradayRules(n_long=min(base.n_long, int(r["n_long"])),
                           n_short=min(base.n_short, int(r["n_short"])),
                           stop_loss=float(r["stop_loss"]), target=float(r["target"]),
                           skip_quantile=float(r["skip_quantile"]),
                           min_prob=float(r.get("min_prob", 0.0)),
                           avoid_results=bool(r.get("avoid_results", False)),
                           avoid_expiry=bool(r.get("avoid_expiry", False)))


def day_profit(days: pd.DataFrame) -> float:
    """Average P&L per market day, skipped days counting as Rs 0 (fair to skipping)."""
    return float(days["pnl"].sum() / len(days)) if len(days) else float("nan")


HOLDOUT_DAYS = 60     # newest days never used for choosing rules, only to confirm them


def tune_rules(feats: pd.DataFrame, target: MI.Target, current=None, max_n: int = 5,
               capital: float = 100_000, days: int = 300) -> dict | None:
    """Choose the trading rules by profit after costs: how many buys and sells (0..max_n,
    long-only or short-only allowed), skipping weak days, stop-loss and target. Scores are
    walk-forward (each day predicted by a model trained only on earlier days). A rule set
    must earn the most on the recent half of the days AND not do worse than the current
    rules on the earlier half; otherwise the current rules stay. Saved to rules.json."""
    from dataclasses import replace

    from stockpredictor.backtest import intraday as B
    from stockpredictor.data import intraday as I

    from stockpredictor.models import take as K

    current = current or B.IntradayRules()
    scores = MI.walk_forward(feats, params=MI.current_params(target), last_days=days)
    if scores.empty or scores["date"].nunique() < 60:
        return None
    # 'Take this trade?' model: out-of-sample probabilities for every candidate (cross-fit
    # on the two halves) so thresholds can be judged fairly; the live one uses all days.
    try:                     # optional: without it the other rules are still tuned
        cand = K.candidates(scores, feats)
        y = K.outcomes(cand, B.IntradayRules(), target.exit_col,
                       K.exit_minutes(target.exit_col), capital)
        take_ok = bool(y.notna().sum() >= 400)
        if take_ok:
            scores = K.with_probs(scores, cand, K.cross_fit(cand, y))
    except (KeyError, ValueError, TypeError):
        take_ok = False
    dates = sorted(scores["date"].unique())
    # Newest HOLDOUT_DAYS: untouched - rules are chosen on the days before and must ALSO do
    # better here (rules that only fit the tuning days' noise fail this check).
    holdout = set(dates[-HOLDOUT_DAYS:]) if len(dates) > HOLDOUT_DAYS + 60 else set()
    tune_days = [x for x in dates if x not in holdout]
    recent = set(tune_days[len(tune_days) // 2:])
    minutes = I.EXIT_MINUTES if target == MI.TRADE else I.WINDOW_MINUTES
    summ = B.summary_columns(feats, target.exit_col)
    results: dict = {}

    def run(r) -> tuple[float, float, float, float]:
        """(recent, earlier) tuning-day profit per day, share of days traded, holdout profit."""
        if r not in results:
            sim = B.simulate(scores, summ, r, capital, exit_col=target.exit_col,
                             exit_minutes=minutes)
            d, t = sim["days"], sim["trades"]
            held = d["date"].isin(holdout)
            late = d["date"].isin(recent)
            earlier = ~late & ~held
            tuned = d[~held]
            traded = (t[~t["date"].isin(holdout)]["date"].nunique() / max(1, len(tuned))
                      if not t.empty else 0.0)
            results[r] = (day_profit(d[late]), day_profit(d[earlier]), traded,
                          day_profit(d[held]) if held.any() else float("nan"))
        return results[r]

    def best_of(options):
        ok = [r for r in options if run(r)[2] >= MIN_TRADE_DAYS] or options[:1]
        return max(ok, key=lambda r: run(r)[0])

    cap = lambda n: min(n, max_n)                              # noqa: E731
    now = replace(current, n_long=cap(current.n_long), n_short=cap(current.n_short))
    # Buys/sells per day together with skipping weak days (they interact), then stop/target,
    # then the 'take this trade?' threshold.
    best = best_of([now, *(replace(now, n_long=cap(a), n_short=cap(b), skip_quantile=q)
                           for a, b in RULE_SIDES for q in RULE_SKIPS)])
    best = best_of([best, *(replace(best, stop_loss=sl, target=tp) for sl, tp in RULE_EXITS)])
    if take_ok:
        best = best_of([best, *(replace(best, min_prob=p) for p in RULE_PROBS)])
    # Known traps: skip results-day stocks / monthly expiry days - only if that earns more.
    best = best_of([best, *(replace(best, avoid_results=a, avoid_expiry=e)
                            for a in (False, True) for e in (False, True))])
    (new_late, new_early, new_days, new_hold), (cur_late, cur_early, _, cur_hold) = \
        run(best), run(now)
    holdout_ok = not holdout or new_hold > cur_hold      # must win on the untouched days too
    adopted = (best != now and new_late > cur_late and new_early >= cur_early
               and new_days >= MIN_TRADE_DAYS and holdout_ok)
    chosen = best if adopted else now
    top = sorted((r for r in results if results[r][2] >= MIN_TRADE_DAYS),
                 key=lambda r: -results[r][0])[:5]
    report = {"target": target.horizon, "adopted": adopted,
              "rules": {"n_long": chosen.n_long, "n_short": chosen.n_short,
                        "stop_loss": chosen.stop_loss, "target": chosen.target,
                        "skip_quantile": chosen.skip_quantile, "min_prob": chosen.min_prob,
                        "avoid_results": chosen.avoid_results,
                        "avoid_expiry": chosen.avoid_expiry},
              "day_profit_recent": run(chosen)[0], "day_profit_before": run(chosen)[1],
              "current_day_profit_recent": cur_late, "tested": len(results),
              "holdout_days": len(holdout), "day_profit_holdout": run(chosen)[3],
              "current_day_profit_holdout": cur_hold,
              "trade_days": run(chosen)[2],
              "profitable": bool(run(chosen)[0] > 0 and run(chosen)[1] > 0),
              "top": [{"n_long": r.n_long, "n_short": r.n_short, "stop": r.stop_loss,
                       "target": r.target, "skip": r.skip_quantile, "min_prob": r.min_prob,
                       "recent": round(results[r][0]), "before": round(results[r][1]),
                       "holdout": None if results[r][3] != results[r][3] else round(results[r][3]),
                       "trade_days": round(results[r][2], 2)} for r in top],
              "period": f"{pd.Timestamp(dates[0]):%Y-%m-%d} to {pd.Timestamp(dates[-1]):%Y-%m-%d}",
              "updated": datetime.now().isoformat(timespec="seconds")}
    target.model_dir.mkdir(parents=True, exist_ok=True)
    if take_ok:                    # the live filter learns from all the days
        K.save(*K.fit(cand, y), target.model_dir)
    (target.model_dir / RULES_FILE).write_text(json.dumps(report, indent=1))
    return report


def compare_exits(ctx, store_dir, rules=None, last_days: int = 120,
                  capital: float = 100_000) -> dict | None:
    """Which exit is better, 12:30 or the close? Both models, walk-forward on the SAME days
    (each day predicted by a model trained only on earlier days), traded with the same rules
    and costs. Saved for the dashboard."""
    from stockpredictor.backtest import intraday as B
    from stockpredictor.data import intraday as I

    rules = rules or B.IntradayRules()
    feats = {t.horizon: intraday_feats(ctx, store_dir, t).dropna(subset=["target"])
             for t in MI.TARGETS}
    common = set.intersection(*(set(f["date"].unique()) for f in feats.values()))
    if len(common) < MI.MIN_TRAIN_DAYS + 20:
        return None
    out, raw = {}, {}
    for t in MI.TARGETS:
        f = feats[t.horizon][feats[t.horizon]["date"].isin(common)]
        raw[t.horizon] = MI.walk_forward(f, params={**MI.current_params(t), "peer_weight": 0.0},
                                         last_days=last_days)
        if raw[t.horizon].empty:
            return None
    for t in MI.TARGETS:
        f = feats[t.horizon][feats[t.horizon]["date"].isin(common)]
        scores = _with_peer(raw[t.horizon], raw[MI.peer_of(t).horizon],
                            MI.current_params(t).get("peer_weight") or 0)
        m = scores.merge(f[["symbol", "date", "target_ret"]], on=["symbol", "date"])
        ic = m.groupby("date").apply(lambda g: g["score"].rank().corr(g["target_ret"].rank()),
                                     include_groups=False).dropna()
        minutes = I.EXIT_MINUTES if t == MI.TRADE else I.WINDOW_MINUTES
        sim = B.simulate(scores, B.summary_columns(f, t.exit_col), rules,
                         capital, exit_col=t.exit_col, exit_minutes=minutes)
        r = B.metrics(sim, capital)
        out[t.horizon] = {
            "label": t.label, "days": r.get("trading_days", 0), "ic": float(ic.mean()),
            "avg_day_pnl": r.get("avg_day_pnl"), "win_days": r.get("win_days"),
            "return_pct": r.get("return_pct"), "direction_accuracy": r.get("direction_accuracy"),
            "random_baseline": r.get("random_baseline"), "trade_win_rate": r.get("trade_win_rate")}
    days = sorted(scores["date"].unique())
    out["period"] = f"{pd.Timestamp(days[0]):%Y-%m-%d} to {pd.Timestamp(days[-1]):%Y-%m-%d}"
    out["updated"] = datetime.now().isoformat(timespec="seconds")
    COMPARE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARE_PATH.write_text(json.dumps(out, indent=1, default=float))
    return out


REPLAY_ROUNDS = 3
REPLAY_DAYS = 120
PI_CANDIDATES = 10       # buy and sell candidates judged per day (paper.intraday.N_CANDIDATES)


def replay_file(target: MI.Target):
    return target.model_dir / "replay_feedback.csv"


def replay_round(feats: pd.DataFrame, target: MI.Target, rules, capital: float,
                 last_days: int = REPLAY_DAYS,
                 params: dict | None = None) -> tuple[dict, pd.DataFrame] | None:
    """One historical paper-trading batch: every day of the last `last_days` is traded by a
    model trained only on earlier days (using the fb_weight already in `feats`), then judged.
    Returns the result and the judged picks (symbol, date, correct) for the next round."""
    from stockpredictor.backtest import intraday as B
    from stockpredictor.data import intraday as I

    scores = MI.walk_forward(feats, params=params or MI.current_params(target),
                             last_days=last_days)
    if scores.empty:
        return None
    m = scores.merge(feats[["symbol", "date", "target_ret"]], on=["symbol", "date"])
    ic = m.groupby("date").apply(lambda g: g["score"].rank().corr(g["target_ret"].rank()),
                                 include_groups=False).dropna()
    minutes = I.EXIT_MINUTES if target == MI.TRADE else I.WINDOW_MINUTES
    summ = B.summary_columns(feats, target.exit_col)
    sim = B.simulate(scores, summ, rules, capital, exit_col=target.exit_col, exit_minutes=minutes)
    r = B.metrics(sim, capital)
    if not r.get("trading_days"):
        return None
    days = sorted(scores["date"].unique())
    result = {"period": f"{pd.Timestamp(days[0]):%Y-%m-%d} to {pd.Timestamp(days[-1]):%Y-%m-%d}",
              "days": r["trading_days"], "avg_day_pnl": r["avg_day_pnl"],
              "win_days": r["win_days"], "accuracy": r["direction_accuracy"],
              "random": r["random_baseline"], "ic": float(ic.mean())}
    # P&L comes from the traded picks (rules.n_long / n_short); the model learns from all
    # 10 buy + 10 sell candidates of each day, like the live paper trading.
    from dataclasses import replace

    n = PI_CANDIDATES
    wide = sim if (rules.n_long, rules.n_short) == (n, n) else B.simulate(
        scores, summ, replace(rules, n_long=n, n_short=n, skip_quantile=0.0), capital,
        exit_col=target.exit_col, exit_minutes=minutes)
    judged = wide["trades"][["symbol", "date", "correct"]].astype({"correct": int})
    return result, judged


def replay_training(ctx, store_dir, conn=None, rules=None, capital: float = 100_000,
                    rounds: int = REPLAY_ROUNDS) -> dict:
    """After the close: `rounds` historical paper-trading batches per intraday model, each
    learning from the previous batch's judged picks (wrong 2x, right 1.5x). If the last round
    beats the first (P&L per day and accuracy), the replay feedback is kept for the live
    model's training; otherwise it is dropped."""
    from stockpredictor.backtest import intraday as B

    rules = rules or B.IntradayRules()
    run_at = datetime.now().isoformat(timespec="seconds")
    out = {}
    for target in MI.TARGETS:
        base = intraday_feats(ctx, store_dir, target)
        if base.dropna(subset=["target"])["date"].nunique() < MI.MIN_TRAIN_DAYS + 20:
            out[target.horizon] = None
            continue
        feedback, results = None, []
        for rnd in range(1, rounds + 1):
            feats = base
            if feedback is not None:
                w = feedback.assign(fb_weight=feedback["correct"].map(FEEDBACK_WEIGHT))
                feats = base.merge(w[["symbol", "date", "fb_weight"]], on=["symbol", "date"],
                                   how="left")
            r = replay_round(feats, target, rules_for(target, rules), capital)
            if r is None:
                break
            res, feedback = r
            results.append({"round": rnd, **res})
        if not results:
            out[target.horizon] = None
            continue
        first, last = results[0], results[-1]
        adopted = len(results) > 1 and last["avg_day_pnl"] > first["avg_day_pnl"] \
            and last["accuracy"] >= first["accuracy"]
        path = replay_file(target)
        if adopted:
            path.parent.mkdir(parents=True, exist_ok=True)
            feedback.assign(date=pd.to_datetime(feedback["date"]).dt.strftime("%Y-%m-%d")) \
                .to_csv(path, index=False)
        else:
            path.unlink(missing_ok=True)
        if conn is not None:
            conn.executemany(
                "INSERT INTO replay_runs (run_at, horizon, round, period, days, avg_day_pnl, "
                "win_days, accuracy, random, ic, adopted) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(run_at, target.horizon, r["round"], r["period"], r["days"], r["avg_day_pnl"],
                  r["win_days"], r["accuracy"], r["random"], r["ic"],
                  int(adopted and r["round"] == len(results))) for r in results])
            conn.commit()
        out[target.horizon] = {"rounds": results, "adopted": adopted}
    return out


def intraday_feedback(conn, feats: pd.DataFrame, horizon: str = "intraday") -> pd.DataFrame:
    """Judged intraday paper picks, buy AND sell, get extra weight in training (wrong ones
    more), so the model concentrates on the calls it actually makes."""
    parts = []
    target = next((t for t in MI.TARGETS if t.horizon == horizon), None)
    if target is not None and replay_file(target).exists():       # kept replay feedback
        parts.append(pd.read_csv(replay_file(target)))
    if INTRADAY_FEEDBACK is not None:                             # cloud: sent by the Mac
        f = INTRADAY_FEEDBACK
        parts.append(f.loc[f["horizon"] == horizon, ["symbol", "date", "correct"]])
    if conn is not None:                                          # real paper picks win
        parts.append(pd.read_sql("SELECT symbol, date, correct FROM predictions WHERE "
                                 "horizon = ? AND correct IS NOT NULL", conn, params=(horizon,)))
    if not parts:
        return feats
    judged = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    if judged.empty:
        return feats
    judged["date"] = pd.to_datetime(judged["date"])
    judged = judged.groupby(["symbol", "date"], as_index=False)["correct"].min()
    out = feats.merge(judged, on=["symbol", "date"], how="left")
    out["fb_weight"] = out["correct"].map(FEEDBACK_WEIGHT)
    return out.drop(columns="correct")


def paper_check(feats, target: MI.Target, params: dict, rules=None,
                capital: float = 100_000) -> dict | None:
    """Paper trading on the last REPLAY_DAYS days with `params` (live rules: 3 buys + 3 sells,
    Rs 1 lakh a day, costs), every day predicted by a model trained only on earlier days."""
    from stockpredictor.backtest import intraday as B

    rules = rules or B.IntradayRules()
    key = (target.horizon, json.dumps(params, sort_keys=True, default=str), repr(rules),
           str(feats["date"].max()), len(feats))
    if key not in _PAPER_CACHE:     # same settings on the same data: same result, skip
        r = replay_round(feats, target, rules, capital, params=params)
        _PAPER_CACHE[key] = r[0] if r else None
    return _PAPER_CACHE[key]


_PAPER_CACHE: dict = {}


def tune_intraday(ctx, store_dir, conn=None, n_candidates: int = 5,
                  seed: int | None = None, check_days: int = 250,
                  log_all: bool = True, target: MI.Target = MI.TRADE,
                  rules=None) -> dict | None:
    """Like tune_longterm: must win on the last 120 days and not lose on the last `check_days`.
    Then the paper-trading check: the new settings must also paper-trade at least as well
    (P&L per day and share of picks right) as the current ones. Every round's paper result
    is saved in tune_checks."""
    feats = intraday_feats(ctx, store_dir, target)
    if feats.empty or feats["date"].nunique() < MI.MIN_TRAIN_DAYS + 10:
        return None
    current = MI.current_params(target)
    peer = peer_walk_forward(ctx, store_dir, target, check_days)
    results = []
    for i, params in enumerate(candidates(current, n_candidates, seed, INTRADAY_GRID)):
        results.append({"params": params, **evaluate_intraday(feats, params, peer_scores=peer), "current": i == 0})
    valid = [r for r in results if not np.isnan(r["ic"])]
    if not valid:
        return None
    base = results[0]
    best = max(valid, key=lambda r: r["ic"])
    adopted = not best["current"] and best["ic"] >= base["ic"] + MARGIN
    check = None
    if adopted and feats["date"].nunique() >= check_days + MI.MIN_TRAIN_DAYS:
        check = {"best": evaluate_intraday(feats, best["params"], check_days, peer)["ic"],
                 "current": evaluate_intraday(feats, base["params"], check_days, peer)["ic"]}
        adopted = not np.isnan(check["best"]) and (np.isnan(check["current"])
                                                   or check["best"] >= check["current"])
    rules = rules_for(target, rules)
    paper = {"current": paper_check(feats, target, base["params"], rules)}
    if adopted:
        paper["new"] = paper_check(feats, target, best["params"], rules)
        new, cur = paper["new"], paper["current"]
        adopted = new is not None and (cur is None or (
            new["avg_day_pnl"] >= cur["avg_day_pnl"] and new["accuracy"] >= cur["accuracy"]))
    if adopted:
        _save_params(target.model_dir, best["params"])
    chosen = best if adopted else base
    report = {"adopted": adopted, "ic": chosen["ic"],
              "direction_accuracy": chosen.get("direction_accuracy"),
              "previous_ic": base["ic"], "tested": len(results), "params": chosen["params"],
              "all": results, "long_check": check,
              "paper": paper.get("new") if adopted else paper["current"],
              "paper_before": paper["current"]}
    if conn is not None and report["paper"]:
        p = report["paper"]
        conn.execute("INSERT INTO tune_checks (run_at, horizon, period, days, avg_day_pnl, "
                     "win_days, accuracy, random, ic, adopted) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (datetime.now().isoformat(timespec="seconds"), target.horizon, p["period"],
                      p["days"], p["avg_day_pnl"], p["win_days"], p["accuracy"], p["random"],
                      p["ic"], int(adopted)))
        conn.commit()
    if log_all or adopted:
        log_run(conn, target.horizon, "tune", f"{feats['date'].max():%Y-%m-%d}", report)
    if adopted:
        MI.IntradayModel.train(feats, target=target).save(target.model_dir)
    return report
