"""Learn which headlines actually matter for their stock.

Google News returns anything mentioning a company: results and orders, but also lists of
ten companies, namesakes abroad and click-bait. Instead of hand-written rules, a LightGBM
model learns from history how unusual the stock's move was on the first close after each
headline (move vs Nifty 50, in units of the stock's normal daily move). Its prediction,
scaled so an average headline = 1.0, is the headline's `relevance`: news that historically
came with real moves counts more in the news signals, noise counts less.

Inputs per headline: the words of the title, whether it names the company, how many other
companies it names, its FinBERT tone and whether it only reports a price move. The model is
checked on the most recent 20% of headlines, which it never trained on.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredictor.config import SHARED_MODELS_DIR
from stockpredictor.data import news as N

MODEL_DIR = SHARED_MODELS_DIR / "news"
MIN_HEADLINES = 1000          # fewer labelled headlines than this: no model (relevance = 1)
VOCAB_SIZE = 400
MIN_WORD_COUNT = 15
CLOSE_IST = pd.Timedelta(hours=15, minutes=30)
PARAMS = {"objective": "huber", "alpha": 1.5, "learning_rate": 0.05, "num_leaves": 15,
          "min_data_in_leaf": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 7}
ROUNDS = 300

_WORD = re.compile(r"[a-z][a-z&']{2,}")
STOP = set("""the and for with from that this into over after amid its are was were has have
had but not you your our out all new now how why what who will can may say says said about
than more most also just ltd limited india indian share shares stock stocks company""".split())
# Words too common in company names to identify one on their own.
GENERIC = set("""india indian bank of the and industries industry limited ltd corporation corp
company co finance financial services energy power tata adani bajaj hdfc icici life
insurance motors motor steel oil gas national state hindustan bharat""".split())


def _words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _name_words(name: str) -> list[str]:
    clean = N._SUFFIXES.sub("", str(name)).replace("&", " and ")
    return [w for w in _words(clean) if w not in {"and", "the", "of"}]


def _company_keys(universe: pd.DataFrame) -> dict[str, str]:
    """A short key per company for spotting it in a title ('Tata Power' -> 'tata power')."""
    keys = {}
    for sym, name in universe[["symbol", "name"]].itertuples(index=False):
        ws = _name_words(name or sym)
        if not ws:
            continue
        keys[sym] = " ".join(ws[:2]) if ws[0] in GENERIC and len(ws) > 1 else ws[0]
    return keys


def _meta(news: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    names = dict(zip(universe["symbol"], universe["name"]))
    keys = _company_keys(universe)
    rows = []
    for sym, title, sent in news[["symbol", "title", "sentiment"]].itertuples(index=False):
        low = " " + " ".join(_words(title)) + " "
        own = [w for w in _name_words(names.get(sym, sym)) if w not in GENERIC] \
            or _name_words(names.get(sym, sym))
        match = np.mean([f" {w} " in low for w in own]) if own else 0.0
        others = sum(f" {k} " in low for s, k in keys.items() if s != sym)
        rows.append({"name_match": match, "symbol_in_title": float(sym.lower() in low.split()),
                     "n_other_companies": others, "sentiment": sent,
                     "abs_sentiment": abs(sent) if pd.notna(sent) else np.nan,
                     "price_move": float(bool(N.PRICE_MOVE.search(str(title)))),
                     "title_words": len(low.split())})
    return pd.DataFrame(rows, index=news.index)


def _bag(news: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    index = {w: i for i, w in enumerate(vocab)}
    x = np.zeros((len(news), len(vocab)), dtype=np.float32)
    for r, title in enumerate(news["title"]):
        for w in set(_words(title)):
            i = index.get(w)
            if i is not None:
                x[r, i] = 1.0
    return pd.DataFrame(x, columns=[f"w_{w}" for w in vocab], index=news.index)


def features(news: pd.DataFrame, universe: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    return pd.concat([_meta(news, universe), _bag(news, vocab)], axis=1)


def reaction(news: pd.DataFrame, daily: pd.DataFrame, indices: pd.DataFrame) -> pd.Series:
    """Unusual move on the first close after each headline: |stock - Nifty| daily return,
    divided by the stock's typical daily |stock - Nifty| move over the previous 3 months."""
    px = daily.assign(px=daily["adj_close"].fillna(daily["close"])).sort_values(["symbol", "date"])
    px["ret"] = px.groupby("symbol")["px"].pct_change()
    nifty = indices[indices["symbol"] == "NIFTY50"].sort_values("date")
    nret = nifty.set_index("date")["close"].pct_change()
    px["abn"] = px["ret"] - px["date"].map(nret)
    px["scale"] = px.groupby("symbol")["abn"].transform(
        lambda x: x.abs().rolling(63, min_periods=20).mean().shift())
    px["z"] = (px["abn"].abs() / px["scale"]).clip(upper=6)

    ist = news["published"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    # Before 15:30 IST the same day's close reacts; later headlines react the next day.
    first = ist.dt.normalize() + np.where(ist - ist.dt.normalize() < CLOSE_IST,
                                          pd.Timedelta(0), pd.Timedelta(days=1))
    left = pd.DataFrame({"symbol": news["symbol"].values,
                         "first": pd.to_datetime(first.values).astype("datetime64[ns]"),
                         "_i": news.index}).sort_values("first")
    right = px[["symbol", "date", "z"]].dropna()
    right = right.assign(date=right["date"].astype("datetime64[ns]")).sort_values("date")
    m = pd.merge_asof(left, right, left_on="first", right_on="date", by="symbol",
                      direction="forward", tolerance=pd.Timedelta(days=5))
    return m.set_index("_i")["z"].reindex(news.index)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 10 else float("nan")


def train(news: pd.DataFrame, universe: pd.DataFrame, daily: pd.DataFrame,
          indices: pd.DataFrame) -> dict | None:
    """Fit on all labelled headlines; report accuracy on the newest 20% (trained without them)."""
    import lightgbm as lgb

    news = news.sort_values("published").reset_index(drop=True)
    y = reaction(news, daily, indices)
    lab = news[y.notna()].copy()
    lab["y"] = y[y.notna()]
    if len(lab) < MIN_HEADLINES:
        return None
    counts = pd.Series([w for t in lab["title"] for w in set(_words(t)) if w not in STOP])
    counts = counts.value_counts()
    vocab = sorted(counts[counts >= MIN_WORD_COUNT].head(VOCAB_SIZE).index)
    x = features(lab, universe, vocab)

    cut = int(len(lab) * 0.8)
    fit = lgb.train(PARAMS, lgb.Dataset(x.iloc[:cut], lab["y"].iloc[:cut]), ROUNDS)
    test_pred = pd.Series(fit.predict(x.iloc[cut:]), index=lab.index[cut:])
    test_y = lab["y"].iloc[cut:]
    report = {"headlines": len(lab), "test_headlines": len(lab) - cut,
              "test_corr": _spearman(test_pred, test_y),
              "baseline_sentiment_corr": _spearman(x["abs_sentiment"].iloc[cut:], test_y),
              "baseline_name_match_corr": _spearman(x["name_match"].iloc[cut:], test_y)}
    top = test_pred >= test_pred.quantile(0.8)
    report["test_move_top20"] = float(test_y[top].mean())
    report["test_move_rest"] = float(test_y[~top].mean())

    model = lgb.train(PARAMS, lgb.Dataset(x, lab["y"]), ROUNDS)
    scale = float(np.mean(model.predict(x)))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "model.txt"))
    imp = pd.Series(model.feature_importance("gain"), index=x.columns).sort_values(ascending=False)
    meta = {"vocab": vocab, "scale": scale, "trained_at": datetime.now().isoformat(timespec="seconds"),
            **report, "top_inputs": [c.removeprefix("w_") for c in imp.head(15).index]}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def add_relevance(news: pd.DataFrame, universe: pd.DataFrame,
                  model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    """news + a `relevance` column (1.0 for everything when no model is trained yet)."""
    news = news.copy()
    news["relevance"] = 1.0
    if news.empty or not (model_dir / "model.txt").exists():
        return news
    try:
        import lightgbm as lgb

        meta = json.loads((model_dir / "meta.json").read_text())
        model = lgb.Booster(model_file=str(model_dir / "model.txt"))
        pred = model.predict(features(news, universe, meta["vocab"]))
        news["relevance"] = np.clip(pred / meta["scale"], 0.05, 5.0)
    except Exception:
        pass    # a broken model file must not stop the app; everything counts equally
    return news


def model_info(model_dir: Path = MODEL_DIR) -> dict | None:
    path = model_dir / "meta.json"
    return json.loads(path.read_text()) if path.exists() else None
