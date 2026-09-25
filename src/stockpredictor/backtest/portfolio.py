"""Long-term portfolio rules and simulator.

`decide` holds the trading rules and is shared by the backtest and the live
paper-trading engine, so both behave identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockpredictor.costs import DEFAULT_COSTS, DeliveryCosts


@dataclass(frozen=True)
class Rules:
    n_hold: int = 5             # the 5 best stocks held, equal weight (of 10 candidates)
    exit_rank: int = 50         # sell when a holding drops below this rank
    stop_loss: float = 0.15     # sell when price falls this far below entry
    cooldown_days: int = 7      # don't re-buy a stopped-out stock for this long
    news_exit: bool = True      # sell on strongly negative news
    max_per_sector: int = 0     # at most this many holdings from one sector (0 = no limit)
    vol_sizing: bool = False    # size positions by volatility (calmer stocks get more)
    regime: str = "off"         # when Nifty 50 is below its 200-day average (a falling
                                # market): "no_buys", "exit" (sell all, hold cash) or
                                # "half" (half-size new positions); "off" = ignore


@dataclass
class Position:
    qty: int
    entry_price: float
    entry_date: pd.Timestamp


def decide(holdings: dict[str, Position], scores: pd.Series, prices: dict[str, float],
           rules: Rules, rebalance: bool, blocked: set[str] = frozenset(),
           negative_news: set[str] = frozenset(),
           severe_news: set[str] = frozenset(),
           side: str = "long", sectors: dict | None = None) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (sells [(symbol, reason)], buys [symbols in priority order]).

    negative_news: never buy these. severe_news: also sell them if held.
    """
    ranks = scores.rank(ascending=False, method="first")
    sells = []
    for sym, pos in holdings.items():
        price = prices.get(sym)
        stopped = price is not None and (
            price <= pos.entry_price * (1 - rules.stop_loss) if side == "long"
            else price >= pos.entry_price * (1 + rules.stop_loss))
        if stopped:
            sells.append((sym, "stop-loss"))
        elif rules.news_exit and sym in severe_news:
            sells.append((sym, "negative news"))
        elif rebalance and ranks.get(sym, math.inf) > rules.exit_rank:
            sells.append((sym, "rank dropped"))

    buys = []
    if rebalance:
        sold = {s for s, _ in sells}
        slots = rules.n_hold - (len(holdings) - len(sold))
        per_sector: dict = {}
        if rules.max_per_sector and sectors:
            for s in holdings:
                if s not in sold:
                    per_sector[sectors.get(s)] = per_sector.get(sectors.get(s), 0) + 1
        # With a sector limit, look a little further down the list for a stock from
        # another sector (up to 2x the holdings).
        depth = rules.n_hold * (2 if rules.max_per_sector and sectors else 1)
        for sym in ranks.sort_values().index:
            if slots <= 0 or ranks[sym] > depth:
                break
            if sym in holdings or sym in blocked or sym in negative_news or sym not in prices:
                continue
            sec = sectors.get(sym) if sectors else None
            if rules.max_per_sector and sectors and per_sector.get(sec, 0) >= rules.max_per_sector:
                continue
            buys.append(sym)
            per_sector[sec] = per_sector.get(sec, 0) + 1
            slots -= 1
    return sells, buys


@dataclass
class SimResult:
    equity: pd.Series
    trades: pd.DataFrame
    total_costs: float
    metrics: dict = field(default_factory=dict)


def simulate(scores: pd.DataFrame, close: pd.DataFrame, rules: Rules = Rules(),
             capital: float = 100_000, rebalance: str = "daily",
             costs: DeliveryCosts = DEFAULT_COSTS, lag: int = 1,
             sectors: dict | None = None, risk_off: set | None = None) -> SimResult:
    """scores: symbol/date/score rows. close: date x symbol close prices.

    lag=1: scores computed after a day's close are traded on the next day's
    close, so no trade uses information it could not have had.
    """
    raw = {d: g.set_index("symbol")["score"] for d, g in scores.groupby("date")}
    all_dates = [d for d in close.index if d >= min(raw)]
    score_by_date = {all_dates[i]: raw[all_dates[i - lag]]
                     for i in range(lag, len(all_dates)) if all_dates[i - lag] in raw}
    dates = [d for d in all_dates if d in score_by_date]
    week = pd.Series(dates).dt.to_period("W-FRI")
    rebalance_days = set(dates) if rebalance == "daily" else \
        set(pd.Series(dates).groupby(week).max())

    vol = close.pct_change().rolling(63, min_periods=20).std() if rules.vol_sizing else None
    cash, holdings, cooldown = capital, {}, {}
    equity, trades, total_costs = {}, [], 0.0
    last_price: dict[str, float] = {}

    for d in dates:
        row = close.loc[d]
        last_price.update(row.dropna().to_dict())
        blocked = {s for s, until in cooldown.items() if d < until}
        sells, buys = decide(holdings, score_by_date[d], last_price, rules,
                             rebalance=d in rebalance_days, blocked=blocked, sectors=sectors)
        off = rules.regime != "off" and risk_off is not None and d in risk_off
        if off and rules.regime in ("no_buys", "exit"):
            buys = []
        if off and rules.regime == "exit":
            sold = {s for s, _ in sells}
            sells = sells + [(s, "risk-off") for s in holdings if s not in sold]
        for sym, reason in sells:
            pos, price = holdings.pop(sym), last_price[sym]
            value = pos.qty * price
            c = costs.cost("sell", value)
            cash += value - c
            total_costs += c
            trades.append(dict(date=d, symbol=sym, side="sell", qty=pos.qty, price=price,
                               cost=c, reason=reason, entry_date=pos.entry_date,
                               ret=price / pos.entry_price - 1))
            if reason == "stop-loss":
                cooldown[sym] = d + pd.Timedelta(days=rules.cooldown_days)

        equity_now = cash + sum(p.qty * last_price[s] for s, p in holdings.items())
        slot = equity_now / rules.n_hold
        for sym in buys:
            price = last_price[sym]
            size = slot * vol_scale(vol, d, sym) if vol is not None else slot
            if off and rules.regime == "half":
                size *= 0.5
            qty = int(min(size, cash) / (price * 1.003))
            if qty <= 0:
                continue
            c = costs.cost("buy", qty * price)
            cash -= qty * price + c
            total_costs += c
            holdings[sym] = Position(qty, price, d)
            trades.append(dict(date=d, symbol=sym, side="buy", qty=qty, price=price, cost=c,
                               reason="top rank", entry_date=d, ret=np.nan))

        equity[d] = cash + sum(p.qty * last_price[s] for s, p in holdings.items())

    return SimResult(pd.Series(equity, name="equity"), pd.DataFrame(trades), total_costs)


def risk_off_days(indices: pd.DataFrame, index: str = "NIFTY50", window: int = 200) -> set:
    """Days on which the PREVIOUS close of `index` was below its `window`-day average
    (known before that day's trading: no look-ahead)."""
    n = indices[indices["symbol"] == index].set_index("date")["close"].sort_index()
    below = (n < n.rolling(window).mean()).shift(1).fillna(False).astype(bool)
    return set(below[below].index)


def vol_scale(vol: pd.DataFrame, d, sym: str) -> float:
    """Position size factor: the day's median volatility / this stock's (0.5x .. 1.5x)."""
    try:
        row = vol.loc[d]
        v, med = row[sym], row.median()
    except KeyError:
        return 1.0
    if not (v > 0 and med > 0):
        return 1.0
    return float(min(1.5, max(0.5, med / v)))


# --- Metrics -----------------------------------------------------------------------

def performance(equity: pd.Series) -> dict:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    dd = (equity / equity.cummax() - 1).min()
    return {"cagr": cagr, "volatility": vol, "sharpe": cagr / vol if vol else np.nan,
            "max_drawdown": dd, "total_return": equity.iloc[-1] / equity.iloc[0] - 1}


def yearly_returns(equity: pd.Series) -> pd.Series:
    last = equity.groupby(equity.index.year).last()
    first = equity.iloc[0]
    return last / last.shift(1).fillna(first) - 1


def trade_stats(trades: pd.DataFrame, equity: pd.Series) -> dict:
    if trades.empty:
        return {}
    sells = trades[trades["side"] == "sell"]
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    buys_value = (trades.loc[trades["side"] == "buy", "qty"]
                  * trades.loc[trades["side"] == "buy", "price"]).sum()
    return {
        "trades_per_year": len(trades) / years,
        "win_rate": (sells["ret"] > 0).mean() if len(sells) else np.nan,
        "avg_hold_days": (sells["date"] - sells["entry_date"]).dt.days.mean(),
        "turnover_per_year": buys_value / equity.mean() / years,
        "exit_reasons": sells["reason"].value_counts().to_dict(),
    }


def equal_weight_benchmark(close: pd.DataFrame, dates) -> pd.Series:
    rets = close.pct_change(fill_method=None).loc[dates].mean(axis=1).fillna(0)
    rets.iloc[0] = 0
    return (1 + rets).cumprod()
