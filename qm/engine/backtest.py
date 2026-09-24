"""The shared daily backtest engine.

Targets are decided from data up to the close of day t. This engine is the
only place that moves them to the open of day t+1. Strategies never shift.

Target values:
    number -> hold this fraction of equity (measured at the open)
    0      -> exit
    NaN    -> leave this position alone

Long only, no leverage. If buys need more cash than there is, all buys that
day are scaled down by the same factor. Sells happen before buys.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    traded: pd.Series          # traded value / equity at open (two sided), per day
    costs: pd.Series           # money paid in costs, per day
    exposure: pd.Series        # invested fraction at the close
    final_weights: pd.Series   # weights at the last close
    pending: pd.Series         # targets from the last close, not yet filled
    skipped: list = field(default_factory=list)
    initial_capital: float = 1.0


TRADE_COLS = ["ticker", "entry_date", "exit_date", "invested", "proceeds", "ret", "days", "open"]


def run_backtest(open_: pd.DataFrame, close: pd.DataFrame, targets: pd.DataFrame,
                 cost_rate: float, initial_capital: float = 1.0, cash_rate: float = 0.0,
                 start=None, end=None) -> BacktestResult:
    dates = close.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) < 2:
        raise ValueError("need at least 2 dates")
    tickers = list(targets.columns)
    unknown = set(tickers) - set(close.columns)
    if unknown:
        raise KeyError(f"targets name tickers with no prices: {sorted(unknown)[:5]}")

    O = open_.reindex(index=dates, columns=tickers).to_numpy(float)
    C = close.reindex(index=dates, columns=tickers).to_numpy(float)
    T = targets.reindex(index=dates, columns=tickers).to_numpy(float)
    if np.nanmin(np.append(T.ravel(), 0.0)) < 0:
        raise ValueError("negative target weight: long only")

    n_days, n = len(dates), len(tickers)
    shares = np.zeros(n)
    last_px = np.full(n, np.nan)
    cash = float(initial_capital)
    daily_cash = (1.0 + cash_rate) ** (1 / 252) - 1.0
    invested = np.zeros(n)
    proceeds = np.zeros(n)
    entry_idx = np.full(n, -1)

    eq = np.empty(n_days)
    traded = np.zeros(n_days)
    costs = np.zeros(n_days)
    expo = np.zeros(n_days)
    trades, skipped = [], []

    def close_lot(j: int, i: int) -> None:
        trades.append((tickers[j], dates[entry_idx[j]], dates[i], invested[j], proceeds[j],
                       proceeds[j] / invested[j] - 1.0 if invested[j] > 0 else np.nan,
                       i - entry_idx[j], False))
        invested[j] = proceeds[j] = 0.0
        entry_idx[j] = -1

    for i in range(n_days):
        if i > 0:
            cash *= 1.0 + daily_cash
            o = O[i]
            held = shares > 0
            # data ended for a held name: close at its last close, and log it
            gone = held & np.isnan(o) & np.isnan(C[i])
            for j in np.where(gone)[0]:
                value = shares[j] * last_px[j]
                cost = value * cost_rate
                cash += value - cost
                proceeds[j] += value - cost
                costs[i] += cost
                shares[j] = 0.0
                skipped.append((dates[i], tickers[j], "data ended while held, closed at last close"))
                close_lot(j, i)

            tgt = T[i - 1]
            act = ~np.isnan(tgt)
            if act.any():
                px = np.where(np.isnan(o), last_px, o)
                eq_open = cash + np.nansum(shares * px)
                can = act & ~np.isnan(o)
                for j in np.where(act & np.isnan(o))[0]:
                    if tgt[j] > 0 or shares[j] > 0:
                        skipped.append((dates[i], tickers[j], "no open price, trade skipped"))
                delta = np.zeros(n)
                delta[can] = tgt[can] * eq_open - shares[can] * o[can]

                sell = delta < 0
                if sell.any():
                    val = -delta[sell]
                    c = val * cost_rate
                    cash += float(np.sum(val - c))
                    costs[i] += float(np.sum(c))
                    traded[i] += float(np.sum(val))
                    proceeds[sell] += val - c
                    exit_all = sell & (tgt == 0)
                    shares[sell] += delta[sell] / o[sell]
                    shares[exit_all] = 0.0

                buy = delta > 0
                if buy.any():
                    need = float(np.sum(delta[buy])) * (1.0 + cost_rate)
                    if need > cash:
                        delta[buy] *= max(cash, 0.0) / need
                    val = delta[buy]
                    c = val * cost_rate
                    cash -= float(np.sum(val + c))
                    costs[i] += float(np.sum(c))
                    traded[i] += float(np.sum(val))
                    invested[buy] += val + c
                    shares[buy] += val / o[buy]
                    new = buy & (entry_idx < 0) & (delta > 0)
                    entry_idx[new] = i
                if abs(cash) < 1e-9 * max(eq_open, 1.0):
                    cash = 0.0
                traded[i] /= eq_open if eq_open > 0 else 1.0

                for j in np.where(sell & (shares <= 1e-12) & (entry_idx >= 0))[0]:
                    shares[j] = 0.0
                    close_lot(j, i)

        ok = ~np.isnan(C[i])
        last_px[ok] = C[i][ok]
        pos_val = float(np.nansum(shares * last_px))
        eq[i] = cash + pos_val
        expo[i] = pos_val / eq[i] if eq[i] > 0 else 0.0

    for j in np.where(entry_idx >= 0)[0]:
        mark = shares[j] * last_px[j]
        trades.append((tickers[j], dates[entry_idx[j]], pd.NaT, invested[j], proceeds[j] + mark,
                       (proceeds[j] + mark) / invested[j] - 1.0 if invested[j] > 0 else np.nan,
                       n_days - 1 - entry_idx[j], True))

    equity = pd.Series(eq, index=dates, name="equity")
    rets = equity.pct_change()
    rets.iloc[0] = equity.iloc[0] / initial_capital - 1.0
    w = pd.Series(shares * np.nan_to_num(last_px) / eq[-1], index=tickers)
    return BacktestResult(
        equity=equity, returns=rets.rename("ret"),
        trades=pd.DataFrame(trades, columns=TRADE_COLS),
        traded=pd.Series(traded, index=dates), costs=pd.Series(costs, index=dates),
        exposure=pd.Series(expo, index=dates), final_weights=w[w > 0],
        pending=pd.Series(T[-1], index=tickers).dropna(), skipped=skipped,
        initial_capital=initial_capital,
    )


def buy_and_hold_targets(close: pd.DataFrame, ticker: str, start) -> pd.DataFrame:
    """Targets for buying one ticker at the first open after start and holding."""
    first = close.index[close.index >= pd.Timestamp(start)][0]
    return pd.DataFrame({ticker: [1.0]}, index=[first])
