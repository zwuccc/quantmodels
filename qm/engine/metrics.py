"""Performance numbers from a BacktestResult."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from qm.engine.backtest import BacktestResult

DAYS = 252


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    r = r.dropna()
    ex = r - ((1 + rf) ** (1 / DAYS) - 1)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(DAYS)) if sd > 1e-12 else 0.0


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    growth = float((1 + r).prod())
    years = len(r) / DAYS
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else -1.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def by_year(r: pd.Series) -> pd.Series:
    return (1 + r.dropna()).groupby(r.dropna().index.year).prod() - 1


def smoothness_r2(r: pd.Series) -> float:
    """R squared of log equity against time. Near 1 means a suspiciously straight line."""
    le = np.log((1 + r.fillna(0)).cumprod().to_numpy())
    if len(le) < 3 or np.std(le) == 0:
        return 0.0
    x = np.arange(len(le))
    return float(np.corrcoef(x, le)[0, 1] ** 2)


def summarize(res: BacktestResult, rf: float = 0.0) -> dict:
    r = res.returns.dropna()
    yrs = by_year(r)
    closed = res.trades[~res.trades["open"]] if len(res.trades) else res.trades
    n_years = len(r) / DAYS
    return {
        "start": r.index[0].date().isoformat(), "end": r.index[-1].date().isoformat(),
        "days": len(r),
        "total_return": float((1 + r).prod() - 1),
        "cagr": cagr(r),
        "vol": float(r.std(ddof=1) * np.sqrt(DAYS)),
        "sharpe": sharpe(r, rf),
        "sharpe_daily": sharpe(r, rf) / np.sqrt(DAYS),
        "skew": float(st.skew(r)) if len(r) > 2 else 0.0,
        "kurt": float(st.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0,
        "max_dd": max_drawdown(r),
        "worst_year": float(yrs.min()) if len(yrs) else np.nan,
        "worst_year_label": int(yrs.idxmin()) if len(yrs) else None,
        "turnover": float(res.traded.sum() / 2 / n_years) if n_years > 0 else np.nan,
        "trades": int(len(closed)),
        "win_rate": float((closed["ret"] > 0).mean()) if len(closed) else np.nan,
        "avg_hold_days": float(closed["days"].mean()) if len(closed) else np.nan,
        "exposure": float(res.exposure.mean()),
        "smooth_r2": smoothness_r2(r),
        "costs_paid_frac": float(res.costs.sum() / res.initial_capital),
    }
