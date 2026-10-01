"""Indicators. All use only data up to and including the current row."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.DataFrame | pd.Series, n: int):
    return x.rolling(n, min_periods=n).mean()


def rsi(close: pd.DataFrame | pd.Series, n: int):
    """Wilder's RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def atr(high, low, close, n: int):
    """Wilder's average true range."""
    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def prior_high(high, n: int):
    """Highest high of the n days BEFORE today (today excluded)."""
    return high.rolling(n, min_periods=n).max().shift(1)


def prior_low(low, n: int):
    return low.rolling(n, min_periods=n).min().shift(1)


def month_start(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the first trading day of each month. Uses only past dates, so it
    is the same whether or not later data exists (month ENDS would need tomorrow)."""
    m = index.month.to_numpy()
    out = np.zeros(len(index), dtype=bool)
    out[1:] = m[1:] != m[:-1]
    return out


def eligibility(md, names: list[str], p: dict) -> pd.DataFrame:
    """Which names a strategy may pick each day. With dated_universe, a stock
    counts only from its S&P 500 join date; otherwise every name always counts."""
    if not p.get("dated_universe"):
        return pd.DataFrame(True, index=md.calendar, columns=names)
    return md.eligible(names)
