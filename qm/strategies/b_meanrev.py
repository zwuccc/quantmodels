"""B. Short term mean reversion: RSI(2) < 10 above the 200 day average.
Exit on a close above the 5 day average, or after max_hold days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qm.data.market import MarketData
from qm.strategies.indicators import rsi, sma


def meanrev(md: MarketData, p: dict) -> pd.DataFrame:
    names = md.tickers("stocks")
    C = md.close[names]
    R = rsi(C, p["rsi_len"]).to_numpy()
    T = sma(C, p["trend_sma"]).to_numpy()
    E = sma(C, p["exit_sma"]).to_numpy()
    c = C.to_numpy()
    n = len(names)
    held = np.zeros(n, bool)
    days = np.zeros(n, int)
    w = 1.0 / p["slots"]
    rows, idx = [], []
    for i in range(len(C)):
        row = np.full(n, np.nan)
        days[held] += 1
        with np.errstate(invalid="ignore"):  # a day with no price gives no signal
            out = held & ((c[i] > E[i]) | (days >= p["max_hold"]))
        row[out] = 0.0
        held[out] = False
        with np.errstate(invalid="ignore"):
            cand = ~held & ~out & (R[i] < p["rsi_entry"]) & (c[i] > T[i])
        free = p["slots"] - int(held.sum())
        if cand.any() and free > 0:
            pick = np.where(cand)[0]
            pick = pick[np.argsort(R[i][pick], kind="stable")][:free]
            row[pick] = w
            held[pick] = True
            days[pick] = 0
        if not np.isnan(row).all():
            rows.append(row)
            idx.append(C.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=names)
