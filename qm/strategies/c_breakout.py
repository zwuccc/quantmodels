"""C. Donchian breakout (Turtle style). Enter on a close above the prior 55 day
high, exit on a close below the prior 20 day low. Size so 1 ATR move = 1% of equity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qm.data.market import MarketData
from qm.strategies.indicators import atr, prior_high, prior_low


def breakout(md: MarketData, p: dict) -> pd.DataFrame:
    names = md.tickers(p["universe"])
    C = md.close[names]
    H = prior_high(md.high[names], p["entry"]).to_numpy()
    L = prior_low(md.low[names], p["exit"]).to_numpy()
    A = atr(md.high[names], md.low[names], C, p["atr"]).to_numpy()
    c = C.to_numpy()
    n = len(names)
    held = np.zeros(n, bool)
    size = np.zeros(n)
    rows, idx = [], []
    for i in range(len(C)):
        row = np.full(n, np.nan)
        gone = held & np.isnan(c[i])
        with np.errstate(invalid="ignore"):
            ex = held & ~gone & (c[i] < L[i])
            cand = ~held & (c[i] > H[i]) & (A[i] > 0)
        out = ex | gone
        row[out] = 0.0
        held[out] = False
        size[out] = 0.0
        if cand.any():
            pick = np.where(cand)[0]
            strength = c[i][pick] / H[i][pick]
            used = size.sum()
            for j in pick[np.argsort(-strength, kind="stable")]:
                wj = min(p["risk_per_trade"] * c[i][j] / A[i][j], p["max_weight"])
                if used + wj > 1.0 + 1e-12:
                    continue
                row[j] = wj
                held[j] = True
                size[j] = wj
                used += wj
        if not np.isnan(row).all():
            rows.append(row)
            idx.append(C.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=names)
