"""D. Cross sectional momentum (Jegadeesh and Titman 1993).
Rank by return from t-lookback to t-skip. Hold the top slice, equal weight,
rebalanced on the first trading day of each month."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qm.data.market import MarketData
from qm.strategies.indicators import eligibility, month_start


def momentum(md: MarketData, p: dict) -> pd.DataFrame:
    names = md.tickers("stocks")
    C = md.close[names]
    mom = C.shift(p["skip"]) / C.shift(p["lookback"]) - 1
    rebal = month_start(C.index)
    el = eligibility(md, names, p)
    rows, idx = [], []
    for i in np.where(rebal)[0]:
        m = mom.iloc[i]
        m = m[m.notna() & C.iloc[i].notna() & el.iloc[i]]
        if len(m) < p.get("min_names", 20):
            continue
        k = int(np.ceil(p["top_frac"] * len(m)))
        top = m.sort_values(ascending=False, kind="stable").index[:k]
        row = pd.Series(0.0, index=names)
        row[top] = 1.0 / k
        rows.append(row.to_numpy())
        idx.append(C.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=names)
