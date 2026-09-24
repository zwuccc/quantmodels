"""Small synthetic market for tests. No network."""
import numpy as np
import pandas as pd

from qm.data.fundamentals import PIT_COLS, Fundamentals
from qm.data.market import MarketData


def random_market(n_days=600, tickers=("SPY", "IEF", "AAA", "BBB", "CCC"), seed=1, start="2010-01-04"):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n_days)
    r = rng.normal(0.0003, 0.012, size=(n_days, len(tickers)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=d, columns=list(tickers))
    gap = np.exp(rng.normal(0, 0.003, size=close.shape))
    open_ = close.shift(1).fillna(close.iloc[0]) * gap
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, size=close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, size=close.shape)))
    vol = pd.DataFrame(1e6, index=d, columns=close.columns)
    kinds = ["etf" if t in ("SPY", "IEF") else "stock" for t in tickers]
    uni = pd.DataFrame({"ticker": list(tickers), "sector": ["ETF" if k == "etf" else "Tech" for k in kinds], "kind": kinds})
    return MarketData(open_, high, low, close, vol, Fundamentals(pd.DataFrame(columns=PIT_COLS)), uni)
