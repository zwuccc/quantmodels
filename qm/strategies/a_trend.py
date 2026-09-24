"""A. Moving average trend on ETFs. Hold the asset above its average, IEF below."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qm.data.market import MarketData
from qm.strategies.indicators import month_start, sma


def _emit_on_change(weights: pd.DataFrame) -> pd.DataFrame:
    """Only send targets on days the wanted mix changes. Positions drift in between."""
    w = weights.dropna(how="any")
    changed = w.ne(w.shift(1)).any(axis=1)
    return w[changed]


def _state(on: pd.Series, ready: pd.Series, check: str, index) -> pd.Series:
    s = on.astype(float).where(ready)
    if check == "monthly":
        s = s.where(pd.Series(month_start(index), index=index)).ffill().where(ready)
    return s


def trend_sma(md: MarketData, p: dict) -> pd.DataFrame:
    c = md.close[p["asset"]]
    avg = sma(c, p["sma"])
    s = _state(c > avg, avg.notna() & md.close[p["safe"]].notna(), p.get("check", "daily"), md.calendar)
    return _emit_on_change(pd.DataFrame({p["asset"]: s, p["safe"]: 1 - s}))


def trend_cross(md: MarketData, p: dict) -> pd.DataFrame:
    c = md.close[p["asset"]]
    fast, slow = sma(c, p["fast"]), sma(c, p["slow"])
    s = _state(fast > slow, slow.notna() & fast.notna() & md.close[p["safe"]].notna(), p.get("check", "daily"), md.calendar)
    return _emit_on_change(pd.DataFrame({p["asset"]: s, p["safe"]: 1 - s}))


def trend_multi(md: MarketData, p: dict) -> pd.DataFrame:
    assets = [a for a in p["assets"] if a in md.close.columns]
    k = len(assets)
    c = md.close[assets]
    avg = sma(c, p["sma"])
    ready = avg.notna().all(axis=1) & md.close[p["safe"]].notna()
    on = (c > avg).astype(float).where(ready, np.nan)
    w = on / k
    w[p["safe"]] = (1 - on).sum(axis=1, min_count=k) / k
    return _emit_on_change(w)
