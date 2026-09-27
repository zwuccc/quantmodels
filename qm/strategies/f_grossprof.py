"""F. Gross profitability (Novy Marx 2013). (Revenue - COGS) / total assets from
the latest annual filing known at the time. Hold the top slice, monthly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qm.data.fundamentals import annual_gross_profitability
from qm.data.market import MarketData
from qm.strategies.indicators import month_start


def grossprof(md: MarketData, p: dict) -> pd.DataFrame:
    sec = md.sector()
    fin = set(p.get("financial_sectors", ["Financials", "Real Estate"]))
    names = [t for t in md.tickers("stocks") if sec.get(t, "") not in fin]
    gp = annual_gross_profitability(md.fundamentals, tuple(p["gpa_bounds"]) if "gpa_bounds" in p else None)
    gp = gp[gp["ticker"].isin(names)]
    C = md.close[names]
    dates = C.index[month_start(C.index)]
    if gp.empty or len(dates) == 0:
        return pd.DataFrame(columns=names, dtype=float)
    grid = pd.DataFrame({"date": dates})
    stale = pd.Timedelta(days=p["max_staleness_days"])
    parts = []
    for t, g in gp.groupby("ticker"):
        m = pd.merge_asof(grid, g.sort_values("filed")[["filed", "gpa"]], left_on="date",
                          right_on="filed", direction="backward")
        m = m[(m["date"] - m["filed"]) <= stale]
        m["ticker"] = t
        parts.append(m)
    wide = pd.concat(parts).pivot(index="date", columns="ticker", values="gpa").reindex(index=dates, columns=names)
    rows, idx = [], []
    for d in dates:
        v = wide.loc[d]
        v = v[v.notna() & C.loc[d].notna()]
        if len(v) < p.get("min_names", 20):
            continue
        k = int(np.ceil(p["top_frac"] * len(v)))
        top = v.sort_values(ascending=False, kind="stable").index[:k]
        row = pd.Series(0.0, index=names)
        row[top] = 1.0 / k
        rows.append(row.to_numpy())
        idx.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=names)
