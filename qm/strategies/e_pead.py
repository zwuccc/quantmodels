"""E. Post earnings announcement drift (Bernard and Thomas 1989).
SUE from XBRL quarterly EPS. Buy top SUE names at the open after the filing
day, hold `hold` trading days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qm.data.fundamentals import sue_events
from qm.data.market import MarketData


def pead(md: MarketData, p: dict) -> pd.DataFrame:
    names = md.tickers("stocks")
    cal = md.calendar
    ev = sue_events(md.fundamentals, p["n_surprises"], p.get("min_sue_sd", 0.0))
    ev = ev[ev["ticker"].isin(names)]
    ev = ev[ev["sue"].abs() <= p.get("max_abs_sue", np.inf)]
    if p.get("dated_universe"):  # only stocks already in the S&P 500 when the filing came out
        added = pd.to_datetime(md.universe.set_index("ticker")["added"], errors="coerce")
        ev = ev[ev["filed"].to_numpy() >= added.reindex(ev["ticker"]).to_numpy("datetime64[ns]")]  # e.g. HIG 2014 Q4 EPS filed as 860000
    ev = ev.sort_values(["filed", "ticker"]).reset_index(drop=True)
    empty = pd.DataFrame(columns=names, dtype=float)
    if ev.empty:
        return empty

    # top slice cutoff from SUEs filed in the trailing window only (no future SUEs)
    win = np.timedelta64(p["rank_window_days"], "D")
    filed = ev["filed"].to_numpy("datetime64[ns]")
    sue = ev["sue"].to_numpy()
    top = np.zeros(len(ev), bool)
    for k in range(len(ev)):
        lo = np.searchsorted(filed, filed[k] - win, side="right")
        hi = np.searchsorted(filed, filed[k], side="right")
        pool = sue[lo:hi]
        if len(pool) >= p["min_rank_pool"]:
            top[k] = sue[k] >= np.quantile(pool, 1 - p["top_frac"])
    ev = ev[top].copy()
    # signal day: first trading day on or after the filed date; fill is the next open
    pos = np.searchsorted(cal.to_numpy("datetime64[ns]"), ev["filed"].to_numpy("datetime64[ns]"), side="left")
    ev = ev[pos < len(cal)]
    ev["i"] = pos[pos < len(cal)]
    by_day = {i: g.sort_values("sue", ascending=False) for i, g in ev.groupby("i")}

    col = {t: j for j, t in enumerate(names)}
    c = md.close[names].to_numpy()
    n = len(names)
    held = np.zeros(n, bool)
    days = np.zeros(n, int)
    w = 1.0 / p["max_positions"]
    rows, idx = [], []
    for i in range(len(cal)):
        row = np.full(n, np.nan)
        days[held] += 1
        out = held & (days >= p["hold"])  # engine waits for a real open if needed
        row[out] = 0.0
        held[out] = False
        if i in by_day:
            for t in by_day[i]["ticker"]:
                j = col[t]
                if np.isnan(c[i][j]):
                    continue
                if held[j]:
                    days[j] = 0  # new top surprise while held: restart the clock
                    continue
                if held.sum() >= p["max_positions"]:
                    break
                row[j] = w
                held[j] = True
                days[j] = 0
        if not np.isnan(row).all():
            rows.append(row)
            idx.append(cal[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=names)
