"""Look ahead check by truncation.

Build targets on the full data, then again on data cut off at date T. If the
strategy only uses the past, the targets up to T must match exactly. Any
difference means the strategy peeked at data after T.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from qm.data.market import MarketData


class LookAheadError(AssertionError):
    pass


def _same(a: pd.DataFrame, b: pd.DataFrame, tol: float) -> tuple[bool, str]:
    idx = a.index.union(b.index)
    cols = a.columns.union(b.columns)
    x = a.reindex(index=idx, columns=cols).to_numpy(float)
    y = b.reindex(index=idx, columns=cols).to_numpy(float)
    both_nan = np.isnan(x) & np.isnan(y)
    diff = ~both_nan & ~(np.abs(x - y) <= tol)
    if diff.any():
        r, c = np.argwhere(diff)[0]
        return False, f"first mismatch {idx[r].date()} {cols[c]}: full={x[r, c]} cut={y[r, c]}"
    return True, ""


def check_no_lookahead(fn: Callable[[MarketData], pd.DataFrame], md: MarketData,
                       cuts: list | None = None, n_cuts: int = 3, seed: int = 0, tol: float = 1e-9) -> list:
    cal = md.calendar
    if cuts is None:
        rng = np.random.default_rng(seed)
        lo, hi = len(cal) // 4, len(cal) - 2
        cuts = [cal[i] for i in sorted(rng.choice(np.arange(lo, hi), size=min(n_cuts, hi - lo), replace=False))]
    full = fn(md)
    for cut in cuts:
        part = fn(md.clip(cut))
        ok, msg = _same(full.loc[:cut], part.loc[:cut], tol)
        if not ok:
            raise LookAheadError(f"targets up to {pd.Timestamp(cut).date()} change when later data is added. {msg}")
    return cuts
