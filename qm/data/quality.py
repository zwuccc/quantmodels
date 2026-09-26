"""Price quality checks. A failed check means the ticker is skipped, not repaired."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


def check_prices(df: pd.DataFrame, calendar: pd.DatetimeIndex | None, q: dict) -> str | None:
    """Return None if the series is usable, else a short reason to skip it.

    calendar is the benchmark's trading days. A day inside the ticker's own
    first..last range that the benchmark traded but the ticker has no close
    for counts as an internal gap.
    """
    if df is None or df.empty:
        return "empty"
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        return f"missing columns {missing}"
    if df.index.has_duplicates:
        return "duplicate dates"
    if not df.index.is_monotonic_increasing:
        return "dates not sorted"
    px = df[["open", "high", "low", "close"]]
    if len(df) < q["min_history_days"]:
        return f"only {len(df)} rows"
    if px.isna().any().any():
        return f"{int(px.isna().any(axis=1).sum())} rows with NaN prices"
    if (px <= 0).any().any():
        return "non positive prices"
    if calendar is not None:
        span = calendar[(calendar >= df.index[0]) & (calendar <= df.index[-1])]
        gaps = len(span.difference(df.index))
        if gaps > q["max_internal_gaps"]:
            return f"{gaps} internal gap days"
    r = df["close"].pct_change().to_numpy()
    big = np.abs(r) > q["bad_jump"]
    if big.any():
        idx = np.where(big)[0]
        for i in idx:
            if i + 1 < len(r) and np.sign(r[i + 1]) == -np.sign(r[i]) and abs(r[i + 1]) > q["bad_jump_reversal"]:
                return f"jump and reversal at {df.index[i].date()} (likely bad adjustment)"
    return None


def stale_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where Yahoo filled a missing day: zero volume and the close copied from the day before."""
    return (df["volume"] == 0) & (df["close"] == df["close"].shift(1))


def trim_leading_stale(df: pd.DataFrame, min_clean_run: int) -> tuple[pd.DataFrame, int, int]:
    """Drop a fake history at the start (years of copied prices before a US listing).

    Real history starts at the first run of min_clean_run days with no filled
    rows. Returns (trimmed frame, filled rows left after that start, rows cut).
    Filled rows after the start are gaps inside real trading.
    """
    st = stale_mask(df).to_numpy()
    if not st.any():
        return df, 0, 0
    run = 0
    for i, s in enumerate(st):
        run = 0 if s else run + 1
        if run == min_clean_run:
            start = i - min_clean_run + 1
            return df.iloc[start:], int(st[start:].sum()), start
    return df.iloc[0:0], 0, len(df)


def check_adjustments(adj: pd.DataFrame, unadj: pd.DataFrame | None, q: dict) -> str | None:
    """Catch broken Yahoo adjustments around spin offs and odd splits.

    On a day Yahoo books a split, or a distribution bigger than big_distribution
    of the price (how it records spin offs), a correct adjustment leaves a normal
    sized move. An adjusted move above max_event_move that day means the
    adjustment is broken (e.g. DHR on the 2016 Fortive spin off: +61%).
    """
    if unadj is None or unadj.empty:
        return "no unadjusted data to audit the adjustments"
    u = unadj.reindex(adj.index)
    prev = u["close_raw"].shift(1)
    event = ((u["dividends"] / prev) > q["big_distribution"]) | (u["splits"].fillna(0).ne(0) & u["splits"].fillna(0).ne(1))
    moves = adj["close"].pct_change()
    bad = event.fillna(False) & (moves.abs() > q["max_event_move"])
    if bad.any():
        d = bad.index[bad.to_numpy()][0]
        return f"broken adjustment on {d.date()}: adjusted move {moves[d]:+.0%} on a split or spin off day"
    return None
