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
