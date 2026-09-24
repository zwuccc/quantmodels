"""Rule 9: results that look too good are treated as bugs until checked."""
from __future__ import annotations

import numpy as np


def too_good_flags(m: dict, cfg: dict) -> list[str]:
    s = cfg["sanity"]
    flags = []
    if m["sharpe"] > s["max_sharpe"]:
        flags.append(f"Sharpe {m['sharpe']:.2f} above {s['max_sharpe']}")
    if m.get("trades", 0) >= 30 and np.isfinite(m.get("win_rate", np.nan)) and m["win_rate"] > s["max_win_rate"]:
        flags.append(f"win rate {m['win_rate']:.0%} above {s['max_win_rate']:.0%}")
    if m["smooth_r2"] > s["max_smoothness_r2"] and m["days"] > 252 * 3 and m["cagr"] > 0.05:
        flags.append(f"equity curve very smooth (R2 {m['smooth_r2']:.3f})")
    if m["days"] > 252 * 5 and m["max_dd"] > -s["min_maxdd_long"] and m["exposure"] > 0.2:
        flags.append(f"max drawdown only {m['max_dd']:.1%} over {m['days'] // 252} years")
    return flags


BUG_CHECKLIST = [
    "look ahead truncation test passes for this strategy",
    "fills happen at the next open, not the signal day close",
    "costs are charged (cost on vs off numbers differ)",
    "no giant one day price jumps in held names (bad adjustment)",
    "gain not concentrated in a handful of trades",
    "survivorship: universe is today's members only",
]
