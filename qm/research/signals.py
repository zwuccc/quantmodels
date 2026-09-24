"""Phase 6: what each strategy would hold as of the latest close. Paper only.

No orders, no broker. This is a list for reading.
"""
from __future__ import annotations

import json

import pandas as pd

from qm.config import results_dir
from qm.engine.splits import holdout_used
from qm.research.common import banner, load_market, research_view, run_window
from qm.strategies import build_targets


def latest_signals(cfg: dict, settings: str = "default") -> pd.DataFrame:
    if not holdout_used(cfg):
        raise SystemExit("Run Phase 5 first. Signals use the latest data, which is inside the holdout.")
    final = json.loads((results_dir(cfg) / "phase4" / "final_settings.json").read_text())
    md, splits = research_view(load_market(cfg), cfg, holdout=True)
    last = md.calendar[-1]
    rows = []
    for name, sets in final.items():
        p = dict(sets[settings], financial_sectors=cfg["universe"]["financial_sectors"])
        tg = build_targets(md, p)
        # run a year so positions from recent signals are in place
        run = run_window(md, cfg, tg, md.calendar[max(0, len(md.calendar) - 260)], last)
        if not run:
            continue
        res = run["on"]
        held = res.final_weights
        pend = res.pending
        for t in sorted(set(held.index) | set(pend.index)):
            w_now = float(held.get(t, 0.0))
            w_next = pend.get(t)
            if pd.isna(w_next):
                action = "hold" if w_now > 0 else None
            elif w_next == 0:
                action = "sell at next open" if w_now > 0 else None
            else:
                action = "buy at next open" if w_now == 0 else "rebalance at next open"
            if action:
                rows.append({"strategy": name, "ticker": t, "weight_now": round(w_now, 4),
                             "target_next_open": None if pd.isna(w_next) else round(float(w_next), 4),
                             "action": action})
    df = pd.DataFrame(rows, columns=["strategy", "ticker", "weight_now", "target_next_open", "action"])
    out = results_dir(cfg) / "signals_latest.csv"
    df.to_csv(out, index=False)
    for line in banner(cfg, list(final)):
        print(line)
    print(f"\nPaper signals as of the close of {last.date()} ({settings} settings). Not advice. No orders.\n")
    for name, g in df.groupby("strategy", sort=False):
        print(f"{name}: " + ", ".join(f"{r.ticker} {r.action}" for r in g.itertuples()))
    print(f"\nFull list: {out}")
    return df
