"""Phase 5: the locked holdout, run exactly once.

Settings come from results/phase4/final_settings.json, written before the
holdout was ever loaded. Two settings per strategy, both fixed in advance:
the paper defaults, and the combo with the best development Sharpe.
Nothing here is tuned. Whatever comes out is reported as is.
"""
from __future__ import annotations

import json

import pandas as pd

from qm.config import results_dir
from qm.engine.splits import claim_holdout
from qm.research.common import (banner, dsr_for, fmt_table, log_run, load_market, research_view,
                                row_from, run_window, trial_log)
from qm.strategies import build_targets, uses_single_stocks


def run_holdout(cfg: dict, confirm: bool) -> pd.DataFrame:
    if not confirm:
        raise SystemExit("The holdout can be used once. Re-run with --holdout to confirm.")
    fs = results_dir(cfg) / "phase4" / "final_settings.json"
    if not fs.exists():
        raise SystemExit("Run Phase 4 first. Final settings must be fixed before the holdout.")
    final = json.loads(fs.read_text())
    claim_holdout(cfg, final)  # refuses if already used
    md, splits = research_view(load_market(cfg), cfg, holdout=True)
    log = trial_log(cfg)
    out = results_dir(cfg) / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, sets in final.items():
        seen = set()
        for label in ("default", "wf_pick"):
            p = dict(sets[label], financial_sectors=cfg["universe"]["financial_sectors"])
            key = json.dumps(sets[label], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            tg = build_targets(md, p)
            run = run_window(md, cfg, tg, splits.holdout_start, splits.snapshot_end, p if uses_single_stocks(p) else False)
            if not run:
                print(f"{name} ({label}): no trades in the holdout")
                continue
            log_run(log, "phase5", name, p, "holdout", run)
            rows.append(row_from(f"{name} ({label})", run, cfg,
                                 {"base": name, "settings": label, **dsr_for(log, name, run["m_on"])}))
            pd.DataFrame({"strategy": run["on"].returns, "spy": run["bench"].returns}).to_csv(
                out / f"{name}_{label}_returns.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    for line in banner(cfg, list(final)):
        print(line)
    print(f"\nHOLDOUT {splits.holdout_start.date()} to {splits.snapshot_end.date()}. Run once. Not tuned.\n")
    if not df.empty:
        print(fmt_table(df, ["strategy", "cagr", "cagr_no_costs", "sharpe", "dsr_own", "max_dd",
                             "spy_cagr", "gap_cagr_vs_spy"]))
    return df
