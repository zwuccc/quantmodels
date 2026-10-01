"""Phase 3: every strategy with its default settings, in sample years only."""
from __future__ import annotations

import pandas as pd

from qm.config import results_dir
from qm.engine.leakcheck import LookAheadError, check_no_lookahead
from qm.engine.metrics import by_year
from qm.research.common import (ORDER, banner, fmt_table, log_run, load_market, params_for,
                                research_view, row_from, run_window, trial_log)
from qm.strategies import build_targets, uses_single_stocks


def run_insample(cfg: dict, names: list[str] | None = None) -> pd.DataFrame:
    names = names or ORDER
    md, splits = research_view(load_market(cfg), cfg)
    view = md.clip(splits.insample_end)
    log = trial_log(cfg)
    out_dir = results_dir(cfg) / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, years = [], {}
    for name in names:
        p = params_for(cfg, name)
        fn = lambda m, p=p: build_targets(m, p)  # noqa: E731
        try:
            check_no_lookahead(fn, view, n_cuts=2)
            leak = "pass"
        except LookAheadError as e:
            leak = f"FAIL: {e}"
        tg = fn(view)
        run = run_window(view, cfg, tg, splits.data_start, splits.insample_end, p if uses_single_stocks(p) else False)
        if not run:
            print(f"{name}: no trades in sample (not enough data)")
            continue
        log_run(log, "phase3", name, p, "insample", run)
        rows.append(row_from(name, run, cfg, {"leak_check": leak}))
        years[name] = by_year(run["on"].returns)
        years.setdefault("SPY", by_year(run["bench"].returns))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(years).to_csv(out_dir / "by_year.csv")
    for line in banner(cfg, names):
        print(line)
    print(f"\nIn sample {splits.data_start.date()} to {splits.insample_end.date()}, costs on "
          f"({cfg['costs']['commission_bps']}+{cfg['costs']['slippage_bps']} bps per side)\n")
    print(fmt_table(df, ["strategy", "start", "cagr", "cagr_no_costs", "sharpe", "max_dd", "turnover",
                         "win_rate", "spy_cagr", "gap_cagr_vs_spy", "ew_universe_cagr", "leak_check"]))
    flagged = df[df["flags"] != ""]
    for _, r in flagged.iterrows():
        print(f"TOO GOOD, check for bugs: {r['strategy']}: {r['flags']}")
    return df
